from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from dotenv import load_dotenv
from typing import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pypdf import PdfReader
import sqlite3
import json
import re

load_dotenv()

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=3000,
        temperature=0.2,
    )
)

MAX_REVISIONS = 3  # how many times the user can ask for a re-analysis


# ************************* Prompts *************************
PARSE_PROMPT = """You are a precise parser. Read the RESUME and the JOB DESCRIPTION.
Extract only what is explicitly written. Never infer or invent.

Return ONLY valid JSON:
{
  "role_title": "job title from the JD, plus level if stated",
  "jd_requirements": [{"skill": "requirement from the JD", "importance": "must-have | nice-to-have"}],
  "resume_facts": {
    "skills": ["..."],
    "projects": [{"name": "...", "tech": ["..."], "summary": "..."}],
    "experience": [{"company": "...", "role": "...", "summary": "..."}],
    "certifications": ["..."],
    "education": "degree, college, CGPA if written"
  }
}

RESUME:
<<RESUME>>

JOB DESCRIPTION:
<<JD>>
"""

ANALYZE_PROMPT = """You are an honest career coach. Using ONLY the parsed data below, compare the resume to the job.
If a requirement has no evidence in resume_facts, mark it Missing. Never invent skills or experience.

Feedback from the user on the previous report (may be None). Address every point, but still never invent facts:
<<FEEDBACK>>

Return ONLY valid JSON:
{
  "role_title": "same role title",
  "verdict_label": "Strong match | Moderate match | Weak match",
  "headline": "one punchy sentence summarizing the result",
  "summary": "2-3 sentences on what the resume shows versus the job",
  "match_score": 0-100 integer,
  "strength_bars": [{"label": "short category", "percent": 0-100}],
  "requirements": [{"requirement": "skill from the JD", "evidence": "what the resume shows, or Not mentioned", "status": "Match | Partial | Unclear | Missing"}],
  "strengths": [{"title": "strength", "detail": "why it helps, grounded in the resume"}],
  "gaps": [{"title": "gap", "detail": "why it matters and what to show"}]
}
Rules: 5-6 strength_bars, 12-18 requirements taken from the JD, 3-5 strengths, 3-5 gaps.

PARSED DATA:
<<PARSED>>
"""

ADVICE_PROMPT = """You are an honest career coach. Using the parsed data and the analysis below, write practical advice.
The suggested_summary must use only facts from resume_facts. Do not claim skills the resume lacks; mention willingness to learn instead.
Project ideas are suggestions only and must never be described as already built.

Feedback from the user on the previous report (may be None). Address every point:
<<FEEDBACK>>

Return ONLY valid JSON:
{
  "resume_changes": [{"section": "Summary | Skills | Experience | Projects | Certifications | Education", "change": "specific recommendation"}],
  "suggested_summary": "professional summary based only on resume facts",
  "summary_caveat": "one short line about claiming only what they can back up in an interview",
  "projects_intro": "2-3 sentences on how existing resume projects relate to this job",
  "project_ideas": [{"title": "project tailored to the JD", "tagline": "why it fits", "features": ["feature 1", "feature 2"]}],
  "projects_note": "short line saying these are suggested features, not features currently in the resume",
  "interview_intro": "one line introducing the topics",
  "interview_priorities": [{"priority": "High | Medium | Additional", "topics": "comma-separated topics"}],
  "interview_note": "one sentence about being ready to explain existing projects and internships",
  "final_label": "Strong fit | Some tailoring needed | Significant tailoring needed",
  "final_text": "2 sentences on where the resume stands against this JD",
  "top_priorities": ["priority 1", "priority 2", "priority 3"],
  "recommendation": "2-3 sentences of honest final advice"
}
Rules: one resume_changes row per section, 1-2 project_ideas, 5-8 interview_priorities ordered High, Medium, Additional, exactly 3 top_priorities.

PARSED DATA:
<<PARSED>>

ANALYSIS:
<<ANALYSIS>>
"""


# ************************* Helpers *************************
class ReportState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    parsed: dict
    analysis: dict
    advice: dict
    report: dict
    feedback: str
    revisions: int
    approved: bool


def extract_pdf_text(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def call_json(prompt):
    """Call the LLM and return a dict. Retries once if the JSON is broken."""
    for _ in range(2):
        response = llm.invoke(prompt)
        try:
            return parse_json(response.content)
        except (json.JSONDecodeError, ValueError):
            prompt += "\n\nYour last reply was not valid JSON. Return ONLY the JSON object."
    raise ValueError("The model did not return valid JSON. Please try again.")


# ************************* Nodes *************************
def parse_node(state: ReportState):
    prompt = (PARSE_PROMPT
              .replace("<<RESUME>>", state["resume_text"])
              .replace("<<JD>>", state["jd_text"]))
    return {"parsed": call_json(prompt), "feedback": "", "revisions": 0, "approved": False}


def analyze_node(state: ReportState):
    prompt = (ANALYZE_PROMPT
              .replace("<<FEEDBACK>>", state.get("feedback") or "None")
              .replace("<<PARSED>>", json.dumps(state["parsed"])))
    return {"analysis": call_json(prompt)}


def advice_node(state: ReportState):
    prompt = (ADVICE_PROMPT
              .replace("<<FEEDBACK>>", state.get("feedback") or "None")
              .replace("<<PARSED>>", json.dumps(state["parsed"]))
              .replace("<<ANALYSIS>>", json.dumps(state["analysis"])))
    advice = call_json(prompt)
    return {"advice": advice, "report": {**state["analysis"], **advice}}


def validation_node(state: ReportState):
    # The graph PAUSES here until the frontend resumes it with the user's decision.
    # Note: when resumed, this node runs again from the top, so keep it free of side effects.
    decision = interrupt("Waiting for the user to review the report")

    approved = bool(decision.get("approved"))
    return {
        "approved": approved,
        "feedback": "" if approved else decision.get("feedback", ""),
        "revisions": state.get("revisions", 0) + (0 if approved else 1),
    }


def route_after_review(state: ReportState):
    if state.get("approved") or state.get("revisions", 0) > MAX_REVISIONS:
        return "done"
    return "retry"


# ************************* Graph *************************
conn = sqlite3.connect(database="career.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ReportState)
graph.add_node("parse_node", parse_node)
graph.add_node("analyze_node", analyze_node)
graph.add_node("advice_node", advice_node)
graph.add_node("validation_node", validation_node)

graph.add_edge(START, "parse_node")
graph.add_edge("parse_node", "analyze_node")
graph.add_edge("analyze_node", "advice_node")
graph.add_edge("advice_node", "validation_node")
graph.add_conditional_edges(
    "validation_node",
    route_after_review,
    {"retry": "analyze_node", "done": END},
)

analyzer = graph.compile(checkpointer=checkpointer)