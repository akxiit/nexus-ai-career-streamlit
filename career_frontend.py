import html
import os
import sys
import uuid

import streamlit as st
from langgraph.types import Command

sys.path.insert(0, os.path.dirname(__file__))

from career_backend import analyzer, extract_pdf_text, MAX_REVISIONS

st.set_page_config(page_title="AI Career Coach", page_icon="🎯", layout="wide")

st.markdown("""<style>
.card{background:#1c1c1c;border:1.5px solid #333;border-radius:18px;padding:22px 26px;margin:10px 0 26px;color:#f2f2f2}
.card.green{border-color:#4a9d55}
.card.orange{border-color:#e8a03a}
.card.red{border-color:#d9534f}
.pill{display:inline-block;padding:4px 14px;border-radius:999px;font-size:.85rem;font-weight:600}
.pill.green{background:#16301b;color:#5fbf6b}
.pill.orange{background:#3a2616;color:#e8803a}
.pill.red{background:#3a1616;color:#f0605d}
.big{font-size:2.4rem;font-weight:700;line-height:1.2;margin:12px 0}
.score{font-size:3.4rem;font-weight:700}
.score.green{color:#5fbf6b}.score.orange{color:#e8803a}.score.red{color:#f0605d}
.muted{color:#9a9a9a;font-size:.9rem}
.row{display:flex;justify-content:space-between;margin-top:16px}
.bar{height:10px;background:#2a2a2a;border-radius:99px;overflow:hidden;margin-top:6px}
.fill{height:100%;border-radius:99px}
.fill.green{background:#4a9d55}.fill.orange{background:#e8803a}.fill.red{background:#d9534f}
table.t{width:100%;border-collapse:collapse;color:#f2f2f2}
table.t th{text-align:left;padding:12px 8px;border-bottom:1px solid #333}
table.t td{padding:12px 8px;border-bottom:1px solid #262626;vertical-align:top}
hr.sep{border:none;border-top:1px solid #333;margin:18px 0}
.pill.gray{background:#2a2a2a;color:#d0d0d0}
ul.feat{margin:14px 0 6px 22px;line-height:1.9}
ol.prio{margin:10px 0 0 22px;line-height:1.9}
</style>""", unsafe_allow_html=True)


# ************************* Helpers *************************
def esc(value):
    return html.escape(str(value))


def pct(value):
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 0


def level(score):
    return "green" if score >= 70 else "orange" if score >= 40 else "red"


STATUS_COLOR = {"match": "green", "partial": "orange", "unclear": "orange", "missing": "red"}


def hero_card(r, score):
    c = level(score)
    return (
        f'<div class="card {c}">'
        f'<span class="pill {c}">{esc(r.get("verdict_label", ""))}</span>'
        f'<div class="big">{esc(r.get("headline", ""))}</div>'
        f'<div>{esc(r.get("summary", ""))}</div>'
        f'<div class="muted" style="margin-top:10px">Assessment based on your uploaded resume and the supplied job description.</div>'
        f'</div>'
    )


def match_card(r, score):
    c = level(score)
    bars = ""
    for b in r.get("strength_bars", []):
        p = pct(b.get("percent"))
        bars += (
            f'<div class="row"><span>{esc(b.get("label", ""))}</span><span>{p}%</span></div>'
            f'<div class="bar"><div class="fill {level(p)}" style="width:{p}%"></div></div>'
        )
    return (
        f'<div class="card">'
        f'<h4 style="margin:0">Estimated skill alignment</h4>'
        f'<div class="score {c}">{score}%</div>'
        f'<div class="muted">Qualitative estimate, not an ATS-generated score.</div>'
        f'<hr class="sep"><h5>Your current strengths</h5>{bars}</div>'
    )


def requirements_table(r):
    rows = ""
    for item in r.get("requirements", []):
        status = str(item.get("status", "Unclear"))
        c = STATUS_COLOR.get(status.lower(), "orange")
        rows += (
            f'<tr><td>{esc(item.get("requirement", ""))}</td>'
            f'<td>{esc(item.get("evidence", ""))}</td>'
            f'<td><span class="pill {c}">{esc(status)}</span></td></tr>'
        )
    return (
        '<div class="card"><table class="t"><tr><th>Job requirement</th>'
        f'<th>Evidence in your resume</th><th>Match</th></tr>{rows}</table></div>'
    )


def list_card(items, pill_text, color):
    body = ""
    for i, item in enumerate(items, 1):
        body += (
            f'<div style="margin-top:14px"><b>{i}. {esc(item.get("title", ""))}</b>'
            f'<div>{esc(item.get("detail", ""))}</div></div>'
        )
    return f'<div class="card {color}"><span class="pill {color}">{pill_text}</span>{body}</div>'


def changes_table(r):
    rows = "".join(
        f'<tr><td><b>{esc(c.get("section", ""))}</b></td><td>{esc(c.get("change", ""))}</td></tr>'
        for c in r.get("resume_changes", [])
    )
    return f'<div class="card"><table class="t"><tr><th>Section</th><th>Recommended change</th></tr>{rows}</table></div>'

PRIORITY_COLOR = {"high": "red", "medium": "orange", "additional": "gray"}


def projects_cards(r):
    out = ""
    for p in r.get("project_ideas", []):
        feats = "".join(f"<li>{esc(f)}</li>" for f in p.get("features", []))
        out += (
            '<div class="card"><span class="pill green">Relevant project idea</span>'
            f'<div class="big" style="font-size:1.8rem">{esc(p.get("title", ""))}</div>'
            f'<div class="muted">{esc(p.get("tagline", ""))}</div><hr class="sep">'
            f'<ul class="feat">{feats}</ul>'
            f'<div class="muted">{esc(r.get("projects_note", ""))}</div></div>'
        )
    return out


def interview_table(r):
    rows = ""
    for item in r.get("interview_priorities", []):
        p = str(item.get("priority", "Additional"))
        c = PRIORITY_COLOR.get(p.lower(), "gray")
        rows += (
            f'<tr><td><span class="pill {c}">{esc(p)}</span></td>'
            f'<td>{esc(item.get("topics", ""))}</td></tr>'
        )
    return f'<div class="card"><table class="t"><tr><th>Priority</th><th>Topics</th></tr>{rows}</table></div>'


def final_card(r, score):
    c = level(score)
    items = "".join(f"<li>{esc(p)}</li>" for p in r.get("top_priorities", []))
    return (
        f'<div class="card {c}"><h3 style="margin:0">Your resume\'s position against this JD</h3>'
        f'<div style="margin:10px 0"><span class="pill {c}">{esc(r.get("final_label", ""))}</span></div>'
        f'<div>{esc(r.get("final_text", ""))}</div><hr class="sep">'
        f'<b>Three things to prioritize:</b><ol class="prio">{items}</ol></div>'
    )

# ************************* Graph helpers *************************
def graph_config():
    return {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "run_name": "career_coach_report",
        "tags": ["career-coach", "streamlit"],
        "metadata": {"app": "nexus-career-coach"},
    }


def load_state():
    """Read the latest saved graph state into session_state."""
    snap = analyzer.get_state(graph_config())
    st.session_state["report"] = snap.values.get("report")
    st.session_state["revisions"] = snap.values.get("revisions", 0)
    st.session_state["approved"] = snap.values.get("approved", False)
    st.session_state["awaiting"] = bool(snap.next)  # True while paused at validation_node


def send_decision(approved, feedback=""):
    with st.spinner("Approving..." if approved else "Re-analyzing with your feedback..."):
        analyzer.invoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config=graph_config(),
        )
    load_state()
    st.rerun()


def review_panel():
    st.divider()
    st.subheader("Was this analysis useful?")

    if st.session_state["approved"]:
        st.success("You approved this report. ✅")
        return

    if not st.session_state["awaiting"]:
        st.info("Maximum number of re-analyses reached. This is the final report.")
        return

    revisions = st.session_state["revisions"]
    can_retry = revisions < MAX_REVISIONS

    feedback = st.text_area(
        "Not satisfied? Tell the AI what to fix, for example: 'I do have React experience from my college project' "
        "or 'project ideas are not relevant to this job'.",
        key=f"feedback_{revisions}",
    )

    col1, col2 = st.columns(2)
    if col1.button("✅ Approve report", use_container_width=True):
        send_decision(True)

    if can_retry:
        if col2.button("🔄 Not satisfied, re-analyze", use_container_width=True):
            if not feedback.strip():
                st.warning("Please write what should be changed so the AI can improve it.")
            else:
                send_decision(False, feedback.strip())
    else:
        col2.caption("No re-analyses left. You can still approve this report.")


# ************************* Sidebar *************************
for key, default in {"report": None, "revisions": 0, "approved": False, "awaiting": False}.items():
    st.session_state.setdefault(key, default)

st.sidebar.title("🎯 AI Career Coach")
resume_file = st.sidebar.file_uploader("Upload resume (PDF)", type="pdf")
jd_text = st.sidebar.text_area("Paste job description", height=250)
analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

if analyze:
    if not resume_file or not jd_text.strip():
        st.sidebar.warning("Please upload a resume and paste a job description.")
    else:
        try:
            st.session_state["thread_id"] = str(uuid.uuid4())  # fresh run = fresh thread
            with st.spinner("Analyzing your resume against the job description..."):
                analyzer.invoke(
                    {"resume_text": extract_pdf_text(resume_file), "jd_text": jd_text},
                    config=graph_config(),
                )
            load_state()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ************************* Main report *************************
report = st.session_state["report"]

if not report:
    st.title("AI Career Coach")
    st.info("Upload your resume and paste a job description in the sidebar, then click Analyze.")
else:
    score = pct(report.get("match_score"))

    st.caption(f"Revision {st.session_state['revisions']} of {MAX_REVISIONS}")
    st.header(f"Resume Analysis: {report.get('role_title', '')}")
    st.markdown(hero_card(report, score), unsafe_allow_html=True)

    st.subheader("1. Overall resume match")
    st.markdown(match_card(report, score), unsafe_allow_html=True)

    st.subheader("2. Job description vs. your resume")
    st.markdown(requirements_table(report), unsafe_allow_html=True)

    st.subheader("3. Your strongest points for this job")
    st.markdown(list_card(report.get("strengths", []), "Relevant strengths", "green"), unsafe_allow_html=True)

    st.subheader("4. Your major gaps")
    st.markdown(list_card(report.get("gaps", []), "Needs attention", "orange"), unsafe_allow_html=True)

    st.subheader("5. Recommended resume changes")
    st.markdown(changes_table(report), unsafe_allow_html=True)
    st.markdown("#### Suggested professional summary")
    st.caption(report.get("summary_caveat", ""))
    st.code(report.get("suggested_summary", ""), language=None, wrap_lines=True)

    st.subheader("6. Projects to consider")
    st.write(report.get("projects_intro", ""))
    st.markdown(projects_cards(report), unsafe_allow_html=True)

    st.subheader("7. Interview preparation priorities")
    st.write(report.get("interview_intro", ""))
    st.markdown(interview_table(report), unsafe_allow_html=True)
    st.write(report.get("interview_note", ""))

    st.subheader("8. Final assessment")
    st.markdown(final_card(report, score), unsafe_allow_html=True)
    st.markdown(f"**My recommendation:** {report.get('recommendation', '')}")

    review_panel()