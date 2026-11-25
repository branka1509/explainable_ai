import json

import pandas as pd
import streamlit as st

import core

core.setup("Agent detective: what went wrong?", "Agentic AI · read the trace, find the faulty step")

if not core.unlocked("trace"):
    st.stop()


@st.cache_data
def load():
    with open(core.ART / "agent_traces.json", encoding="utf-8") as f:
        return json.load(f)


D = load()
CASES, CATS = D["cases"], D["categories"]
# Answer options are shown in a different order for each file, so the position gives nothing away.
# Correct option lands at position 3 (File 1), 1 (File 2), 4 (File 3), 2 (File 4).
OPTION_ORDER = {"A": [2, 3, 0, 1], "B": [1, 0, 3, 2], "C": [3, 0, 1, 2], "D": [0, 3, 2, 1]}
PRESENTER = core.is_presenter()
stage = st.session_state.setdefault("tr_stage", 3 if PRESENTER else 1)


def next_button(to, label="Continue →"):
    if stage < to:
        if st.button(label, key=f"tr_next_{to}", type="primary"):
            st.session_state["tr_stage"] = to
            st.rerun()
        st.stop()


STYLE = {  # type: (icon, label, background, text colour)
    "thought": ("💭", "Thinks", "#F1F3F7", core.NAVY),
    "call": ("🔧", "Calls tool", "#E8EEFB", core.BLUE),
    "result": ("📄", "Tool result", "#FFFFFF", "#333333"),
    "output": ("✅", "Final output", core.NAVY, "#FFFFFF"),
}


def render_trace(steps, highlight=None):
    rows = []
    for n, (kind, text) in enumerate(steps, start=1):
        icon, label, bg, fg = STYLE[kind]
        if kind == "result" and text.startswith("ERROR"):
            bg = "#FFF1DB"
        border = f"3px solid {core.RED}" if highlight and n in highlight else "1px solid #D5DAE3"
        body = f"<code style='background:none;color:{fg}'>{text}</code>" if kind == "call" else text
        rows.append(
            f"<div style='display:flex;gap:12px;align-items:flex-start;background:{bg};color:{fg};border:{border};"
            f"border-radius:8px;padding:8px 12px;margin-bottom:6px'>"
            f"<div style='min-width:28px;font-weight:700'>{n}</div>"
            f"<div style='min-width:120px;font-size:0.85rem;opacity:0.85'>{icon} {label}</div>"
            f"<div style='flex:1'>{body}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


# ================================================================= A. the agent and its results
st.header("A · The agent's results")
st.markdown(f"Our bank uses an AI agent for mortgage files. **Its task:** {D['task']}")
st.markdown("**Tools it can use:** " + " · ".join(f"`{t}` ({d})" for t, d in D["tools"]))
st.markdown("This morning it processed four files. Here is what it reported:")
cols = st.columns(4)
for col, (k, c) in zip(cols, CASES.items()):
    with col.container(border=True):
        st.markdown(f"**{c['title']}**")
        st.caption(c["client"])
        st.markdown(f"<div style='color:{core.NAVY}'>{c['output']}</div>", unsafe_allow_html=True)
        st.segmented_control("Would you sign it?", ["Sign", "Don't sign"], key=f"tr_sign_{k}")
core.question("All four outputs look professional. <b>Can you tell from the output alone which ones you can trust?</b>")
next_button(2, "Open the traces →")

# ================================================================= B. investigate
st.divider()
st.header("B · Investigate: read the trace")
st.markdown("Every agent run leaves a **trace**: each thought, each tool call and each result, in order. "
            "Pick a file, read the trace, and find **the first step where it went wrong**.")

pick = st.radio("File", list(CASES), format_func=lambda k: CASES[k]["title"], horizontal=True, key="tr_case")
c = CASES[pick]
st.caption(c["client"])
done = st.session_state.get(f"tr_done_{pick}", False)
render_trace(c["steps"], highlight=c["answer"] if done else None)

with st.form(f"tr_form_{pick}"):
    f1, f2 = st.columns([1, 2])
    step = f1.selectbox("First step that went wrong", list(range(1, len(c["steps"]) + 1)), index=None,
                        placeholder="Step number", key=f"tr_step_{pick}")
    cat = f2.radio("What kind of problem?", [CATS[j] for j in OPTION_ORDER[pick]], index=None, key=f"tr_cat_{pick}")
    ok = st.form_submit_button("Check my answer", type="primary")

if ok:
    if step is None or cat is None:
        st.warning("Pick a step and a type of problem.")
    else:
        st.session_state[f"tr_done_{pick}"] = True
        st.session_state[f"tr_score_{pick}"] = (step in c["answer"]) + (cat == c["category"])
        st.rerun()

if done:
    s = st.session_state.get(f"tr_score_{pick}", 0)
    (st.success if s == 2 else st.info if s == 1 else st.warning)(
        {2: "Spot on: right step and right type.", 1: "Half right. Look at the highlighted step.",
         0: "Not this time. Look at the highlighted step."}[s])
    st.markdown(f"**Step {' / '.join(map(str, c['answer']))} · {c['category']}**  \n{c['why']}", unsafe_allow_html=True)
    st.markdown(f"🛡️ **What would have caught it:** {c['control']}")

n_done = sum(st.session_state.get(f"tr_done_{k}", False) for k in CASES)
st.progress(n_done / len(CASES), text=f"{n_done} of {len(CASES)} files investigated")
if n_done < len(CASES) and not PRESENTER:
    st.stop()
next_button(3, "Debrief →")

# ================================================================= C. debrief
st.divider()
st.header("C · Debrief")
rows = []
for k, c in CASES.items():
    rows.append({"File": c["title"], "You would have signed": st.session_state.get(f"tr_sign_{k}") or "–",
                 "What went wrong": c["category"], "Visible in the output?": "No" if k != "A" else "Only the decision",
                 "Control that catches it": c["control"]})
st.table(pd.DataFrame(rows).set_index("File"))
score = sum(st.session_state.get(f"tr_score_{k}", 0) for k in CASES)
if n_done == len(CASES):
    st.markdown(f"**Your score: {score} of {2 * len(CASES)}.**")

core.question(
    "<b>Take-aways</b><br>"
    "• The <b>final output</b> of an agent can look perfect while a step in the middle was wrong. You only see it in the <b>trace</b>.<br>"
    "• An agent has <b>many more places to fail</b> than a single model: data extraction, tools, documents it reads, and its own reasoning.<br>"
    "• Classic XAI still matters <i>inside</i> the chain: SHAP explains the <code>credit_model.score</code> step, but not the steps around it.<br>"
    "• File 4 had no bug at all. The agent <b>pursued the goal we measured</b> and found a shortcut. That's where we go next."
)

if st.button("Start this step over"):
    for k in list(st.session_state):
        if k.startswith("tr_"):
            del st.session_state[k]
    st.rerun()
