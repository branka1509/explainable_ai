import json

import pandas as pd
import streamlit as st

import core

core.setup("Where did the answer come from?", "Generative AI · an assistant answers from our lending policy")

if not core.unlocked("sources"):
    st.stop()


@st.cache_data
def load():
    with open(core.ART / "llm_sources.json", encoding="utf-8") as f:
        return json.load(f)


D = load()
SRC, ITEMS, SET, TRUTH = D["sources"], D["items"], D["settings"], D["truth"]
IDS = list(ITEMS)

stage = st.session_state.setdefault("src_stage", 3 if core.is_presenter() else 1)


def next_button(to, label="Continue →"):
    if stage < to:
        if st.button(label, key=f"src_next_{to}", type="primary"):
            st.session_state["src_stage"] = to
            st.rerun()
        st.stop()


def chat(role, html):
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(html, unsafe_allow_html=True)


# ================================================================= A. question and answer
st.header("A · The assistant's answer")
st.markdown("A bank employee asks the internal assistant. The assistant can read **four passages** of our lending policy "
            "and **cites** them in its answer.")

left, right = st.columns([3, 2], gap="large")
with left:
    chat("user", D["question"])
    lines = "".join(f"<li>{ITEMS[i]['text']} <b>[{ITEMS[i]['cite']}]</b></li>" for i in IDS)
    chat("assistant", f"You will need:<ul>{lines}</ul>")
with right:
    st.markdown("**The policy passages the assistant can read**")
    for k, s in SRC.items():
        with st.container(border=True):
            st.markdown(f"**[{k}] {s['title']}**  \n{s['text']}")
st.caption(D["note"])

core.question("Every item has a citation. <b>Does that mean every item came from our policy?</b>")
next_button(2, "Make your guess →")

# ================================================================= B. guess
st.divider()
st.header("B · Your guess: our policy or general knowledge?")
st.markdown("For each item, decide where you think it **really** came from. A general-purpose LLM has read a lot about Swiss mortgages.")
for i in IDS:
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"{ITEMS[i]['text']} **[{ITEMS[i]['cite']}]**")
    c2.segmented_control("Source", ["Our policy", "General knowledge"], key=f"src_guess_{i}", label_visibility="collapsed")
n_guessed = sum(st.session_state.get(f"src_guess_{i}") is not None for i in IDS)
st.progress(n_guessed / len(IDS), text=f"{n_guessed} of {len(IDS)} guessed")
if n_guessed < len(IDS) and not core.is_presenter():
    st.stop()
next_button(3, "Run the experiment: remove sources →")

# ================================================================= C. experiment
st.divider()
st.header("C · The experiment: take a source away")
st.markdown("We ask the **same question again**, but hide one passage from the assistant. "
            "If an item **survives** without its source, the assistant didn't need that source: it knew it anyway.")
choice = st.radio("Hide from the assistant", list(SET), format_func=lambda k: SET[k]["label"], horizontal=True, key="src_remove")
setting = SET[choice]

BADGE = {"same": ("✓ unchanged", "#E8EEFB", core.NAVY), "changed": ("~ detail lost", "#FFF1DB", "#9A5500"),
         "gone": ("✗ disappeared", "#FBE3E3", core.RED)}
left, right = st.columns([3, 2], gap="large")
with left:
    rows = []
    for a in setting["answer"]:
        item = ITEMS[a["id"]]
        text = a.get("text", item["text"])
        label, bg, fg = BADGE[a["status"]]
        if choice != "none" and a["status"] == "same" and item["cite"] == choice:
            label, bg, fg = ("✓ survives without its source!", "#E3F4E8", "#1E6B35")
        shown = f"<s>{item['text']}</s>" if a["status"] == "gone" else text
        cite = "" if (choice == "all" or item["cite"] == choice) else f" <b>[{item['cite']}]</b>"
        lost = f"<br><span style='font-size:0.8rem;color:{fg}'>lost: {a['lost']}</span>" if a.get("lost") and a["status"] == "changed" else ""
        if choice not in ("none", "all") and item["cite"] != choice:
            rows.append(f"<li style='margin-bottom:6px;color:{core.GREY}'>{shown}{cite}</li>")
            continue
        badge = "" if choice == "none" else (f"<span style='background:{bg};color:{fg};border-radius:6px;padding:1px 7px;"
                                             f"font-size:0.8rem;white-space:nowrap'>{label}</span>")
        rows.append(f"<li style='margin-bottom:6px'>{shown}{cite} {badge}{lost}</li>")
    for e in setting["extra"]:
        rows.append(f"<li style='margin-bottom:6px'>{e} <span style='background:#EDE7F6;color:#5B3C99;border-radius:6px;"
                    f"padding:1px 7px;font-size:0.8rem'>+ new: from general knowledge</span></li>")
    chat("user", D["question"])
    chat("assistant", f"You will need:<ul>{''.join(rows)}</ul>")
with right:
    if choice == "none":
        st.info("Pick a passage to hide and watch what happens to its items.")
    else:
        hit = [a for a in setting["answer"] if choice == "all" or ITEMS[a["id"]]["cite"] == choice]
        gone = sum(a["status"] == "gone" for a in hit)
        lost = sum(a["status"] == "changed" for a in hit)
        st.metric("Items citing this source" if choice != "all" else "Items in the answer", f"{len(hit)}")
        st.metric("…that disappeared", gone)
        st.metric("…that survived anyway", len(hit) - gone)
        if lost:
            st.caption(f"{lost} of them lost a detail")
        if setting["extra"]:
            st.caption(f"+ {len(setting['extra'])} new items the assistant added from general knowledge")
        st.markdown("**Try:** hide [2] and look at the form HB-12 vs. the debt enforcement extract. Then hide everything.")

a = st.radio("Hide [2]. What do you learn about the debt enforcement extract?", [
    "It came from [2]: it disappeared", "The assistant knew it anyway: only the '3 months' detail came from [2]",
    "Nothing, the citation proves it came from [2]"], index=None, key="src_q1")
if a is not None:
    (st.success if a.startswith("The assistant knew") else st.warning)(
        "The extract survives without [2]; only 'not older than three months' is lost. The citation [2] was correct, "
        "but the assistant didn't *need* [2] for that item: it is standard Swiss knowledge. Form HB-12, by contrast, "
        "disappears: that one really came from our policy.")
next_button(4, "Reveal: where each item really came from →")

# ================================================================= D. reveal
st.divider()
st.header("D · Where each item really came from")
rows = []
score = 0
for i in IDS:
    g = st.session_state.get(f"src_guess_{i}")
    truth = TRUTH[i]["verdict"]
    ok = (g == "Our policy" and truth == "From our policy") or (g == "General knowledge" and truth != "From our policy")
    score += ok
    rows.append({"Item": ITEMS[i]["text"], "Cited": f"[{ITEMS[i]['cite']}]", "Your guess": g or "–",
                 "What really drove it": truth, "Why": TRUTH[i]["detail"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if all(st.session_state.get(f"src_guess_{i}") for i in IDS):
    st.markdown(f"**You matched {score} of {len(IDS)}.**")

core.question(
    "<b>Take-aways</b><br>"
    "• A <b>citation</b> tells you what the assistant <i>says</i> it used. <b>Taking the source away</b> tells you what it <i>actually</i> needed.<br>"
    "• Most of this answer would have been the same <b>without our policy</b>. Only the bank-specific details (HB-12, three months, approved valuer) truly came from it.<br>"
    "• That's dangerous where our policy <b>differs</b> from general practice: the assistant may fall back on what it 'knows'.<br>"
    "• This is the same idea as LIME/SHAP, with <b>policy passages as the features</b>. Research tools (e.g. ContextCite) automate it."
)

if st.button("Start this step over"):
    for k in list(st.session_state):
        if k.startswith("src_"):
            del st.session_state[k]
    st.rerun()
