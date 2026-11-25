import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core

core.setup("5 · Why this person?", "SHAP: splitting each prediction fairly between the features")

if not core.unlocked("shap"):
    st.stop()


# ================================================================= data
@st.cache_data
def load_shap():
    d = np.load(core.ART / "shap.npz", allow_pickle=True)
    vals = pd.DataFrame(d["values"], columns=core.FEATURES)
    X = pd.DataFrame(d["X"], columns=core.FEATURES)
    return vals, X, float(d["base"]), int(d["n_ten"])


vals, X, BASE, N_TEN = load_shap()
ten = core.ten_applicants()
PEOPLE = list(ten.index)
G_VALS, G_X = vals.iloc[N_TEN:], X.iloc[N_TEN:]          # 1,000 random test applicants
imp = core.importance()
LABEL = core.NAMES


stage = st.session_state.setdefault("shap_stage", 3 if core.is_presenter() else 1)


def next_button(to, label="Continue →"):
    if stage < to:
        if st.button(label, key=f"shap_next_{to}", type="primary"):
            st.session_state["shap_stage"] = to
            st.rerun()
        st.stop()


# ================================================================= A. local explanations
st.header("A · Why did the model decide this for one person?")
st.markdown(
    f"SHAP starts from the **average prediction ({BASE:.0%})** and splits the difference to this person's prediction "
    "between the features, **fairly over all possible orders** (remember slide: *revealing facts one at a time*). "
    "**Red** pushes the risk of denial **up**, **blue** pushes it **down**. The contributions add up exactly to the prediction."
)

c1, c2 = st.columns([2, 3])
person = c1.selectbox("Applicant", PEOPLE, index=0, key="shap_person")
compare = c2.toggle("Compare with a second applicant", value=False, key="shap_compare")
people = [person]
if compare:
    other = c2.selectbox("Second applicant", [p for p in PEOPLE if p != person],
                         index=([p for p in PEOPLE if p != person].index("Person 10") if person == "Person 1" else 0),
                         key="shap_other")
    people.append(other)

st.caption("How to read it: **E[f(X)]** = the average prediction (8.5%), **f(x)** = this person's prediction. "
           "Each bar is one feature's push: **+0.39 means +39 percentage points**. Red pushes the risk up, blue down.")
cols = st.columns(len(people), gap="large")
for col, p in zip(cols, people):
    i = PEOPLE.index(p)
    a = ten.loc[p]
    with col:
        st.image(str(core.ART / "shap_plots" / f"waterfall_{i+1}.png"), width="stretch")
        decided = "Denied" if a.denied == 1 else "Approved"
        reason = f" · bank's stated reason: **{a.denial_reason_1}**" if a.denied == 1 and isinstance(a.denial_reason_1, str) else ""
        st.markdown(f"**{p}**: model {a.p_denied:.0%} · bank: **{decided}**{reason}")

st.subheader("Questions")
Q = [
    ("shap_q1", "What pushed **Person 1**'s risk up the most?",
     ["Loan amount", "Debt-to-income", "Loan type"], "Debt-to-income",
     "Debt-to-income (55%) adds about +39 points, the same reason the bank gave."),
    ("shap_q2", "**Person 10** has the same 55% debt-to-income as Person 1. What does SHAP say about it for Person 10?",
     ["It pushes the risk up a lot", "It pushes the risk up only a little", "It pushes the risk down"],
     "It pushes the risk up only a little",
     "Only about +6 points: for a VA loan the model barely cares about debt-to-income (remember the ICE lines). "
     "The VA loan itself pulls the risk down."),
    ("shap_q3", "The bank denied **Person 10** for *credit history*. Why doesn't SHAP mention it?",
     ["SHAP made a mistake", "The model never saw credit history, so SHAP can't use it", "Credit history doesn't matter"],
     "The model never saw credit history, so SHAP can't use it",
     "SHAP explains **the model**, not **the world**. If a reason isn't among the model's inputs, it can't appear in the explanation."),
]
for key, q, opts, right, why in Q:
    a = st.radio(q, opts, index=None, key=key)
    if a is not None:
        (st.success if a == right else st.warning)(why)

core.question("**Try:** Person 2 was also denied for <i>credit history</i>. What does SHAP blame instead? "
              "And for Person 5 (FHA loan, approved): which feature drives the model's 54%?")
next_button(2, "From one person to everyone →")

# ================================================================= B. global SHAP
st.divider()
st.header("B · From individuals to the whole portfolio")
st.markdown("Compute SHAP for **1,000 applicants** and look at all contributions together.")

left, right = st.columns([2, 3], gap="large")
with left:
    st.image(str(core.ART / "shap_plots" / "bar.png"), width="stretch")
    st.caption("Average size of each feature's push, whichever direction.")
with right:
    st.image(str(core.ART / "shap_plots" / "beeswarm.png"), width="stretch")

st.markdown("**How to read the dots:** every dot is one applicant. Right of the line = pushes the risk **up**; "
            "**red = high** value of the feature, **blue = low**. For debt-to-income, the red dots (high DTI) sit far to the right. "
            "For *loan type*, red means VA / rural loans: they pull the risk down.")
B = [
    ("shap_b1", "For most applicants, what does debt-to-income do?",
     ["Pushes the risk up a lot", "Pushes it down a little", "Nothing"], "Pushes it down a little",
     "For 88% of applicants, debt-to-income *lowers* the risk by about 3 points: most people have a moderate DTI. "
     "Above 60% DTI it adds about 65 points on average. A few huge pushes, many small ones: "
     "that is why the *average* push (left chart) looks modest."),
    ("shap_b2", "Unlike the feature importance charts, what can this chart show?",
     ["Only which feature matters most", "The direction of the effect and how it varies across people", "The model's accuracy"],
     "The direction of the effect and how it varies across people",
     "Global SHAP keeps the **direction** (left/right) and the **spread** across people, and every dot is one person's explanation."),
]
for key, q, opts, right, why in B:
    a = st.radio(q, opts, index=None, key=key)
    if a is not None:
        (st.success if a == right else st.warning)(why)
next_button(3, "Compare with feature importance →")

# ================================================================= C. confront with feature importance
st.divider()
st.header("C · SHAP vs. the feature importance we saw before")

shares = pd.DataFrame({
    "Gini (step 3)": pd.Series(imp["gini"]),
    "Permutation (step 3)": pd.Series(imp["permutation_mean"]).clip(lower=0),
    "SHAP global": G_VALS.abs().mean(),
})
shares = shares / shares.sum()
ranks = shares.rank(ascending=False).astype(int)
order = shares["SHAP global"].sort_values(ascending=False).index

fig, axs = plt.subplots(1, 3, figsize=(13, 5), dpi=130, sharey=True)
for ax, col in zip(axs, shares.columns):
    s = shares.loc[order, col][::-1]
    colours = [core.RED if (col != "SHAP global" and abs(ranks.loc[f, col] - ranks.loc[f, "SHAP global"]) >= 2) else core.BLUE
               for f in s.index]
    ax.barh([LABEL[f] for f in s.index], s.values, color=colours, height=0.6)
    for k, (f, v) in enumerate(s.items()):
        ax.text(v + 0.006, k, f"#{ranks.loc[f, col]}", va="center", fontsize=9, color=core.NAVY)
    ax.set_title(col, loc="left", fontweight="bold", color=core.NAVY, fontsize=12)
    ax.set_xlim(0, shares.max().max() * 1.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    core.style_axes(ax); ax.tick_params(axis="y", length=0)
axs[0].set_xlabel("Share of total importance")
plt.tight_layout()
st.pyplot(fig, width="stretch")
st.caption("Features ordered by SHAP. Red = ranked at least two places differently than by SHAP.")

C = [
    ("shap_c1", "Do all three methods agree on the most important feature?",
     ["Yes", "No"], "Yes", "Yes: debt-to-income is #1 everywhere. The headline is robust."),
    ("shap_c2", "Which method is the odd one out?",
     ["Gini", "Permutation", "SHAP"], "Gini",
     "Gini puts **loan type** at #2 and **property value** low. SHAP and permutation both rank property value and loan amount high."),
    ("shap_c3", "Why do SHAP and permutation agree more with each other than with Gini?",
     ["Coincidence",
      "Both look at the model's behaviour on applicants; Gini only counts how the trees were built on training data",
      "Gini is wrong"],
     "Both look at the model's behaviour on applicants; Gini only counts how the trees were built on training data",
     "Gini rewards features the trees **split on often** during training. Permutation and SHAP measure what the model actually **does** with each feature."),
]
for key, q, opts, right, why in C:
    a = st.radio(q, opts, index=None, key=key)
    if a is not None:
        (st.success if a == right else st.warning)(why)

st.subheader("Three methods, three different questions")
st.markdown(
    """
| Method | Question it answers | Measured on | Direction? | For one person? |
|---|---|---|---|---|
| **Gini** | How much did the trees *use* this feature when they were built? | Training data | No | No |
| **Permutation** | How much *worse* does the model get without this feature? | New data | No | No |
| **SHAP** | How much does this feature *push* each prediction, and which way? | New data, per person | **Yes** | **Yes** |
    """
)

core.question(
    "**Take-aways**<br>"
    "• SHAP gives a <b>local</b> explanation (why <i>this</i> person) that adds up exactly to the prediction, and a <b>global</b> picture by combining many people.<br>"
    "• Overlap with feature importance on the big picture (debt-to-income first), differences in the details, because each method asks a <b>different question</b>.<br>"
    "• High global importance ≠ the reason for each person: debt-to-income decided Person 1 but barely mattered for Person 10.<br>"
    "• SHAP explains <b>the model</b>, not the world (credit history), and it fills unknown features with other applicants' values, "
    "so related features (loan amount vs. property value: +15 and −10 for Person 1) share the credit somewhat arbitrarily."
)

if st.button("Start this step over"):
    for k in list(st.session_state):
        if k.startswith("shap_"):
            del st.session_state[k]
    st.rerun()
