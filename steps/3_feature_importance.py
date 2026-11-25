import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core

core.setup("3 · What matters to the model?", "Feature importance: the first step towards an explanation")

if not core.unlocked("importance"):
    st.stop()

imp = core.importance()
gini = pd.Series(imp["gini"]).rename(index=core.NAMES)
perm = pd.Series(imp["permutation_mean"]).rename(index=core.NAMES)
perm_sd = pd.Series(imp["permutation_std"]).rename(index=core.NAMES)

st.markdown("The first question most people ask: **which inputs does the model rely on most?** "
            "There are two classic ways to measure this. Switch between them.")

method = st.segmented_control("Method", ["Gini (impurity)", "Permutation", "Compare both"],
                              default="Gini (impurity)", label_visibility="collapsed")
method = method or "Gini (impurity)"


def bar(ax, s, title, xlabel, fmt, err=None):
    s = s.sort_values()
    e = None if err is None else err.reindex(s.index)
    ax.barh(s.index, s.values, color=core.BLUE, height=0.62, xerr=e,
            error_kw=dict(ecolor=core.GREY, lw=1, capsize=2))
    off = s.max() * 0.015
    for i, v in enumerate(s.values):
        ax.text(v + (0 if e is None else e.values[i]) + off, i, fmt(v), va="center", color=core.NAVY, fontsize=9)
    ax.set_title(title, loc="left", fontweight="bold", color=core.NAVY)
    ax.set_xlabel(xlabel, color=core.GREY, fontsize=9)
    ax.set_xlim(0, s.max() * 1.2)
    core.style_axes(ax)
    ax.tick_params(axis="y", length=0)


GINI_TXT = ("**How it works:** each time a tree splits on a feature, the groups become \"purer\" (more clearly approved or denied). "
            "Gini importance **adds up how much each feature improved purity** across all 300 trees, on the **training** data. "
            "It comes for free with tree models.")
PERM_TXT = ("**How it works:** take one feature in **new (test) data** and **shuffle it** across applicants, breaking its link to the outcome. "
            "Then measure **how much worse the model gets**. A big drop means the model really needs that feature. "
            "This works for *any* model.")

if method == "Compare both":
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    bar(axs[0], gini, "Gini (impurity)", "Share of impurity reduction (training data)", lambda v: f"{v:.1%}")
    bar(axs[1], perm, "Permutation", "Drop in AUC when shuffled (test data)", lambda v: f"{v:.3f}", perm_sd)
    plt.tight_layout(w_pad=3)
    st.pyplot(fig, width="stretch")
    ranks = pd.DataFrame({"Rank by Gini": gini.rank(ascending=False).astype(int),
                          "Rank by permutation": perm.rank(ascending=False).astype(int)})
    ranks["Moves"] = ranks["Rank by Gini"] - ranks["Rank by permutation"]
    ranks = ranks.sort_values("Rank by Gini")
    left, right = st.columns([2, 3], gap="large")
    left.dataframe(ranks, width="stretch")
    right.markdown(
        "**They agree on the headline:** debt-to-income comes first in both.\n\n"
        "**They disagree on the rest:** loan type is #2 by Gini but #5 by permutation; property value jumps from #4 to #2.\n\n"
        "**Why?** Gini is measured on training data and rewards features the trees *split on often*. "
        "Permutation is measured on new data and shows what the model *needs*. "
        "Property value, loan amount and loan-to-value are closely related, so they share or steal credit.\n\n"
        "**Same model, two rankings.**"
    )
else:
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
        if method.startswith("Gini"):
            bar(ax, gini, "Gini (impurity) importance", "Share of total impurity reduction (training data)", lambda v: f"{v:.1%}")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        else:
            bar(ax, perm, "Permutation importance", "Drop in AUC when the feature is shuffled (test data)", lambda v: f"{v:.3f}", perm_sd)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
    with right:
        st.markdown(GINI_TXT if method.startswith("Gini") else PERM_TXT)
        st.caption("Tip: switch to *Compare both* to see where the two methods disagree.")

st.divider()
st.subheader("Can you make a decision based on this?")
st.markdown("Answer from the chart alone.")

QUIZ = [
    ("Does a **higher** debt-to-income ratio **increase** the risk of denial?",
     ["Yes, it increases the risk", "No, it decreases the risk", "The chart can't tell me"],
     "The chart shows *how much* the model uses debt-to-income, not *in which direction*."),
    ("**From which** debt-to-income level does it become a problem?",
     ["36%", "43%", "50%", "The chart can't tell me"],
     "Importance has no thresholds. It is one number per feature."),
    ("Is the effect of debt-to-income the **same for everyone**, e.g. conventional and VA loans?",
     ["Yes", "No", "The chart can't tell me"],
     "Importance averages over everyone. Remember Person 1 and Person 10?"),
    ("Was debt-to-income the reason **Person 1** was rejected?",
     ["Yes", "No", "The chart can't tell me"],
     "Global importance says nothing about one individual."),
]
for i, (q, opts, why) in enumerate(QUIZ):
    ans = st.radio(q, opts, index=None, key=f"quiz_{i}")
    if ans is not None:
        if ans == "The chart can't tell me":
            st.success(f"Exactly. {why}")
        else:
            st.warning(f"Careful. {why}")

if all(st.session_state.get(f"quiz_{i}") for i in range(len(QUIZ))):
    core.question("**Feature importance tells us <i>what</i> matters, but not <i>how</i>, <i>for whom</i>, or <i>why this person</i>.**<br>"
                  "Next: <b>partial dependence</b> and <b>ICE plots</b> show the direction and shape of an effect.")
