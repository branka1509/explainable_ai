import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import core

core.setup("1 · Meet the model", "What the bank's data science team built")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("#### Input")
    st.markdown("13 facts about each application: **income, loan amount, property value, "
                "debt-to-income, loan-to-value, loan type**, age, neighbourhood, …")
with c2:
    st.markdown("#### Model")
    st.markdown("**Gradient boosting**: 300 small decision trees, each correcting the previous ones. "
                "Trained on **45,520** past applications.")
with c3:
    st.markdown("#### Output")
    st.markdown("A **probability that the application is denied**. "
                "Above 50%, the model recommends **reject**.")

st.divider()

_, test = core.load_data()
y = test["denied"].to_numpy()
p = core.test_predictions()
pred = (p >= 0.5).astype(int)
acc = (pred == y).mean()
base = 1 - y.mean()
cm = confusion_matrix(y, pred)
recall = cm[1, 1] / cm[1].sum()

st.subheader("The data science team reports:")
st.markdown(f"<div class='big-number'>{acc:.0%} accurate</div>"
            f"<span class='muted'>on 15,174 applications the model had never seen</span>",
            unsafe_allow_html=True)

core.question("**Would you sign off on deploying this model?**")
vote = st.radio("Your answer", ["Yes, 93% is great", "No", "I need more information first"],
                index=None, horizontal=True, label_visibility="collapsed")

if vote is None:
    st.stop()

st.markdown("### Let's look closer")
m1, m2, m3 = st.columns(3)
m1.metric("Model accuracy", f"{acc:.0%}")
m2.metric("Accuracy of 'approve everyone'", f"{base:.0%}", help="A rule with no model at all")
m3.metric("Denials the model catches", f"{recall:.0%}", help="Share of actually denied applications that the model flags")

left, right = st.columns([3, 2], gap="large")
with left:
    cells = {(0, 0): (cm[0, 0], "Correctly approved", core.BLUE, "white"),
             (0, 1): (cm[0, 1], "Wrongly rejected", core.LIGHT, core.NAVY),
             (1, 0): (cm[1, 0], "Missed denials", core.LIGHT, core.NAVY),
             (1, 1): (cm[1, 1], "Correctly rejected", core.BLUE, "white")}
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.set_xlim(-0.6, 7.6); ax.set_ylim(0, 5.4); ax.axis("off")
    x0, y0, w, h, g = 1.9, 0.3, 2.6, 1.9, 0.12
    for (r, c), (n, lab, fc, tc) in cells.items():
        x, yy = x0 + c * (w + g), y0 + (1 - r) * (h + g)
        ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec="none"))
        ax.text(x + w / 2, yy + h * 0.58, f"{n:,}", ha="center", va="center", fontsize=22, fontweight="bold", color=tc)
        ax.text(x + w / 2, yy + h * 0.27, lab, ha="center", va="center", fontsize=11, color=tc)
    ax.text(x0 + w + g / 2, y0 + 2 * h + g + 0.65, "What the model predicted", ha="center", fontsize=11, color=core.GREY)
    for c, t in enumerate(["Approve", "Reject"]):
        ax.text(x0 + c * (w + g) + w / 2, y0 + 2 * h + g + 0.22, t, ha="center", fontsize=13, fontweight="bold", color=core.NAVY)
    ax.text(-0.35, y0 + h + g / 2, "What the bank\nactually decided", ha="center", va="center", fontsize=11, color=core.GREY, rotation=90)
    for r, t in enumerate(["Approved", "Denied"]):
        ax.text(x0 - 0.2, y0 + (1 - r) * (h + g) + h / 2, t, ha="right", va="center", fontsize=13, fontweight="bold", color=core.NAVY)
    st.pyplot(fig, width="stretch")

with right:
    st.markdown(
        f"""
**What this shows**
- Only **{y.mean():.0%}** of applications are denied. A rule that **approves everyone** is already **{base:.0%}** accurate.
- The model catches only **{recall:.0%}** of the denials at a 50% cut-off. **{cm[1,0]:,}** denials slip through.
- When the model *does* say reject, it is right **{cm[1,1]/cm[:,1].sum():.0%}** of the time.

**Accuracy alone tells you very little.** And where to set the cut-off is a **business decision**, not a property of the model.
        """
    )

with st.expander("For the curious: the ROC curve and AUC"):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(fpr, tpr, color=core.BLUE, lw=2, label=f"Our model (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color=core.GREY, lw=1.2, label="Random guessing (AUC = 0.5)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    core.style_axes(ax); ax.grid(color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    st.pyplot(fig, width="content")
    st.markdown(f"**AUC = {auc:.2f}**: pick one denied and one approved applicant at random. "
                f"The model ranks the denied one as riskier **{auc:.0%}** of the time. "
                "That is a good, realistic model for this data, without credit scores.")

st.page_link("steps/2_be_the_bank.py", label="Next: be the bank →", icon="🧑‍⚖️")
