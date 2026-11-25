import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import core

core.setup("6 · Can we trust the explanation?", "Stability of SHAP explanations: an exercise")

if not core.unlocked("stable"):
    st.stop()


@st.cache_data
def load():
    d = np.load(core.ART / "stability.npz", allow_pickle=True)
    s = np.load(core.ART / "shap.npz", allow_pickle=True)
    return ({k: d[k] for k in d.files}, s["values"][: int(s["n_ten"])], float(s["base"]))


D, MAIN_SHAP, MAIN_BASE = load()
ten = core.ten_applicants()
PEOPLE = list(ten.index)
F, LABEL = core.FEATURES, core.NAMES
KINDS = list(D["kinds"])
KIND_LABEL = {"debt_to_income": "Debt-to-income", "income_k": "Income",
              "loan_amount": "Loan amount", "property_value": "Property value"}
KIND_UNIT = {"debt_to_income": lambda s: f"{s:+d} points", "income_k": lambda s: f"{s:+d}%",
             "loan_amount": lambda s: f"{s:+d}k $", "property_value": lambda s: f"{s:+d}k $"}
KIND_NOTE = {"debt_to_income": "Only debt-to-income changes (e.g. slightly different monthly debts).",
             "income_k": "Same debts, different income, so **debt-to-income changes too**.",
             "loan_amount": "Same home, different loan, so **loan-to-value changes too**.",
             "property_value": "Same loan, different home price, so **loan-to-value changes too**."}
ZERO = 3  # index of "no change" in every step list

stage = st.session_state.setdefault("stab_stage", 3 if core.is_presenter() else 1)


def next_button(to, label="Continue →"):
    if stage < to:
        if st.button(label, key=f"stab_next_{to}", type="primary"):
            st.session_state["stab_stage"] = to
            st.rerun()
        st.stop()


def fmt(f, v):
    if f in core.CATEGORICAL:
        return core.CATEGORICAL[f].get(int(round(v)), str(v))
    if f in ("loan_amount", "property_value"):
        return f"${v/1000:,.0f}k"
    if f == "income_k":
        return f"${v:,.0f}k"
    if f in ("loan_to_value", "debt_to_income"):
        return f"{v:.0f}%"
    return f"{v:.0f}"


# ================================================================= A. small changes
st.header("A · Small change, same explanation?")
st.markdown(
    "A good explanation should not change dramatically when the applicant changes **a tiny bit**. "
    "Pick an applicant, change **one thing slightly**, and watch what happens to the prediction and to the SHAP explanation."
)

c1, c2 = st.columns([1, 1])
person = c1.selectbox("Applicant", PEOPLE, key="stab_person")
kind = c2.radio("What changes?", KINDS, format_func=KIND_LABEL.get, horizontal=True, key="stab_kind")
pi, ki = PEOPLE.index(person), KINDS.index(kind)
steps = [int(s) for s in D["steps"][ki]]
step = st.select_slider("How much?", options=steps, value=0, format_func=lambda s: "no change" if s == 0 else KIND_UNIT[kind](s),
                        key=f"stab_step_{kind}")
si = steps.index(step)
st.caption(KIND_NOTE[kind])

x0, x1 = D["X"][pi, ki, ZERO], D["X"][pi, ki, si]
s0, s1 = D["shap"][pi, ki, ZERO], D["shap"][pi, ki, si]
p0, p1 = float(D["p"][pi, ki, ZERO]), float(D["p"][pi, ki, si])
top0, top1 = F[int(np.abs(s0).argmax())], F[int(np.abs(s1).argmax())]
shift = np.abs(s1 - s0)
big = F[int(shift.argmax())]

m1, m2, m3 = st.columns(3)
m1.metric("Predicted risk of denial", f"{p1:.0%}", f"{(p1-p0)*100:+.0f} points" if step else None, delta_color="inverse")
m2.metric("Top reason", LABEL[top1], "changed!" if top1 != top0 else ("unchanged" if step else None),
          delta_color="off" if top1 == top0 else "inverse")
m3.metric("Biggest shift in one contribution", f"{shift.max()*100:.0f} points" if step else "–")
if step:
    m3.caption(f"in {LABEL[big]}")

changed = [f for f in F if abs(x1[F.index(f)] - x0[F.index(f)]) > 1e-6]
if step:
    st.markdown("**What changed:** " + " · ".join(f"{LABEL[f]} {fmt(f, x0[F.index(f)])} → **{fmt(f, x1[F.index(f)])}**" for f in changed))

order = np.argsort(-np.abs(s0))[:8][::-1]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
y = np.arange(len(order))
ax.barh(y + 0.19, s0[order], height=0.36, color=core.GREY, label=f"Original ({p0:.0%})")
ax.barh(y - 0.19, s1[order], height=0.36, color=core.BLUE, label=f"After the change ({p1:.0%})")
ax.axvline(0, color=core.NAVY, lw=1)
ax.set_yticks(y); ax.set_yticklabels([LABEL[F[k]] for k in order])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}"))
ax.set_xlabel("SHAP contribution (percentage points)")
core.style_axes(ax); ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
st.pyplot(fig, width="stretch")

st.subheader("Your challenges")
CH = [
    ("Challenge 1 · Person 1",
     "Find the **smallest** change that brings Person 1's risk **below 50%**.",
     "Debt-to-income **−3 points** (55% → 52%), or income **+10%** (which also brings DTI to 50%): "
     "the risk collapses from **80% to 19%** and the top reason switches from debt-to-income to loan amount. "
     "−1 or −2 points change nothing at all. **A cliff.**"),
    ("Challenge 2 · Person 10",
     "Make Person 10's home **slightly cheaper**. What happens?",
     "Only **$5k cheaper** and the risk jumps from **6% to 35%**: the loan is now bigger than the home "
     "(loan-to-value just above 100%), and loan-to-value becomes the top reason."),
    ("Challenge 3 · Person 2",
     "Increase Person 2's **loan by $25k**. Does the prediction change? Does the explanation?",
     "The prediction barely moves (**98% → 99%**), but debt-to-income's contribution drops by about **24 points** "
     "and the credit moves to loan-to-value. **Same decision, a different story.**"),
]
for title, task, answer in CH:
    with st.expander(title):
        st.markdown(task)
        if st.toggle("Show what we found", key=f"stab_{title}"):
            st.info(answer)

a = st.radio("When Person 1's explanation flipped (Challenge 1), what was unstable?",
             ["SHAP made a calculation error", "The model itself: its prediction jumped, and SHAP reflects that faithfully",
              "Nothing, that's normal"], index=None, key="stab_q1")
if a is not None:
    if a.startswith("The model"):
        st.success("Right. The prediction itself jumps from 80% to 19%. In the public data, debt-to-income between 50% and 60% "
                   "is only reported as a band, so the model never saw 50–54% and learned a **cliff**. SHAP explains the cliff correctly. "
                   "But for a customer, an explanation that flips with a 3-point change is hard to defend.")
    else:
        st.warning("Look at the prediction: it moves from 80% to 19%. The explanation changes because the model's answer changes.")

b = st.radio("And in Challenge 3 (Person 2)?",
             ["Again the model: the prediction jumped", "The explanation: the prediction stayed, but the credit moved between related features"],
             index=None, key="stab_q2")
if b is not None:
    if b.startswith("The explanation"):
        st.success("Exactly. Loan amount, property value and loan-to-value are linked. When one moves, SHAP can redistribute the "
                   "credit between them although the decision is the same. Read contributions of related features with care.")
    else:
        st.warning("Check the prediction: 98% → 99%. Almost no change, yet the contributions shift a lot.")

next_button(2, "Another source of instability →")

# ================================================================= B. reference group
st.divider()
st.header("B · Same person, different reference group")
st.markdown(
    "SHAP measures every push **relative to an 'average applicant'** (the reference group). "
    "Analysts often use a random sample of **100 applicants** for speed. What if a colleague draws a **different** 100?"
)
person_b = st.selectbox("Applicant", PEOPLE, key="stab_person_b")
pb = PEOPLE.index(person_b)
R = D["ref_shap"][:, pb, :]
main = MAIN_SHAP[pb]
order = np.argsort(-np.abs(main))[:8][::-1]

fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
for k, fi in enumerate(order):
    ax.plot([R[:, fi].min(), R[:, fi].max()], [k, k], color=core.LIGHT, lw=8, solid_capstyle="round", zorder=1)
    ax.scatter(R[:, fi], np.full(5, k), color=core.BLUE, s=28, zorder=2, label="5 analysts, 100 applicants each" if k == 0 else None)
    ax.scatter(main[fi], k, marker="D", color=core.RED, s=40, zorder=3, label="Step 5 (1,000 applicants)" if k == 0 else None)
ax.axvline(0, color=core.NAVY, lw=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LABEL[F[k]] for k in order])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}"))
ax.set_xlabel("SHAP contribution (percentage points)")
core.style_axes(ax); ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower right", fontsize=9)
plt.tight_layout()
st.pyplot(fig, width="stretch")

bases = D["ref_base"]
tops = {LABEL[F[int(np.abs(R[r]).argmax())]] for r in range(5)}
c1, c2, c3 = st.columns(3)
c1.metric("'Average applicant' baseline", f"{bases.min():.1%} – {bases.max():.1%}", help="Across the 5 reference groups")
c2.metric("Largest disagreement on one feature", f"{(R.max(0) - R.min(0)).max()*100:.0f} points")
c3.metric("Top reason", " / ".join(sorted(tops)))
c3.caption("the same for all 5 analysts" if len(tops) == 1 else "differs between analysts!")

c = st.radio("What would you tell the board about these differences?",
             ["The explanations are unreliable and useless",
              "The main reason is robust; differences of a few points should not be over-interpreted",
              "There are no differences"], index=None, key="stab_q3")
if c is not None:
    if c.startswith("The main reason"):
        st.success("Yes. The top reason holds, but single contributions move by a few points and the baseline by up to 3 points. "
                   "Use a large reference group, and don't present SHAP numbers with false precision.")
    else:
        st.warning("Look again: the top reason is the same in all five runs, but the exact numbers move a little.")

next_button(3, "Take-aways →")

# ================================================================= C. take-aways
st.divider()
st.header("C · What this means in practice")
st.markdown(
    """
| Source of instability | What we saw | What to do |
|---|---|---|
| **Model cliffs** | 3 points of DTI or a $5k cheaper home flip the prediction and the explanation | Test explanations with small changes before using them; flag applicants near a cliff for human review |
| **Related features** | Same decision, but credit moves between loan amount, property value and loan-to-value | Report related features as a group ("loan size"), not one by one |
| **Reference group** | Baseline 8–11%, contributions ± a few points | Use a large, documented reference group; round numbers; report the top reasons, not decimals |
    """
)
core.question("**Key message:** an explanation is only as stable as the model behind it and the choices of the analyst. "
              "Before an explanation goes to a customer or a regulator, ask: <b>would it still hold if the applicant were slightly different?</b><br>"
              "Next: instead of explaining the risk, <b>what would the applicant need to change</b> to be approved? (counterfactuals)")

if st.button("Start this step over"):
    for k in list(st.session_state):
        if k.startswith("stab_"):
            del st.session_state[k]
    st.rerun()
