import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core

core.setup("4 · How does it matter?", "Partial dependence (PDP) and individual conditional expectation (ICE)")

if not core.unlocked("pdp"):
    st.stop()


# ================================================================= data
@st.cache_data
def load_ice():
    d = np.load(core.ART / "ice.npz", allow_pickle=True)
    _, test = core.load_data()
    ids = d["applicant_id"]
    rows = test.set_index("applicant_id").loc[ids].reset_index()
    ten = core.ten_applicants().reset_index()
    rows["person"] = list(ten["person"]) + [None] * (len(rows) - len(ten))
    curves = {f: (d[f"{f}__grid"], d[f"{f}__ice"]) for f in core.FEATURES}
    return rows, curves, int(d["n_ten"])


rows, curves, N_TEN = load_ice()
SAMPLE = slice(N_TEN, None)          # the 500 random test applicants (PDP = their average)
FEATS = core.FEATURES
LABEL = core.NAMES


def xfmt(f, v):
    if f in core.CATEGORICAL:
        return core.CATEGORICAL[f].get(int(v), str(v))
    if f in ("loan_amount", "property_value"):
        return f"${v/1000:,.0f}k"
    return f"{v:,.0f}"


def xticks(ax, f, grid):
    if f in core.CATEGORICAL:
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([xfmt(f, v) for v in grid], fontsize=9)
    else:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if f in ("loan_amount", "property_value"):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))


def xs(f, grid):
    return np.arange(len(grid)) if f in core.CATEGORICAL else grid


def pdp_of(f):
    grid, P = curves[f]
    return grid, P[SAMPLE].mean(0)


def deviation(f):
    """Max distance (in probability points) between each person's centred ICE line and the centred PDP."""
    grid, P = curves[f]
    S = P[SAMPLE]
    pdp = S.mean(0)
    return np.abs((S - S[:, [0]]) - (pdp - pdp[0])).max(1)


def pct_axis(ax):
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))


# ================================================================= stage control
PRESENTER = core.is_presenter()
stage = st.session_state.setdefault("pdp_stage", 5 if PRESENTER else 1)


def next_button(to, label="Continue →"):
    if stage < to:
        if st.button(label, key=f"next_{to}", type="primary"):
            st.session_state["pdp_stage"] = to
            st.rerun()
        st.stop()


# ================================================================= A. PDP for all features
st.header("A · The average effect of each feature")
st.markdown(
    "A **partial dependence plot (PDP)** answers: *if we change one feature and keep everything else as it is, "
    "how does the **average** predicted risk of denial change?* "
    "We take 500 real applicants, set everyone's value of the feature to 15%, 20%, 25%, … and average the predictions."
)
same_scale = st.toggle("Same vertical scale for all features", value=True,
                       help="On: compare how strong the effects are. Off: zoom in on the shape of each effect.")
if not same_scale:
    st.caption("⚠️ Each chart now has its own scale. Small effects (like age) can look dramatic. Check the numbers on the axis.")

fig, axs = plt.subplots(4, 4, figsize=(14, 11), dpi=130, sharey=same_scale)
for ax, f in zip(axs.flat, FEATS):
    grid, pdp = pdp_of(f)
    if f in core.CATEGORICAL:
        ax.bar(range(len(grid)), pdp, color=core.BLUE, width=0.55)
    else:
        ax.plot(grid, pdp, color=core.BLUE, lw=2.4)
    xticks(ax, f, grid)
    ax.set_title(LABEL[f], fontsize=12, color=core.NAVY, fontweight="bold", loc="left")
    core.style_axes(ax); ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    pct_axis(ax); ax.tick_params(labelsize=9)
    if same_scale:
        ax.set_ylim(0, 1)
for ax in axs.flat[len(FEATS):]:
    ax.axis("off")
fig.supylabel("Average predicted risk of denial", color=core.NAVY)
plt.tight_layout()
st.pyplot(fig, width="stretch")

st.subheader("What does the model seem to say?")
QA = [
    ("What happens to the risk of denial as **debt-to-income** rises?",
     ["It rises steadily", "Flat until about 50%, then it jumps", "It falls"], 1),
    ("What happens as the **property value** rises?",
     ["Risk rises", "Risk is very high for cheap homes, then falls and flattens", "No effect"], 1),
    ("And the applicant's **age**?",
     ["Strong effect", "Almost no effect", "Risk falls with age"], 1),
]
for i, (q, opts, correct) in enumerate(QA):
    a = st.radio(q, opts, index=None, key=f"pdp_q{i}")
    if a is not None:
        (st.success if opts.index(a) == correct else st.warning)(
            f"The PDP shows: **{opts[correct].lower()}**.")

core.question("**These are averages over 500 people.** Are you sure this is true for <b>everyone</b>?")
next_button(2, "Look at individuals →")

# ================================================================= B. ICE
st.divider()
st.header("B · One line per person: ICE plots")
st.markdown(
    "An **ICE plot** shows the same experiment **for each applicant separately**: one thin line per person. "
    "The **PDP is simply the average of all these lines** (the thick line)."
)
c1, c2, c3 = st.columns([2, 2, 3])
f_ice = c1.selectbox("Feature", FEATS, index=FEATS.index("debt_to_income"), format_func=LABEL.get, key="ice_feat")
n_lines = c2.slider("Number of applicants shown", 50, 500, 150, 50, key="ice_n")
centred = c3.toggle("Centre all lines at the start (c-ICE)", value=False, key="ice_centre",
                    help="Shifts every line to start at 0, so you see how much each person's risk *changes*.")
highlight = st.multiselect("Highlight applicants from step 2", [f"Person {i}" for i in range(1, 11)],
                           default=["Person 1", "Person 10"] if f_ice == "debt_to_income" else [], key="ice_hl")

grid, P = curves[f_ice]
x = xs(f_ice, grid)
S = P[SAMPLE][:n_lines]
pdp = P[SAMPLE].mean(0)
base = (lambda M: M - M[..., :1]) if centred else (lambda M: M)

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
for r in base(S):
    ax.plot(x, r, color=core.BLUE, alpha=0.18, lw=0.9)
ax.plot(x, base(pdp[None])[0], color=core.NAVY, lw=4, label="PDP (average)")
for p in highlight:
    i = int(p.split()[1]) - 1
    ax.plot(x, base(P[i][None])[0], color=core.RED if i == 0 else "#E07B00", lw=2.6, label=p)
xticks(ax, f_ice, grid)
ax.set_xlabel(LABEL[f_ice]); ax.set_ylabel("Change in risk" if centred else "Predicted risk of denial")
core.style_axes(ax); ax.grid(axis="y", color="#E5E7EB", lw=0.8); pct_axis(ax)
ax.legend(frameon=False, loc="upper left")
st.pyplot(fig, width="stretch")

if f_ice == "debt_to_income" and {"Person 1", "Person 10"} <= set(highlight):
    st.info("**Person 1** (conventional loan) jumps to a high risk above 50% debt-to-income. "
            "**Person 10** (VA loan) barely moves. Same debt-to-income, different story.")

core.question("Try several features. **Do the thin lines look like the thick line?** For which features yes, for which no?")
next_button(3)

# ================================================================= C. which PDPs are honest
st.divider()
st.header("C · Which averages can you trust?")
st.markdown("For each feature: is the PDP a **fair summary** of most people, or does it **hide different groups**? "
            "Use the ICE plot above to check.")
QUIZ = {"debt_to_income": "Hides differences", "applicant_age": "Fair summary",
        "tract_income_pct_of_area": "Fair summary", "loan_amount": "Hides differences"}
answers = {}
cols = st.columns(len(QUIZ))
for col, (f, right) in zip(cols, QUIZ.items()):
    answers[f] = col.radio(LABEL[f], ["Fair summary", "Hides differences"], index=None, key=f"trust_{f}")

if all(answers.values()) or PRESENTER:
    if all(answers.values()):
        score = sum(answers[f] == QUIZ[f] for f in QUIZ)
        st.success(f"You classified {score} of {len(QUIZ)} like the data does. Here is the measurement:")
    share = pd.Series({LABEL[f]: (deviation(f) < 0.10).mean() for f in FEATS}).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    colours = [core.BLUE if v >= 0.8 else core.RED for v in share.values]
    ax.barh(share.index, share.values, color=colours, height=0.6)
    for i, v in enumerate(share.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=core.NAVY)
    ax.axvline(0.8, color=core.GREY, ls="--", lw=1)
    ax.set_xlim(0, 1.12); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of applicants whose line stays within 10 points of the PDP")
    core.style_axes(ax); ax.tick_params(axis="y", length=0)
    st.pyplot(fig, width="stretch")
    st.markdown("**Blue:** the average describes almost everyone. **Red:** the average describes a minority, "
                "and the effect depends on *who* you are.")
    next_button(4, "Who doesn't follow the average? →")
else:
    st.stop()

# ================================================================= D. who deviates
st.divider()
st.header("D · Who doesn't follow the average?")
c1, c2, c3 = st.columns(3)
f_dev = c1.selectbox("Feature", FEATS, index=FEATS.index("debt_to_income"), format_func=LABEL.get, key="dev_feat")
thr = c2.slider("Counts as 'different' if the line is more than … points away", 5, 30, 10, 5, key="dev_thr") / 100
group_by = c3.selectbox("Colour the groups by", ["Follows / differs", "Loan type", "Occupancy", "Co-applicant"],
                        key="dev_group")

grid, P = curves[f_dev]
x = xs(f_dev, grid)
S = P[SAMPLE]
info = rows.iloc[N_TEN:].reset_index(drop=True)
dev = deviation(f_dev)
differs = dev > thr
pdp = S.mean(0)

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
if group_by == "Follows / differs":
    for r, d in zip(S, differs):
        ax.plot(x, r, color=core.RED if d else core.GREY, alpha=0.35 if d else 0.15, lw=1)
    ax.plot([], [], color=core.RED, lw=2, label=f"Differs from the average ({differs.mean():.0%})")
    ax.plot([], [], color=core.GREY, lw=2, label=f"Follows the average ({1-differs.mean():.0%})")
else:
    col = {"Loan type": "loan_type", "Occupancy": "occupancy", "Co-applicant": "has_coapplicant"}[group_by]
    palette = ["#002FA8", "#E07B00", "#2CA02C", "#8A8F98"]
    for k, (val, name) in enumerate(core.CATEGORICAL[col].items()):
        m = (info[col] == val).to_numpy()
        if m.sum() < 5:
            continue
        for r in S[m]:
            ax.plot(x, r, color=palette[k], alpha=0.12, lw=0.8)
        ax.plot(x, S[m].mean(0), color=palette[k], lw=3.2, label=f"{name} (n={m.sum()}): group average")
ax.plot(x, pdp, color=core.NAVY, lw=4, ls="--", label="PDP (everyone)")
xticks(ax, f_dev, grid)
ax.set_xlabel(LABEL[f_dev]); ax.set_ylabel("Predicted risk of denial")
core.style_axes(ax); ax.grid(axis="y", color="#E5E7EB", lw=0.8); pct_axis(ax)
ax.legend(frameon=False, loc="upper left", fontsize=9)
st.pyplot(fig, width="stretch")

if differs.sum() >= 5:
    prof = pd.DataFrame({
        "Differs from average": info[differs].pipe(lambda d: pd.Series({
            "Applicants": len(d), "Conventional loans": (d.loan_type == 1).mean(),
            "FHA loans": (d.loan_type == 2).mean(), "VA loans": (d.loan_type == 3).mean(),
            "Median income (k$)": d.income_k.median(), "Median debt-to-income": d.debt_to_income.median(),
            "Median loan-to-value": d.loan_to_value.median()})),
        "Follows average": info[~differs].pipe(lambda d: pd.Series({
            "Applicants": len(d), "Conventional loans": (d.loan_type == 1).mean(),
            "FHA loans": (d.loan_type == 2).mean(), "VA loans": (d.loan_type == 3).mean(),
            "Median income (k$)": d.income_k.median(), "Median debt-to-income": d.debt_to_income.median(),
            "Median loan-to-value": d.loan_to_value.median()})),
    })
    show = prof.copy().astype(object)
    for r in ["Conventional loans", "FHA loans", "VA loans"]:
        show.loc[r] = prof.loc[r].map(lambda v: f"{v:.0%}")
    for r in ["Applicants", "Median income (k$)", "Median debt-to-income", "Median loan-to-value"]:
        show.loc[r] = prof.loc[r].map(lambda v: f"{v:,.0f}")
    with st.expander("Who are the people who differ?", expanded=True):
        st.dataframe(show, width="stretch")

if f_dev == "debt_to_income":
    st.markdown("**Try:** *Colour the groups by → Loan type.* Conventional loans jump right after 50% debt-to-income, "
                "government-backed FHA loans tolerate more, and VA loans barely react. "
                "**The single PDP averages three different rules into one curve that fits none of them.**")
core.question("**Why does this matter for the bank?** If you set a policy from the average (e.g. <i>'reject above 50% DTI'</i>), "
              "who would be treated wrongly?")
next_button(5, "One more catch: the independence assumption →")

# ================================================================= E. independence
st.divider()
st.header("E · The hidden assumption: features are independent")
st.markdown(
    "To draw a PDP or ICE line, we **change one feature and keep all others fixed**. "
    "That only makes sense if the feature can change **on its own**. But some features are tied together:\n\n"
    "**Loan-to-value = loan amount ÷ property value.**\n\n"
    "So when the PDP changes the *property value* of Person 1 from \\$655k to \\$150k, it keeps the **\\$655k loan** and the "
    "**100% loan-to-value**. That applicant cannot exist: a \\$655k loan on a \\$150k home is 437% loan-to-value, not 100%."
)

grid, P = curves["property_value"]
info_all = rows.reset_index(drop=True)
loans = info_all["loan_amount"].to_numpy()
too_big = loans[:, None] > grid[None, :]               # PDP asks: loan bigger than the home itself
share_by_x = too_big[SAMPLE].mean(0)                   # at each property value: share of the 500 such cases
share_imp = too_big[SAMPLE].mean()
real_share = (core.load_data()[1].eval("loan_amount > property_value")).mean()

_, test = core.load_data()
real = test.sample(4000, random_state=0)
p1 = info_all.iloc[0]

c1, c2 = st.columns(2, gap="large")
with c1:
    fig, ax = plt.subplots(figsize=(6.2, 5), dpi=130)
    ax.fill_between([0, 1500], [0, 1500], [1500, 1500], color=core.RED, alpha=0.06, label="Loan bigger than the home")
    ax.scatter(real.property_value / 1000, real.loan_amount / 1000, s=4, color=core.GREY, alpha=0.35, label="Real applicants")
    bad = too_big[0]
    ax.scatter(grid[~bad] / 1000, np.full((~bad).sum(), p1.loan_amount / 1000), s=36, color=core.BLUE, zorder=3,
               label="PDP points for Person 1")
    ax.scatter(grid[bad] / 1000, np.full(bad.sum(), p1.loan_amount / 1000), s=46, marker="x", color=core.RED,
               zorder=3, label="PDP points for Person 1: loan > home")
    ax.set_xlim(0, 1500); ax.set_ylim(0, 1300)
    ax.set_xlabel("Property value (k$)"); ax.set_ylabel("Loan amount (k$)")
    ax.set_title("What the PDP asks the model about", loc="left", color=core.NAVY, fontweight="bold", fontsize=11)
    core.style_axes(ax); ax.grid(color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.2)
    st.pyplot(fig, width="stretch")
with c2:
    pdp = P[SAMPLE].mean(0)
    fig, ax = plt.subplots(figsize=(6.2, 5), dpi=130)
    ax2 = ax.twinx()
    ax2.fill_between(grid / 1000, 0, share_by_x, color=core.RED, alpha=0.12, step="mid")
    ax2.set_ylim(0, 1); ax2.set_yticks([]); ax2.spines[:].set_visible(False)
    ax.plot(grid / 1000, pdp, color=core.NAVY, lw=3, label="PDP: property value", zorder=3)
    ax.fill_between([], [], color=core.RED, alpha=0.12, label="Share of PDP points where the loan > the home")
    ax.set_ylim(0, max(0.7, pdp.max() * 1.1))
    ax.set_xlabel("Property value (k$)"); ax.set_ylabel("Average predicted risk of denial")
    ax.set_title("The scary part of the PDP", loc="left", color=core.NAVY, fontweight="bold", fontsize=11)
    core.style_axes(ax); ax.grid(axis="y", color="#E5E7EB", lw=0.8); pct_axis(ax)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    st.pyplot(fig, width="stretch")

m1, m2 = st.columns(2)
m1.metric("PDP points where the loan is bigger than the home", f"{share_imp:.0%}")
m2.metric("Real applicants where the loan is bigger than the home", f"{real_share:.1%}")
st.caption("The left side of the property-value PDP is built mostly on applicants who borrow more than their home is worth, "
           "while still showing a loan-to-value of e.g. 95%. In the real data this is rare, so the model is guessing there.")

a = st.radio("So why does the PDP show such a high risk for cheap homes?", [
    "Cheap homes are genuinely very risky",
    "The model was asked about impossible applicants (big loans on cheap homes) and extrapolated",
    "The model is broken"], index=None, key="indep_q")
if a is not None:
    if a.startswith("The model was asked"):
        st.success("Exactly. The PDP forced the model into combinations it never saw. Its answers there are guesses, "
                   "and the average plot presents those guesses as facts.")
    else:
        st.warning("Look at the left chart again: where are the red crosses compared with the real applicants?")

if a is not None:
    core.question(
        "**Take-aways**<br>"
        "• A PDP shows the <b>average</b> effect; it can hide groups for whom the effect is very different. ICE plots reveal them.<br>"
        "• PDP and ICE assume you can change <b>one feature on its own</b>. With related features (loan, property value, loan-to-value) "
        "they show the model's answers for applicants who <b>don't exist</b>.<br>"
        "• Remedies: check correlations first; use methods that stay inside realistic data (e.g. <b>ALE plots</b>); "
        "or change related features <b>together</b>.<br><br>"
        "Next: instead of averages, <b>why did the model decide this for Person 1?</b>"
    )

if st.button("Start this step over"):
    st.session_state["pdp_stage"] = 1
    st.rerun()
