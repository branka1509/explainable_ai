"""Render the classic SHAP plots (waterfall per applicant, beeswarm, bar) as images.  python make_shap_plots.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import core

OUT = core.ART / "shap_plots"; OUT.mkdir(exist_ok=True)
d = np.load(core.ART / "shap.npz", allow_pickle=True)
vals, X, base, n_ten = d["values"], d["X"], float(d["base"]), int(d["n_ten"])
names = [core.NAMES[f] for f in core.FEATURES]

def fmt(f, v):
    if f in core.CATEGORICAL: return core.CATEGORICAL[f].get(int(round(v)), str(v))
    if f in ("loan_amount", "property_value"): return f"${v/1000:,.0f}k"
    if f == "income_k": return f"${v:,.0f}k"
    if f in ("loan_to_value", "debt_to_income"): return f"{v:.0f}%"
    if f == "tract_income_pct_of_area": return f"{v:.0f}%"
    if f == "loan_term_months": return f"{v:.0f} mo"
    return f"{v:.0f}"

people = list(core.ten_applicants().index)
for i, p in enumerate(people):
    disp = np.array([fmt(f, X[i, k]) for k, f in enumerate(core.FEATURES)], dtype=object)
    e = shap.Explanation(values=vals[i], base_values=base, data=X[i], display_data=disp, feature_names=names)
    shap.plots.waterfall(e, max_display=14, show=False)
    fig = plt.gcf(); fig.set_size_inches(8, 6.2)
    plt.title(f"{p}: why the model predicts {base + vals[i].sum():.0%}", loc="left", fontsize=12)
    plt.savefig(OUT / f"waterfall_{i+1}.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close("all")

g = shap.Explanation(values=vals[n_ten:], base_values=np.full(len(vals) - n_ten, base), data=X[n_ten:], feature_names=names)
shap.plots.beeswarm(g, max_display=14, show=False)
fig = plt.gcf(); fig.set_size_inches(9, 6.5)
plt.title("SHAP summary: gradient boosting, 1,000 test applicants", fontsize=12)
plt.xlabel("SHAP value (change in predicted probability of denial)")
plt.savefig(OUT / "beeswarm.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close("all")

shap.plots.bar(g, max_display=14, show=False)
fig = plt.gcf(); fig.set_size_inches(8, 6)
plt.title("Mean |SHAP value| over 1,000 applicants", fontsize=12)
plt.savefig(OUT / "bar.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close("all")
print("saved", sorted(p.name for p in OUT.iterdir()))
