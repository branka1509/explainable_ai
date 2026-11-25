"""Precompute heavy results once (run locally, commit the outputs in artifacts/).

    python precompute.py
"""
import json
from sklearn.inspection import permutation_importance
import core

model = core.load_model()
_, test = core.load_data()
X, y = test[core.FEATURES], test["denied"]

perm = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
out = {
    "gini": dict(zip(core.FEATURES, map(float, model.feature_importances_))),
    "permutation_mean": dict(zip(core.FEATURES, map(float, perm.importances_mean))),
    "permutation_std": dict(zip(core.FEATURES, map(float, perm.importances_std))),
}
with open(core.ART / "importance.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved artifacts/importance.json")

# ---------------------------------------------------------------- PDP / ICE curves (step 4)
import numpy as np
import pandas as pd

ten = core.ten_applicants().reset_index()
sample = test.sample(500, random_state=1)
rows = pd.concat([ten[["applicant_id"] + core.FEATURES], sample[["applicant_id"] + core.FEATURES]], ignore_index=True)
ice = {"applicant_id": rows["applicant_id"].to_numpy(), "n_ten": np.array(len(ten))}
for f in core.FEATURES:
    if f in core.CATEGORICAL:
        grid = np.array(sorted(test[f].unique()), dtype=float)
    elif f == "debt_to_income":
        grid = np.array(sorted(test[f].unique()), dtype=float)
    else:
        grid = np.unique(np.round(np.quantile(test[f], np.linspace(0.02, 0.98, 30)), 2))
    X = pd.concat([rows[core.FEATURES].assign(**{f: v}) for v in grid], ignore_index=True)
    P = model.predict_proba(X)[:, 1].reshape(len(grid), len(rows)).T
    ice[f"{f}__grid"] = grid
    ice[f"{f}__ice"] = P.astype(np.float32)
np.savez_compressed(core.ART / "ice.npz", **ice)
print("saved artifacts/ice.npz")

# ---------------------------------------------------------------- SHAP values (step 5)
# Interventional TreeSHAP on the probability scale: "unknown" features are filled in from 1,000 training applicants.
import shap
train, _ = core.load_data()
background = train[core.FEATURES].sample(1000, random_state=0)
explainer = shap.TreeExplainer(model, data=shap.maskers.Independent(background, max_samples=1000),
                               model_output="probability", feature_perturbation="interventional")
global_rows = test.sample(1000, random_state=2)
ten_rows = core.ten_applicants().reset_index()
X = pd.concat([ten_rows[core.FEATURES], global_rows[core.FEATURES]], ignore_index=True).astype(float)
sv = explainer.shap_values(X)
np.savez_compressed(core.ART / "shap.npz",
                    values=sv.astype(np.float32), X=X.to_numpy(np.float32),
                    base=np.array(float(explainer.expected_value)), n_ten=np.array(len(ten_rows)),
                    applicant_id=np.concatenate([ten_rows["applicant_id"].to_numpy(), global_rows["applicant_id"].to_numpy()]))
print("saved artifacts/shap.npz")

# ---------------------------------------------------------------- SHAP stability (step 6)
# (a) small, realistic changes to each of the 10 applicants; linked values move together
STAB_KINDS = ["debt_to_income", "income_k", "loan_amount", "property_value"]
STAB_STEPS = {"debt_to_income": [-3, -2, -1, 0, 1, 2, 3],          # percentage points
              "income_k": [-10, -5, -2, 0, 2, 5, 10],               # percent
              "loan_amount": [-25, -10, -5, 0, 5, 10, 25],          # thousand $
              "property_value": [-25, -10, -5, 0, 5, 10, 25]}       # thousand $

def perturb(row, kind, step):
    r = row.copy()
    if kind == "debt_to_income":
        r["debt_to_income"] = row["debt_to_income"] + step
    elif kind == "income_k":                                        # same debts, different income -> DTI moves too
        r["income_k"] = row["income_k"] * (1 + step / 100)
        r["debt_to_income"] = row["debt_to_income"] / (1 + step / 100)
    elif kind == "loan_amount":                                     # loan-to-value follows the loan
        r["loan_amount"] = row["loan_amount"] + step * 1000
        r["loan_to_value"] = r["loan_amount"] / row["property_value"] * 100
    elif kind == "property_value":                                  # loan-to-value follows the property value
        r["property_value"] = row["property_value"] + step * 1000
        r["loan_to_value"] = row["loan_amount"] / r["property_value"] * 100
    return r

rows = []
for _, r in ten_rows.iterrows():
    base_row = r[core.FEATURES].astype(float)
    for k in STAB_KINDS:
        for s in STAB_STEPS[k]:
            rows.append(perturb(base_row, k, s))
Xs = pd.DataFrame(rows)[core.FEATURES]
sv_s = explainer.shap_values(Xs)
n_p, n_k, n_s = len(ten_rows), len(STAB_KINDS), 7
# (b) the same 10 applicants explained against 5 different reference groups of 100 applicants
ref_vals, ref_base = [], []
for seed in range(5):
    e = shap.TreeExplainer(model, data=train[core.FEATURES].sample(100, random_state=100 + seed),
                           model_output="probability", feature_perturbation="interventional")
    ref_vals.append(e.shap_values(ten_rows[core.FEATURES].astype(float)))
    ref_base.append(float(e.expected_value))
np.savez_compressed(
    core.ART / "stability.npz",
    kinds=np.array(STAB_KINDS), steps=np.array([STAB_STEPS[k] for k in STAB_KINDS]),
    X=Xs.to_numpy(np.float32).reshape(n_p, n_k, n_s, -1),
    shap=sv_s.astype(np.float32).reshape(n_p, n_k, n_s, -1),
    p=model.predict_proba(Xs)[:, 1].astype(np.float32).reshape(n_p, n_k, n_s),
    ref_shap=np.array(ref_vals, dtype=np.float32), ref_base=np.array(ref_base),
)
print("saved artifacts/stability.npz")
