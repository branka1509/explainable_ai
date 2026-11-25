"""Shared building blocks for every page: data, model, the 10 applicants, styling, unlock codes."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 12, "font.family": "sans-serif"})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"

# ---------------------------------------------------------------- features
FEATURES = ["loan_amount", "property_value", "loan_to_value", "income_k", "debt_to_income",
            "loan_term_months", "loan_type", "occupancy", "manufactured_home",
            "applicant_age", "has_coapplicant", "tract_income_pct_of_area", "tract_median_housing_age"]

NAMES = {
    "loan_amount": "Loan amount", "property_value": "Property value", "loan_to_value": "Loan-to-value",
    "income_k": "Income", "debt_to_income": "Debt-to-income", "loan_term_months": "Loan term",
    "loan_type": "Loan type", "occupancy": "Occupancy", "manufactured_home": "Manufactured home",
    "applicant_age": "Applicant age", "has_coapplicant": "Co-applicant",
    "tract_income_pct_of_area": "Neighbourhood income", "tract_median_housing_age": "Neighbourhood housing age",
}

CATEGORICAL = {
    "loan_type": {1: "Conventional", 2: "FHA", 3: "VA", 4: "USDA rural"},
    "occupancy": {1: "Own home", 2: "2nd home", 3: "Investment"},
    "manufactured_home": {0: "No", 1: "Yes"},
    "has_coapplicant": {0: "No", 1: "Yes"},
}

UNITS = {
    "loan_amount": "$", "property_value": "$", "loan_to_value": "%", "income_k": "k$", "debt_to_income": "%",
    "loan_term_months": "months", "applicant_age": "years", "tract_income_pct_of_area": "% of area median",
    "tract_median_housing_age": "years",
}

# ---------------------------------------------------------------- colours
BLUE, NAVY, LIGHT, GREY, RED = "#002FA8", "#002060", "#D8E1F7", "#8A8F98", "#C00000"

# ---------------------------------------------------------------- unlock codes
# The codes are NOT stored in this (public) code. They live in Streamlit secrets:
#   locally: .streamlit/secrets.toml (git-ignored)   online: Streamlit Cloud -> app settings -> Secrets
#   presenter_key = "..."            -> open everything with ?presenter=<presenter_key>
#   [unlock_codes]
#   importance = "..."   pdp = "..."   shap = "..."   stable = "..."   sources = "..."   trace = "..."
# A page key listed below stays locked until its code is set in secrets (the presenter view still opens it).
LOCKED_PAGES = ["importance", "pdp", "shap", "stable", "sources", "trace"]


def _secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets file at all
        return default


def is_presenter() -> bool:
    """True when the URL carries ?presenter=<presenter_key from secrets>."""
    key = _secret("presenter_key")
    return key is not None and st.query_params.get("presenter") == str(key)


# ================================================================= data & model
@st.cache_data
def load_data():
    return pd.read_csv(DATA / "hmda_train.csv"), pd.read_csv(DATA / "hmda_test.csv")


@st.cache_resource(show_spinner="Loading the credit model ...")
def load_model():
    """Load the saved model; if the scikit-learn version differs, retrain the identical model (~30 s, once)."""
    try:
        with open(ART / "credit_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        if bundle["metrics"]["sklearn_version"] == sklearn.__version__:
            return bundle["model"]
    except Exception:
        pass
    train, _ = load_data()
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                      random_state=42).fit(train[FEATURES], train["denied"])


@st.cache_data
def test_predictions():
    _, test = load_data()
    return load_model().predict_proba(test[FEATURES])[:, 1]


@st.cache_data
def ten_applicants():
    """The 10 test-set applicants used throughout the class (fixed IDs, same order every time)."""
    _, test = load_data()
    ids = pd.read_csv(ART / "ten_applicants.csv")
    df = ids.merge(test, on="applicant_id", how="left").set_index("person")
    df["p_denied"] = load_model().predict_proba(df[FEATURES])[:, 1]
    return df


@st.cache_data
def importance():
    with open(ART / "importance.json") as f:
        return json.load(f)


# ================================================================= page helpers
def setup(title: str, subtitle: str | None = None):
    """Common page header + styling. Call at the top of every page."""
    st.logo(str(ROOT / "assets" / "xai_logo.png"))
    st.markdown(
        f"""
        <style>
        h1, h2, h3 {{ color: {BLUE}; }}
        .stMetric [data-testid="stMetricValue"] {{ color: {BLUE}; font-weight: 700; }}
        .big-number {{ font-size: 2.4rem; font-weight: 800; color: {BLUE}; line-height: 1.1; }}
        .muted {{ color: {GREY}; font-size: 0.9rem; }}
        .question {{ background: {LIGHT}; border-radius: 8px; padding: 0.9rem 1.1rem; color: {NAVY}; margin: 0.4rem 0 0.8rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(title)
    if subtitle:
        st.markdown(f"<span class='muted'>{subtitle}</span>", unsafe_allow_html=True)


def question(text: str):
    import re
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(f"<div class='question'>{html}</div>", unsafe_allow_html=True)


def unlocked(page_key: str) -> bool:
    """Gate a page behind a code the presenter announces. ?presenter=<presenter_key> opens everything."""
    if is_presenter():
        return True
    code = dict(_secret("unlock_codes", {}) or {}).get(page_key)
    if st.session_state.get(f"unlocked_{page_key}") or (code is None and page_key not in LOCKED_PAGES):
        return True
    st.info("🔒 This step opens when your instructor gives you the code.")
    entered = st.text_input("Code", key=f"code_{page_key}", type="password")
    if entered:
        if code is not None and entered.strip().lower() == str(code).lower():
            st.session_state[f"unlocked_{page_key}"] = True
            st.rerun()
        else:
            st.error("Not quite. Wait for the instructor's code.")
    return False


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_k(x):
    return f"${x/1000:,.0f}k"
