"""Entry point. Run with:  streamlit run app.py

To add a new step: create a file in steps/ and add ONE line to the right section below.
"""
import streamlit as st

st.set_page_config(page_title="Who gets a mortgage? | XAI", page_icon="🏠", layout="wide")

pages = {
    "": [
        st.Page("steps/0_start.py", title="Start: the case", icon="🏠", default=True),
    ],
    "Predictive AI": [
        st.Page("steps/1_meet_the_model.py", title="1 · Meet the model", icon="🤖"),
        st.Page("steps/2_be_the_bank.py", title="2 · Be the bank", icon="🧑‍⚖️"),
        st.Page("steps/3_feature_importance.py", title="3 · What matters to the model?", icon="📊"),
        st.Page("steps/4_pdp_ice.py", title="4 · How does it matter?", icon="📈"),
        st.Page("steps/5_shap.py", title="5 · Why this person?", icon="🔍"),
        st.Page("steps/6_stability.py", title="6 · Can we trust the explanation?", icon="🧪"),
        # st.Page("steps/7_counterfactuals.py", title="7 · What would change it?", icon="🎚️"),
        # st.Page("steps/8_bias.py", title="8 · Is it fair?", icon="⚖️"),
    ],
    "Generative AI": [
        st.Page("steps/g1_sources.py", title="Where did the answer come from?", icon="📚"),
    ],
    "Agentic AI": [
        st.Page("steps/g2_agent_trace.py", title="Agent detective: what went wrong?", icon="🕵️"),
    ],
}

st.navigation(pages).run()
