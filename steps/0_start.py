import streamlit as st
import core

core.setup("Who gets a mortgage?", "Explainable AI · a hands-on case")

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(
        """
A bank uses an **AI model** to predict whether a **mortgage application will be denied**.
The model is accurate. But can we **understand** it, **trust** it, and **explain** its decisions
to customers, regulators and the board?

In this app you will work with **real data**: 60,694 home-purchase mortgage applications
from Maryland (USA) in 2023, published under the U.S. Home Mortgage Disclosure Act.

**How it works**
- Use the menu on the left. Go through the steps **in order**.
- Some steps are **locked**. Your instructor gives you the code when it's time.
- There are no wrong clicks. Nothing you do here changes the model.
        """
    )
with right:
    core.question("**Your role today**<br>You sit on the bank's AI committee. "
                  "The data science team wants to deploy this model. Do you sign off?")
    st.markdown("**The steps**")
    st.markdown(
        """
1. **Meet the model**: how good is it?
2. **Be the bank**: decide on 10 real applicants
3. **What matters to the model?**: feature importance
4. **How does it matter?**: PDP and ICE plots
5. **Why this person?**: SHAP
6. **Can we trust the explanation?**: stability

*More steps unlock as the day continues.*
        """
    )

st.page_link("steps/1_meet_the_model.py", label="Start with step 1 →", icon="🤖")
