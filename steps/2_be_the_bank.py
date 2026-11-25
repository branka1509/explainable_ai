import streamlit as st
import core

core.setup("2 · Be the bank", "Ten real applicants the model has never seen")

ten = core.ten_applicants()

st.markdown("Look at each applicant and decide: **approve or reject?** "
            "When you have decided on all ten, reveal what the **model** predicted and what the **bank** actually did.")

revealed = st.session_state.get("bank_revealed", False)

rows = [ten.index[:5], ten.index[5:]]
for row in rows:
    cols = st.columns(5, gap="small")
    for col, person in zip(cols, row):
        a = ten.loc[person]
        with col.container(border=True):
            st.markdown(f"**{person}**")
            st.markdown(
                f"Income **{core.fmt_k(a.income_k * 1000)}**  \n"
                f"Loan **{core.fmt_k(a.loan_amount)}**  \n"
                f"Property **{core.fmt_k(a.property_value)}**  \n"
                f"Loan-to-value **{a.loan_to_value:.0f}%**  \n"
                f"Debt-to-income **{a.debt_to_income:.0f}%**  \n"
                f"{a.loan_type_name} loan · age {a.applicant_age_band}"
            )
            st.segmented_control("Your decision", ["Approve", "Reject"], key=f"vote_{person}",
                                 label_visibility="collapsed", disabled=revealed)
            if revealed:
                pd_ = a.p_denied
                colour = core.RED if pd_ >= 0.5 else core.BLUE
                st.markdown(f"<div style='font-size:1.6rem;font-weight:800;color:{colour}'>{pd_:.0%}</div>"
                            f"<span class='muted'>model: chance of denial</span>", unsafe_allow_html=True)
                st.markdown(f"Model: **{'Reject' if pd_ >= 0.5 else 'Approve'}**  \n"
                            f"Bank: **{'Denied' if a.denied == 1 else 'Approved'}**")

votes = {p: st.session_state.get(f"vote_{p}") for p in ten.index}
n_done = sum(v is not None for v in votes.values())

st.write("")
if not revealed:
    st.progress(n_done / 10, text=f"{n_done} of 10 decided")
    if st.button("Reveal the model and the bank's decisions", type="primary", disabled=n_done < 10):
        st.session_state["bank_revealed"] = True
        st.rerun()
    if n_done < 10:
        st.caption("Decide on all ten first.")
    st.stop()

# ---- after the reveal
model_says = {p: "Reject" if ten.loc[p, "p_denied"] >= 0.5 else "Approve" for p in ten.index}
bank_says = {p: "Reject" if ten.loc[p, "denied"] == 1 else "Approve" for p in ten.index}
agree_bank = sum(votes[p] == bank_says[p] for p in ten.index)
agree_model = sum(votes[p] == model_says[p] for p in ten.index)
model_bank = sum(model_says[p] == bank_says[p] for p in ten.index)

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("You agreed with the bank", f"{agree_bank} / 10")
m2.metric("You agreed with the model", f"{agree_model} / 10")
m3.metric("Model agreed with the bank", f"{model_bank} / 10")

core.question(
    "**Discuss**<br>"
    "• Why does the model give one person 2% and another 98%?<br>"
    "• Pick a rejected applicant. What would you <b>tell them</b> if they asked <i>why</i>?<br>"
    "• Where the model and the bank disagree: who is right?<br>"
    "• <b>Person 1 and Person 10</b> both have a 55% debt-to-income ratio. Why 80% vs 6%?"
)
st.markdown("**The model only gives us a number. It does not tell us why.**")

if st.button("Start over"):
    for p in ten.index:
        st.session_state.pop(f"vote_{p}", None)
    st.session_state["bank_revealed"] = False
    st.rerun()

st.page_link("steps/3_feature_importance.py", label="Next: what matters to the model? →", icon="📊")
