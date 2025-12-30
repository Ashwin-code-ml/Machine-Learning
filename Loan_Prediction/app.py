import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Prediction AI",
    page_icon="💰",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #2b1055, #4b1b78);
    color: white;
}
h1 {
    color: #FFD700;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(90deg, #FFD700, #C9A227);
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
}
.card {
    background-color: #3c1d6e;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
.result {
    background-color: #FFD700;
    color: black;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_prediction_pipeline.pkl")

# ---------------- TITLE ----------------
st.markdown("<h1>💰 Loan Approval Prediction</h1>", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Applicant Information")

person_age = st.slider("Age", 18, 70, 30)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox(
    "Education",
    ["high_school", "bachelor", "master", "doctorate"]
)
person_income = st.number_input("Annual Income", min_value=0, step=1000)
person_emp_exp = st.slider("Employment Experience (Years)", 0, 40, 5)
person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amnt = st.number_input("Loan Amount", min_value=1000, step=500)
loan_intent = st.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
     "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
)
loan_int_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0)
loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.25)

cb_person_cred_hist_length = st.slider(
    "Credit History Length (Years)", 0, 40, 5
)
credit_score = st.slider("Credit Score", 300, 850, 650)

previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default",
    ["Yes", "No"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Loan Status"):
    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(
            f"<div class='result'>✅ Loan Approved<br>Confidence: {probability:.2%}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result'>❌ Loan Rejected<br>Risk Score: {(1-probability):.2%}</div>",
            unsafe_allow_html=True
        )