import streamlit as st
import joblib
import numpy as np

st.title("🏦 Loan Prediction App")

# Load model
model = joblib.load("artifacts/model.pkl")

# Inputs
no_of_dependents = st.number_input("No of Dependents", 0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Income", 0.0)
loan_amount = st.number_input("Loan Amount", 0.0)
loan_term = st.number_input("Loan Term", 0)

cibil_score = st.number_input("CIBIL Score", 300, 900)
residential_assets_value = st.number_input("Residential Assets", 0.0)
commercial_assets_value = st.number_input("Commercial Assets", 0.0)
luxury_assets_value = st.number_input("Luxury Assets", 0.0)
bank_asset_value = st.number_input("Bank Assets", 0.0)

# Convert categorical
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if st.button("Predict"):

    features = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")