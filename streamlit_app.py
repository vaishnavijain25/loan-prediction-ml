import streamlit as st
import requests

st.title("🏦 Loan Prediction App")

# -------- INPUT FIELDS --------

no_of_dependents = st.number_input("No of Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term", min_value=0)

cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets", min_value=0.0)
commercial_assets_value = st.number_input("Commercial Assets", min_value=0.0)
luxury_assets_value = st.number_input("Luxury Assets", min_value=0.0)
bank_asset_value = st.number_input("Bank Assets", min_value=0.0)

# -------- BUTTON --------

if st.button("Predict"):

    # ✅ Better validation
    if income_annum <= 0 or loan_amount <= 0 or cibil_score <= 0:
        st.warning("⚠️ Please fill all required fields properly")
    else:
        url = "http://127.0.0.1:5000/predict"

        data = {
            "no_of_dependents": int(no_of_dependents),
            "education": education,
            "self_employed": self_employed,
            "income_annum": float(income_annum),
            "loan_amount": float(loan_amount),
            "loan_term": int(loan_term),
            "cibil_score": int(cibil_score),
            "residential_assets_value": float(residential_assets_value),
            "commercial_assets_value": float(commercial_assets_value),
            "luxury_assets_value": float(luxury_assets_value),
            "bank_asset_value": float(bank_asset_value)
        }

        try:
            response = requests.post(url, json=data)

            # ✅ Check status
            if response.status_code != 200:
                st.error("❌ API Error")
                st.stop()

            result = response.json()

            # ✅ Prediction output
            if "prediction" in result:
                if result["prediction"] == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Rejected")
            else:
                st.error(result.get("error", "Something went wrong"))

        except requests.exceptions.ConnectionError:
            st.error("❌ Flask API is not running. Please start app.py")

        except Exception as e:
            st.error(f"Error: {e}")
