import streamlit as st
import pickle
import numpy as np

# =========================
# LOAD ARTIFACTS
# =========================

model = pickle.load(open("artifacts/model.pkl", "rb"))

encoders = pickle.load(
    open("artifacts/encoders.pkl", "rb")
)

feature_columns = pickle.load(
    open("artifacts/feature_columns.pkl", "rb")
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Prediction App")

st.write(
    "Enter applicant details to predict loan approval."
)

# =========================
# INPUTS
# =========================

no_of_dependents = st.number_input(
    "No of Dependents",
    min_value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Income",
    min_value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=0
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900
)

residential_assets_value = st.number_input(
    "Residential Assets",
    min_value=0.0
)

commercial_assets_value = st.number_input(
    "Commercial Assets",
    min_value=0.0
)

luxury_assets_value = st.number_input(
    "Luxury Assets",
    min_value=0.0
)

bank_asset_value = st.number_input(
    "Bank Assets",
    min_value=0.0
)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict"):

    try:

        # =========================
        # INPUT DICTIONARY
        # =========================

        input_dict = {

            "no_of_dependents":
                no_of_dependents,

            "education":
                education,

            "self_employed":
                self_employed,

            "income_annum":
                income_annum,

            "loan_amount":
                loan_amount,

            "loan_term":
                loan_term,

            "cibil_score":
                cibil_score,

            "residential_assets_value":
                residential_assets_value,

            "commercial_assets_value":
                commercial_assets_value,

            "luxury_assets_value":
                luxury_assets_value,

            "bank_asset_value":
                bank_asset_value
        }

        # =========================
        # FEATURE ORDER FIX
        # =========================

        input_data = []

        for col in feature_columns:

            value = input_dict[col]

            # =========================
            # ENCODING
            # =========================

            if col in encoders:

                le = encoders[col]

                value = le.transform(
                    [str(value).strip()]
                )[0]

            input_data.append(float(value))

        # =========================
        # NUMPY ARRAY
        # =========================

        input_array = np.array(
            input_data,
            dtype=np.float32
        ).reshape(1, -1)

        # =========================
        # PREDICTION
        # =========================

        prediction = model.predict(
            input_array
        )

        prediction = int(
            prediction[0]
        )

        # =========================
        # RESULT
        # =========================

        if prediction == 1:

            st.success(
                "✅ Loan Approved"
            )

        else:

            st.error(
                "❌ Loan Rejected"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )