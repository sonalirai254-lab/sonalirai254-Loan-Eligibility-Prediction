"""
Streamlit web app for Loan Eligibility Prediction.

Run:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")


st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.title("Loan Eligibility Predictor")
    st.write(
        "Enter the applicant's details below and click **Predict** to see whether "
        "the loan is likely to be approved."
    )

    model = load_model()
    if model is None:
        st.error(
            "Model not found. Please run `python loan_prediction.py` first to "
            "train and save the model."
        )
        st.stop()

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"]
            )

        with col2:
            applicant_income = st.number_input(
                "Applicant Income", min_value=0, value=5000, step=100
            )
            coapplicant_income = st.number_input(
                "Coapplicant Income", min_value=0, value=0, step=100
            )
            loan_amount = st.number_input(
                "Loan Amount (in thousands)", min_value=0, value=120, step=1
            )
            loan_amount_term = st.number_input(
                "Loan Amount Term (months)", min_value=0, value=360, step=12
            )
            credit_history = st.selectbox(
                "Credit History", [1.0, 0.0],
                format_func=lambda v: "Good (1)" if v == 1.0 else "Bad (0)",
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": loan_amount_term,
                    "Credit_History": credit_history,
                    "Property_Area": property_area,
                }
            ]
        )

        pred = int(model.predict(row)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(row)[0][1])

        st.subheader("Prediction Result")
        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        if proba is not None:
            st.caption(f"Approval probability: {proba * 100:.1f}%")


if __name__ == "__main__":
    main()
