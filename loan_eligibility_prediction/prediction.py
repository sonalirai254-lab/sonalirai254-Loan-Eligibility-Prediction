"""Prediction system: load the saved model and predict on a new applicant."""

import os
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python loan_prediction.py` first."
        )
    return joblib.load(path)


def predict_applicant(applicant: dict, model=None) -> dict:
    """
    Predict loan approval for a single applicant dict.

    Expected keys: Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
    Credit_History, Property_Area.
    """
    if model is None:
        model = load_model()
    row = pd.DataFrame([applicant])
    pred = int(model.predict(row)[0])
    proba = (
        float(model.predict_proba(row)[0][1])
        if hasattr(model, "predict_proba")
        else None
    )
    return {
        "prediction": "Loan Approved" if pred == 1 else "Loan Rejected",
        "approved": bool(pred),
        "probability": proba,
    }


if __name__ == "__main__":
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 130,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }
    print(predict_applicant(sample))
