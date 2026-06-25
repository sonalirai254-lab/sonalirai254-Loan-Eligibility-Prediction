"""
Loan Eligibility Prediction
---------------------------
Loads loan_data.csv, preprocesses the data, trains Logistic Regression and
Random Forest models, evaluates both, and saves the better model to model.pkl.

Run:
    python loan_prediction.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "loan_data.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")

TARGET = "Loan_Status"
CATEGORICAL = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC),
            ("cat", categorical_pipeline, CATEGORICAL),
        ]
    )


def evaluate(name: str, model, X_test, y_test) -> float:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification report:")
    print(classification_report(y_test, preds))
    return acc


def main() -> None:
    print(f"Loading data from {DATA_PATH}")
    df = load_data(DATA_PATH)

    # Target: 'Y' -> 1 (approved), 'N' -> 0 (rejected)
    y = df[TARGET].map({"Y": 1, "N": 0}).astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=200, random_state=42),
                ),
            ]
        ),
    }

    scores = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        scores[name] = evaluate(name, pipe, X_test, y_test)

    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (accuracy={scores[best_name]:.4f})")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
