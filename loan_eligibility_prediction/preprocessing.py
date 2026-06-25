"""Preprocessing: missing values, encoding, feature/target split, train/test split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def split_features_target(df: pd.DataFrame):
    """Separate features (X) and target (y). Target: Y -> 1, N -> 0."""
    y = df[TARGET].map({"Y": 1, "N": 0}).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Pipeline: impute missing + scale numerics, impute + one-hot encode categoricals."""
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


def train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
