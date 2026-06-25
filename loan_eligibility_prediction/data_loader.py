"""Data loading utilities."""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "loan_data.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the loan dataset from CSV and drop the ID column."""
    df = pd.read_csv(path)
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    return df


def show_dataset_info(df: pd.DataFrame) -> None:
    """Print sample rows and basic dataset info."""
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    show_dataset_info(load_data())
