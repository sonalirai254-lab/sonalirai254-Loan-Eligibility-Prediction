"""Model saving / loading helpers."""

import os
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")


def save_model(model, path: str = MODEL_PATH) -> str:
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)
