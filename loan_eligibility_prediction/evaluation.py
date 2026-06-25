"""Evaluation metrics: accuracy, confusion matrix, classification report."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
