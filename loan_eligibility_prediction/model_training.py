"""Train Logistic Regression and Random Forest, pick the better one."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from evaluation import evaluate


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
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


def train_and_select(X_train, X_test, y_train, y_test):
    """Train both models, evaluate them, return (best_name, best_model, scores)."""
    models = build_models()
    scores = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        scores[name] = evaluate(name, pipe, X_test, y_test)

    best_name = max(scores, key=scores.get)
    print(f"\nBest model: {best_name} (accuracy={scores[best_name]:.4f})")
    return best_name, models[best_name], scores
