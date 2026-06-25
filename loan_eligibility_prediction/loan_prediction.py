"""
End-to-end pipeline that orchestrates every module:

  1. Data loading
  2. Data preprocessing
  3. Model training (LR + RF) and selection
  4. Model evaluation (printed inside training)
  5. Model saving

Run:
    python loan_prediction.py
"""

from data_loader import load_data, show_dataset_info
from preprocessing import split_features_target, train_test
from model_training import train_and_select
from model_io import save_model


def main() -> None:
    # 1. Data loading
    df = load_data()
    show_dataset_info(df)

    # 2. Preprocessing (encoding + imputation live inside the model pipeline)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test(X, y)

    # 3 + 4. Train, evaluate, select best
    _, best_model, _ = train_and_select(X_train, X_test, y_train, y_test)

    # 5. Save
    save_model(best_model)


if __name__ == "__main__":
    main()
