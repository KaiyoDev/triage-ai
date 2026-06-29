"""Tuning hyperparameter và so sánh mô hình bằng GridSearchCV."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, REPORT_DIR, X_TEST, X_TRAIN, Y_TEST, Y_TRAIN
from utils import ensure_dir


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Tuning Random Forest hyperparameters."""
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    scorer = make_scorer(f1_score, average="weighted", zero_division=0)
    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)

    print(f"Best RF params: {grid.best_params_}")
    print(f"Best RF F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Tuning XGBoost hyperparameters."""
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    scorer = make_scorer(f1_score, average="weighted", zero_division=0)
    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)

    print(f"Best XGB params: {grid.best_params_}")
    print(f"Best XGB F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_tuned(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> None:
    """In kết quả đánh giá chi tiết."""
    y_pred = model.predict(X_test)
    print(f"\n=== {name} TUNED ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")


def main() -> None:
    if not X_TRAIN.exists():
        raise FileNotFoundError("Chưa có dữ liệu train. Chạy 01_preprocessing.py trước.")

    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN).squeeze()
    y_test = pd.read_csv(Y_TEST).squeeze()

    print("Tuning Random Forest...")
    rf_best = tune_random_forest(X_train, y_train)
    evaluate_tuned(rf_best, X_test, y_test, "Random Forest")

    print("\nTuning XGBoost...")
    xgb_best = tune_xgboost(X_train, y_train)
    evaluate_tuned(xgb_best, X_test, y_test, "XGBoost")

    ensure_dir(MODELS_DIR)
    joblib.dump(rf_best, MODELS_DIR / "random_forest_tuned.pkl")
    joblib.dump(xgb_best, MODELS_DIR / "xgboost_tuned.pkl")

    print("\nĐã lưu model tuned.")


if __name__ == "__main__":
    main()
