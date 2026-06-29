"""Ensemble VotingClassifier kết hợp Random Forest và XGBoost."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, REPORT_DIR, X_TEST, X_TRAIN, Y_TEST, Y_TRAIN
from utils import ensure_dir


def train_ensemble(X_train: pd.DataFrame, y_train: pd.Series) -> VotingClassifier:
    """Train soft-voting ensemble from RF and XGBoost."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_ensemble(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate ensemble."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    classes = sorted(y_test.unique())
    y_bin = label_binarize(y_test, classes=classes)
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
    }

    print("\n=== ENSEMBLE RESULTS ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def main() -> None:
    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN).squeeze()
    y_test = pd.read_csv(Y_TEST).squeeze()

    model = train_ensemble(X_train, y_train)
    metrics = evaluate_ensemble(model, X_test, y_test)

    ensure_dir(MODELS_DIR)
    joblib.dump(model, MODELS_DIR / "ensemble_model.pkl")

    ensure_dir(REPORT_DIR)
    with open(REPORT_DIR / "ensemble_metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== Ensemble Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")

    print("\nĐã lưu ensemble model và report.")


if __name__ == "__main__":
    main()
