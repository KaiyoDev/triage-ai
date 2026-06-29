"""Huấn luyện XGBoost và so sánh với Random Forest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
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

# XGBoost import with fallback
xgboost_available = True
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    xgboost_available = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR,
    REPORT_DIR,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
)
from utils import ensure_dir


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> tuple["XGBClassifier", dict[int, int]]:
    """Train XGBoost classifier, remapping labels to 0-based."""
    if not xgboost_available:
        raise ImportError("xgboost chưa được cài đặt. Chạy: pip install xgboost")

    print("Đang huấn luyện XGBoost...")
    unique_labels = sorted(y_train.unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train_mapped = y_train.map(label_map)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train_mapped)
    print("Huấn luyện XGBoost hoàn tất.")
    return model, label_map


def evaluate_xgboost(
    model: "XGBClassifier",
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Đánh giá mô hình XGBoost."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    classes = sorted(y_test.unique())
    if len(classes) > 2:
        y_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
    else:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
    }

    print("\n=== KẾT QUẢ ĐÁNH GIÁ XGBOOST ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def save_artifacts(model: "XGBClassifier", metrics: dict) -> None:
    """Lưu model và báo cáo."""
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORT_DIR)

    model_path = MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nĐã lưu model tại: {model_path}")

    report_path = REPORT_DIR / "xgboost_metrics.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== XGBoost Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")
    print(f"Đã lưu báo cáo tại: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện XGBoost Triage AI")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    if not X_TRAIN.exists():
        raise FileNotFoundError("Chưa có dữ liệu train. Chạy 01_preprocessing.py trước.")

    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN).squeeze()
    y_test = pd.read_csv(Y_TEST).squeeze()

    model, label_map = train_xgboost(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    inv_label_map = {v: k for k, v in label_map.items()}
    y_test_mapped = y_test.map(label_map)

    metrics = evaluate_xgboost(model, X_test, y_test_mapped)
    save_artifacts(model, metrics)
    print(f"Label map: {label_map}")


if __name__ == "__main__":
    main()
