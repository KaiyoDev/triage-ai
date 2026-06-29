"""Huấn luyện mô hình Random Forest cho phân loại mức độ ưu tiên."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR,
    REPORT_DIR,
    TARGET,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
)
from utils import ensure_dir


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
) -> RandomForestClassifier:
    """Train Random Forest với tham số tối ưu."""
    print("Đang huấn luyện Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Huấn luyện hoàn tất.")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Đánh giá mô hình."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    classes = sorted(y_test.unique())
    if len(classes) > 2:
        y_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
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

    print("\n=== KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 3) -> None:
    """Stratified K-Fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    print(f"\n=== Stratified {cv}-Fold CV ===")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_cv, y_train_cv)
        score = model.score(X_val_cv, y_val_cv)
        scores.append(score)
        print(f"Fold {fold}: accuracy = {score:.4f}")
    print(f"Mean accuracy: {sum(scores)/len(scores):.4f}")


def save_artifacts(model: RandomForestClassifier, metrics: dict) -> None:
    """Lưu model và báo cáo."""
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORT_DIR)

    model_path = MODELS_DIR / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nĐã lưu model tại: {model_path}")

    report_path = REPORT_DIR / "random_forest_metrics.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Random Forest Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")
    print(f"Đã lưu báo cáo tại: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Random Forest Triage AI")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    if not X_TRAIN.exists():
        raise FileNotFoundError("Chưa có dữ liệu train. Chạy 01_preprocessing.py trước.")

    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN).squeeze()
    y_test = pd.read_csv(Y_TEST).squeeze()

    # Stratified K-Fold on train set
    cross_validate(X_train, y_train, cv=5)

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)


if __name__ == "__main__":
    main()
