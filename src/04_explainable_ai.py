"""Giải thích mô hình bằng SHAP."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR,
    REPORT_DIR,
    X_TEST,
    X_TRAIN,
)
from utils import ensure_dir


def explain_with_shap(model_path: Path | None = None, sample_size: int = 500) -> None:
    """Tạo SHAP summary plot cho mô hình Random Forest hoặc XGBoost."""
    if model_path is None:
        model_path = MODELS_DIR / "random_forest_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    model = joblib.load(model_path)

    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)

    # Lấy mẫu để giải thích
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    print("Tính SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    ensure_dir(REPORT_DIR)

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    summary_path = REPORT_DIR / "shap_summary.png"
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu SHAP summary tại: {summary_path}")

    # SHAP bar plot (tầm quan trọng trung bình)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    bar_path = REPORT_DIR / "shap_bar.png"
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu SHAP bar plot tại: {bar_path}")


if __name__ == "__main__":
    explain_with_shap()
