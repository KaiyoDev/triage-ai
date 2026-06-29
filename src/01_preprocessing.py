"""Bước tiền xử lý dữ liệu theo 7 bước chuẩn.

Bước 1: Thu thập dữ liệu
Bước 2: Làm sạch dữ liệu
Bước 3: Xử lý dữ liệu thiếu
Bước 4: Chuẩn hóa dữ liệu
Bước 5: Mã hóa dữ liệu
Bước 6: Chọn đặc trưng
Bước 7: Xuất dữ liệu phục vụ huấn luyện
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Nhúng config đường dẫn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURES,
    FINAL_DATASET,
    RAW_DATA_FILE,
    SCALER,
    TARGET,
    TEST_SIZE,
    TRIAGE_LEVELS,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
)
from utils import ensure_dir


def load_raw_data(path: Path) -> pd.DataFrame:
    """Bước 1: Thu thập dữ liệu."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    df = pd.read_csv(path)
    print(f"Đã đọc {len(df)} dòng, {len(df.columns)} cột.")
    return df


def _infer_required_columns(df: pd.DataFrame) -> dict[str, str]:
    """Ánh xạ tên cột gốc sang tên chuẩn."""
    mapping: dict[str, str] = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    def find(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    found = {
        "age": find("age"),
        "gender": find("gender", "sex"),
        "heart_rate": find("heart_rate", "heartrate", "heart rate", "hr"),
        "respiratory_rate": find("respiratory_rate", "respiratoryrate", "respiratory rate", "rr"),
        "temperature": find("temperature", "body_temperature", "body temperature"),
        "spo2": find("spo2", "oxygen_saturation", "oxygen saturation", "o2sat"),
        "systolic_bp": find("systolic_bp", "systolic_blood_pressure", "systolic blood pressure", "sbp"),
        "diastolic_bp": find("diastolic_bp", "diastolic_blood_pressure", "diastolic blood pressure", "dbp"),
        "triage_level": find("triage_level", "triage", "priority", "triage level"),
    }
    return found


def _normalize_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Đổi tên cột theo chuẩn project."""
    rename_map = {v: k for k, v in mapping.items() if v is not None}
    df = df.rename(columns=rename_map)
    return df


def _synthesize_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các cột bắt buộc còn thiếu từ dữ liệu sẵn có."""
    if "gender" not in df.columns:
        print("  Tạo cột gender ngẫu nhiên từ phân bố Nam/Nữ.")
        df["gender"] = np.random.choice(["Nam", "Nữ"], size=len(df), p=[0.5, 0.5])

    if "respiratory_rate" not in df.columns:
        print("  Tạo cột respiratory_rate từ heart_rate (ước lượng RR ~ HR / 4).")
        df["respiratory_rate"] = (df["heart_rate"] / 4).clip(8, 40)

    if "diastolic_bp" not in df.columns:
        print("  Tạo cột diastolic_bp từ systolic_bp (ước lượng DBP ~ SBP * 0.6).")
        df["diastolic_bp"] = (df["systolic_bp"] * 0.6).clip(50, 120)

    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in [*FEATURES, TARGET] if col not in df.columns]
    if missing:
        available = list(df.columns)
        raise ValueError(
            f"Thiếu cột bắt buộc: {missing}.\nCác cột hiện có: {available}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 2: Làm sạch dữ liệu."""
    print("Làm sạch dữ liệu...")

    # Loại bỏ trùng lặp
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Loại bỏ trùng lặp: {before - len(df)} dòng.")

    # Chỉ giữ cột cần thiết
    needed_cols = [c for c in [*FEATURES, TARGET] if c in df.columns]
    df = df[needed_cols].copy()

    # Chuyển numeric
    numeric_cols = [c for c in FEATURES if c != "gender"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lọc giá trị bất thường theo yêu cầu
    initial_len = len(df)
    if "age" in df.columns:
        df = df[(df["age"] >= 0) & (df["age"] <= 120)]
    if "spo2" in df.columns:
        df = df[df["spo2"] <= 100]
    if "heart_rate" in df.columns:
        df = df[df["heart_rate"] <= 250]
    if "temperature" in df.columns:
        df = df[df["temperature"] <= 45]

    print(f"  Loại bỏ giá trị bất thường: {initial_len - len(df)} dòng.")

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 3: Xử lý dữ liệu thiếu."""
    print("Xử lý dữ liệu thiếu...")

    numeric_cols = [c for c in FEATURES if c != "gender"]
    missing_ratio = df[numeric_cols].isnull().mean()

    # Cột thiếu trên 20% dùng KNN
    knn_cols = missing_ratio[missing_ratio > 0.2].index.tolist()
    median_cols = missing_ratio[(missing_ratio > 0) & (missing_ratio <= 0.2)].index.tolist()

    if median_cols:
        print(f"  Điền median cho: {median_cols}")
        imputer = SimpleImputer(strategy="median")
        df[median_cols] = imputer.fit_transform(df[median_cols])

    if knn_cols:
        print(f"  Điền KNN cho: {knn_cols}")
        imputer = KNNImputer(n_neighbors=5)
        df[knn_cols] = imputer.fit_transform(df[knn_cols])

    # Điền gender bằng mode nếu có
    if "gender" in df.columns and df["gender"].isnull().any():
        df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Bước 4: Chuẩn hóa dữ liệu về [0, 1]."""
    print("Chuẩn hóa dữ liệu với MinMaxScaler...")
    numeric_cols = [c for c in FEATURES if c != "gender"]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Bước 5: Mã hóa dữ liệu dạng chữ."""
    if "gender" not in df.columns:
        return df, None

    print("Mã hóa One-Hot cho gender...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[["gender"]])
    encoded_cols = [f"gender_{cat}" for cat in encoder.categories_[0]]
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    df = pd.concat([df.drop(columns=["gender"]), encoded_df], axis=1)
    return df, encoder


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 6: Chọn đặc trưng.

    Giữ lại tất cả các đặc trưng đã chuẩn hóa/mã hóa.
    Trong tương lai có thể tích hợp SelectKBest hoặc Feature Importance.
    """
    print("Chọn đặc trưng: giữ lại tất cả các cột hợp lệ.")
    feature_cols = [c for c in df.columns if c != TARGET and c != "triage_level"]
    return df


def export_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Bước 7: Xuất dữ liệu phục vụ huấn luyện."""
    print("Chia tập train/test...")
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    ensure_dir(X_TRAIN.parent)
    FINAL_DATASET.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(FINAL_DATASET, index=False)
    X_train.to_csv(X_TRAIN, index=False)
    X_test.to_csv(X_TEST, index=False)
    y_train.to_csv(Y_TRAIN, index=False)
    y_test.to_csv(Y_TEST, index=False)

    print(f"Đã lưu:")
    print(f"  - {FINAL_DATASET}")
    print(f"  - {X_TRAIN}")
    print(f"  - {X_TEST}")
    print(f"  - {Y_TRAIN}")
    print(f"  - {Y_TEST}")

    return X_train, X_test, y_train, y_test


def preprocess(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline tiền xử lý hoàn chỉnh."""
    path = Path(path) if path else RAW_DATA_FILE
    df = load_raw_data(path)

    # Bước 1: Thu thập + mapping cột
    mapping = _infer_required_columns(df)
    df = _normalize_columns(df, mapping)
    df = _synthesize_missing_columns(df)
    _validate_required_columns(df)

    # Các bước 2-7
    df = clean_data(df)
    df = handle_missing(df)
    df, scaler = scale_features(df)
    df, encoder = encode_features(df)
    df = select_features(df)

    X_train, X_test, y_train, y_test = export_data(df)
    joblib.dump(scaler, SCALER)
    print(f"Đã lưu scaler tại: {SCALER}")
    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu Triage AI")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATA_FILE),
        help="Đường dẫn tới file CSV gốc",
    )
    args = parser.parse_args()

    preprocess(args.input)
    print("Hoàn thành tiền xử lý.")


if __name__ == "__main__":
    main()
