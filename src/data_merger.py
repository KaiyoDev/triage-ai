"""Gộp nhiều nguồn dữ liệu triage thành một schema chuẩn."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES, RAW_DATA_FILE, TARGET


def _load_synthetic(path: Path) -> pd.DataFrame:
    """Load synthetic_medical_triage.csv và map về schema chuẩn."""
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "age": "age",
            "heart_rate": "heart_rate",
            "systolic_blood_pressure": "systolic_bp",
            "oxygen_saturation": "spo2",
            "body_temperature": "temperature",
            "triage_level": "triage_level",
        }
    )
    df["gender"] = np.random.choice(["Nam", "Nữ"], size=len(df))
    df["respiratory_rate"] = (df["heart_rate"] / 4).clip(8, 40)
    df["diastolic_bp"] = (df["systolic_bp"] * 0.6).clip(50, 120)
    return df[[*FEATURES, TARGET]]


def _load_patient_priority(path: Path) -> pd.DataFrame:
    """Load patient_priority.csv, map target từ màu sang level 1-5."""
    df = pd.read_csv(path)
    color_to_level = {
        "red": 1,
        "orange": 2,
        "yellow": 3,
        "green": 4,
        "blue": 5,
    }
    df[TARGET] = df["triage"].str.lower().map(color_to_level)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    df["age"] = df["age"].fillna(df["age"].median())
    df["gender"] = df["gender"].map({0.0: "Nam", 1.0: "Nữ"}).fillna("Nữ")
    df["heart_rate"] = df["max heart rate"]
    df["respiratory_rate"] = 18  # default
    df["temperature"] = 37.0  # default
    df["spo2"] = 98.0  # default
    df["systolic_bp"] = df["blood pressure"]
    df["diastolic_bp"] = (df["systolic_bp"] * 0.6).clip(50, 120)

    return df[[*FEATURES, TARGET]]


def merge_datasets(output_path: Path | None = None) -> pd.DataFrame:
    """Gộp tất cả dataset raw khả dụng."""
    parts: list[pd.DataFrame] = []

    synthetic_path = RAW_DATA_FILE.parent / "synthetic_medical_triage.csv"
    if synthetic_path.exists():
        print(f"Loading {synthetic_path.name}...")
        parts.append(_load_synthetic(synthetic_path))

    patient_priority_path = RAW_DATA_FILE.parent / "patient_priority.csv"
    if patient_priority_path.exists():
        print(f"Loading {patient_priority_path.name}...")
        parts.append(_load_patient_priority(patient_priority_path))

    if not parts:
        raise FileNotFoundError("Không tìm thấy dataset nào trong data/raw/")

    merged = pd.concat(parts, ignore_index=True)
    merged = merged[[*FEATURES, TARGET]]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"Saved merged dataset: {output_path} ({len(merged)} rows)")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge raw triage datasets")
    parser.add_argument("--output", type=str, default="data/raw/merged_triage.csv")
    args = parser.parse_args()

    merge_datasets(Path(args.output))


if __name__ == "__main__":
    main()
