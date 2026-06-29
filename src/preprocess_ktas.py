"""Làm sạch và chuẩn hóa dữ liệu KTAS gốc."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def clean_ktas(path: Path) -> pd.DataFrame:
    """Đọc và làm sạch data.csv."""
    df = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False)

    # Loại bỏ các dòng có giá trị không phải số ở Age
    df = df[pd.to_numeric(df["Age"], errors="coerce").notna()]

    # Chọn các cột cần thiết
    selected = pd.DataFrame()
    selected["age"] = pd.to_numeric(df["Age"], errors="coerce")
    selected["gender"] = pd.to_numeric(df["Sex"], errors="coerce").map({1: "Nam", 2: "Nữ"})
    selected["heart_rate"] = pd.to_numeric(df["HR"], errors="coerce")
    selected["respiratory_rate"] = pd.to_numeric(df["RR"], errors="coerce")
    selected["temperature"] = pd.to_numeric(df["BT"], errors="coerce")
    selected["spo2"] = pd.to_numeric(df["Saturation"], errors="coerce")
    selected["systolic_bp"] = pd.to_numeric(df["SBP"], errors="coerce")
    selected["diastolic_bp"] = pd.to_numeric(df["DBP"], errors="coerce")
    selected["triage_level"] = pd.to_numeric(df["KTAS_expert"], errors="coerce")

    # Chỉ giữ KTAS 1-5
    selected = selected[selected["triage_level"].isin([1, 2, 3, 4, 5])]

    # Lọc giá trị bất thường
    selected = selected[(selected["age"] >= 0) & (selected["age"] <= 120)]
    selected = selected[(selected["heart_rate"] >= 0) & (selected["heart_rate"] <= 250)]
    selected = selected[(selected["respiratory_rate"] >= 0) & (selected["respiratory_rate"] <= 100)]
    selected = selected[(selected["temperature"] >= 30) & (selected["temperature"] <= 45)]
    selected = selected[(selected["spo2"] >= 0) & (selected["spo2"] <= 100)]
    selected = selected[(selected["systolic_bp"] >= 0) & (selected["systolic_bp"] <= 300)]
    selected = selected[(selected["diastolic_bp"] >= 0) & (selected["diastolic_bp"] <= 200)]

    # Thêm feature engineering
    selected["pulse_pressure"] = selected["systolic_bp"] - selected["diastolic_bp"]
    selected["shock_index"] = selected["heart_rate"] / selected["systolic_bp"]
    selected["map"] = (selected["systolic_bp"] + 2 * selected["diastolic_bp"]) / 3
    selected["tachycardia"] = (selected["heart_rate"] > 100).astype(int)
    selected["bradycardia"] = (selected["heart_rate"] < 60).astype(int)
    selected["hypotension"] = (selected["systolic_bp"] < 90).astype(int)
    selected["hypoxia"] = (selected["spo2"] < 90).astype(int)
    selected["fever"] = (selected["temperature"] >= 38).astype(int)
    selected["tachypnea"] = (selected["respiratory_rate"] > 20).astype(int)

    selected = selected.dropna()
    return selected


def main() -> None:
    raw_path = Path("data/raw/data.csv")
    df = clean_ktas(raw_path)
    print(f"Số mẫu sau khi làm sạch: {len(df)}")
    print("\nPhân bố KTAS:")
    print(df["triage_level"].value_counts().sort_index())
    print("\nTrung bình chỉ số theo level:")
    print(df.groupby("triage_level")[["heart_rate", "respiratory_rate", "temperature", "spo2", "systolic_bp", "diastolic_bp"]].mean())

    output = Path("data/raw/ktas_cleaned.csv")
    df.to_csv(output, index=False)
    print(f"\nĐã lưu: {output}")


if __name__ == "__main__":
    main()
