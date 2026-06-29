"""Sinh dữ liệu synthetic theo quy tắc lâm sàng."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_rules import NOISE_RATIO, TARGET_COUNTS, VITAL_RANGES


def sample_from_range(rng: np.random.Generator, value_range: tuple | list):
    """Lấy mẫu ngẫu nhiên từ khoảng hoặc danh sách các khoảng."""
    if isinstance(value_range, list):
        # Chọn một trong các khoảng rồi lấy mẫu
        chosen = rng.choice(value_range)
        return rng.uniform(chosen[0], chosen[1])
    return rng.uniform(value_range[0], value_range[1])


def add_noise(value: float, noise_ratio: float, rng: np.random.Generator) -> float:
    """Thêm nhiễu ±noise_ratio%."""
    return value * (1 + rng.uniform(-noise_ratio, noise_ratio))


def generate_patient(level: int, rng: np.random.Generator) -> dict:
    """Sinh một bệnh nhân synthetic cho triage level cụ thể."""
    ranges = VITAL_RANGES[level]

    age = int(sample_from_range(rng, ranges["age"]))
    heart_rate = add_noise(sample_from_range(rng, ranges["heart_rate"]), NOISE_RATIO, rng)
    respiratory_rate = add_noise(sample_from_range(rng, ranges["respiratory_rate"]), NOISE_RATIO, rng)
    temperature = add_noise(sample_from_range(rng, ranges["temperature"]), NOISE_RATIO, rng)
    spo2 = add_noise(sample_from_range(rng, ranges["spo2"]), NOISE_RATIO, rng)
    systolic_bp = add_noise(sample_from_range(rng, ranges["systolic_bp"]), NOISE_RATIO, rng)
    diastolic_bp = add_noise(sample_from_range(rng, ranges["diastolic_bp"]), NOISE_RATIO, rng)

    # Đảm bảo giá trị hợp lý
    age = max(0, min(120, age))
    heart_rate = max(0, min(250, heart_rate))
    respiratory_rate = max(0, min(100, respiratory_rate))
    temperature = max(30, min(45, temperature))
    spo2 = max(0, min(100, spo2))
    systolic_bp = max(0, min(300, systolic_bp))
    diastolic_bp = max(0, min(200, diastolic_bp))

    # Feature engineering
    pulse_pressure = systolic_bp - diastolic_bp
    shock_index = heart_rate / systolic_bp if systolic_bp > 0 else 0
    map_value = (systolic_bp + 2 * diastolic_bp) / 3

    return {
        "age": age,
        "gender": rng.choice(["Nam", "Nữ"]),
        "heart_rate": round(heart_rate, 1),
        "respiratory_rate": round(respiratory_rate, 1),
        "temperature": round(temperature, 2),
        "spo2": round(spo2, 1),
        "systolic_bp": round(systolic_bp, 1),
        "diastolic_bp": round(diastolic_bp, 1),
        "pulse_pressure": round(pulse_pressure, 1),
        "shock_index": round(shock_index, 3),
        "map": round(map_value, 1),
        "tachycardia": int(heart_rate > 100),
        "bradycardia": int(heart_rate < 60),
        "hypotension": int(systolic_bp < 90),
        "hypoxia": int(spo2 < 90),
        "fever": int(temperature >= 38),
        "tachypnea": int(respiratory_rate > 20),
        "triage_level": level,
    }


def generate_synthetic(target_counts: dict[int, int] | None = None, random_state: int = 42) -> pd.DataFrame:
    """Sinh toàn bộ synthetic dataset theo phân bố mục tiêu."""
    rng = np.random.default_rng(random_state)
    counts = target_counts or TARGET_COUNTS

    rows = []
    for level, n in counts.items():
        for _ in range(n):
            rows.append(generate_patient(level, rng))

    return pd.DataFrame(rows)


def merge_with_real(synthetic_df: pd.DataFrame, real_path: Path) -> pd.DataFrame:
    """Ghép synthetic với dữ liệu thực."""
    real = pd.read_csv(real_path)

    # Đảm bảo real có đủ các cột feature engineering
    if "pulse_pressure" not in real.columns:
        real["pulse_pressure"] = real["systolic_bp"] - real["diastolic_bp"]
        real["shock_index"] = real["heart_rate"] / real["systolic_bp"]
        real["map"] = (real["systolic_bp"] + 2 * real["diastolic_bp"]) / 3
        real["tachycardia"] = (real["heart_rate"] > 100).astype(int)
        real["bradycardia"] = (real["heart_rate"] < 60).astype(int)
        real["hypotension"] = (real["systolic_bp"] < 90).astype(int)
        real["hypoxia"] = (real["spo2"] < 90).astype(int)
        real["fever"] = (real["temperature"] >= 38).astype(int)
        real["tachypnea"] = (real["respiratory_rate"] > 20).astype(int)

    merged = pd.concat([real, synthetic_df], ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh dữ liệu synthetic theo quy tắc lâm sàng")
    parser.add_argument("--output", type=str, default="data/raw/augmented_ktas.csv", help="File output")
    parser.add_argument("--real", type=str, default="data/raw/ktas_cleaned.csv", help="Dữ liệu thực")
    args = parser.parse_args()

    print("Sinh dữ liệu synthetic theo quy tắc lâm sàng...")
    synthetic_df = generate_synthetic()
    print("Phân bố synthetic:")
    print(synthetic_df["triage_level"].value_counts().sort_index())

    merged = merge_with_real(synthetic_df, Path(args.real))
    merged.to_csv(args.output, index=False)
    print(f"\nĐã lưu dataset ghép tại: {args.output}")
    print(f"Tổng số mẫu: {len(merged)}")
    print("\nPhân bố cuối cùng:")
    print(merged["triage_level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
