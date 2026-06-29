"""Sinh synthetic dataset triage lớn, chất lượng cao theo quy tắc lâm sàng."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _generate_for_level(level: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n samples for a specific triage level."""
    age = rng.integers(1, 95, size=n)
    gender = rng.choice(["Nam", "Nữ"], size=n, p=[0.52, 0.48])

    if level == 1:
        # Critical: severe vital instability
        heart_rate = rng.normal(140, 18, size=n).clip(80, 220).astype(int)
        respiratory_rate = rng.normal(35, 7, size=n).clip(18, 60).astype(int)
        temperature = rng.normal(40.0, 1.0, size=n).clip(37.0, 42.0).round(1)
        spo2 = rng.normal(80, 7, size=n).clip(60, 95).round(1)
        systolic_bp = rng.normal(70, 14, size=n).clip(50, 110).astype(int)
    elif level == 2:
        # Emergent
        heart_rate = rng.normal(115, 12, size=n).clip(80, 160).astype(int)
        respiratory_rate = rng.normal(26, 4, size=n).clip(15, 45).astype(int)
        temperature = rng.normal(39.0, 0.7, size=n).clip(37.5, 41.5).round(1)
        spo2 = rng.normal(91, 4, size=n).clip(78, 99).round(1)
        systolic_bp = rng.normal(100, 11, size=n).clip(70, 140).astype(int)
    elif level == 3:
        # Urgent
        heart_rate = rng.normal(96, 10, size=n).clip(70, 130).astype(int)
        respiratory_rate = rng.normal(21, 3, size=n).clip(12, 35).astype(int)
        temperature = rng.normal(38.0, 0.5, size=n).clip(36.5, 40.0).round(1)
        spo2 = rng.normal(96, 2, size=n).clip(88, 100).round(1)
        systolic_bp = rng.normal(120, 9, size=n).clip(95, 150).astype(int)
    elif level == 4:
        # Less urgent
        heart_rate = rng.normal(80, 8, size=n).clip(60, 110).astype(int)
        respiratory_rate = rng.normal(17, 2, size=n).clip(10, 28).astype(int)
        temperature = rng.normal(37.2, 0.4, size=n).clip(36.2, 38.5).round(1)
        spo2 = rng.normal(98, 1.3, size=n).clip(92, 100).round(1)
        systolic_bp = rng.normal(125, 7, size=n).clip(100, 150).astype(int)
    else:
        # Non-urgent
        heart_rate = rng.normal(72, 7, size=n).clip(55, 100).astype(int)
        respiratory_rate = rng.normal(14, 2, size=n).clip(8, 22).astype(int)
        temperature = rng.normal(36.8, 0.3, size=n).clip(36.0, 38.0).round(1)
        spo2 = rng.normal(99, 0.8, size=n).clip(94, 100).round(1)
        systolic_bp = rng.normal(118, 6, size=n).clip(95, 145).astype(int)

    diastolic_bp = (systolic_bp * rng.uniform(0.55, 0.7, size=n)).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "spo2": spo2,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "triage_level": level,
    })


def generate_synthetic(
    n: int = 100_000,
    output_path: Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate large synthetic triage dataset."""
    rng = np.random.default_rng(random_state)

    # Realistic ED triage distribution
    distribution = {
        1: 0.05,
        2: 0.15,
        3: 0.35,
        4: 0.30,
        5: 0.15,
    }

    parts = []
    for level, ratio in distribution.items():
        count = int(n * ratio)
        parts.append(_generate_for_level(level, count, rng))

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} rows -> {output_path}")
        print(df["triage_level"].value_counts().sort_index())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic triage dataset")
    parser.add_argument("--n", type=int, default=100_000, help="Number of rows")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_large.csv")
    args = parser.parse_args()

    generate_synthetic(args.n, Path(args.output))


if __name__ == "__main__":
    main()
