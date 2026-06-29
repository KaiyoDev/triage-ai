"""Project configuration shared across scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"

# Required features for the model
FEATURES = [
    "age",
    "gender",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
]

TARGET = "triage_level"

# Expected triage levels (1 = highest priority, 5 = lowest priority)
TRIAGE_LEVELS = [1, 2, 3, 4, 5]

# Files
RAW_DATA_FILE = RAW_DIR / "synthetic_medical_triage.csv"
FINAL_DATASET = PROCESSED_DIR / "dataset_final.csv"
X_TRAIN = PROCESSED_DIR / "X_train.csv"
X_TEST = PROCESSED_DIR / "X_test.csv"
Y_TRAIN = PROCESSED_DIR / "y_train.csv"
Y_TEST = PROCESSED_DIR / "y_test.csv"

# Train / test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
