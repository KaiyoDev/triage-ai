"""Định nghĩa ngưỡng lâm sàng cho việc sinh dữ liệu synthetic.

Các ngưỡng dựa trên thang phân loại triage lâm sàng thông thường:
- Level 1: Nguy kịch cần hồi sức (Resuscitation)
- Level 2: Khẩn cấp (Emergency)
- Level 3: Cần xử lý nhanh (Urgent)
- Level 4: Ít khẩn cấp hơn (Less Urgent)
- Level 5: Không khẩn cấp (Non-Urgent)
"""
from __future__ import annotations

# Ngưỡng nhịp tim (bpm)
HR_BRADYCARDIA = 60
HR_TACHYCARDIA = 100
HR_SEVERE_TACHYCARDIA = 140

# Ngưỡng nhịp thở (lần/phút)
RR_BRADYPNEA = 12
RR_TACHYPNEA = 20
RR_SEVERE_TACHYPNEA = 30

# Ngưỡng nhiệt độ (°C)
TEMP_HYPOTHERMIA = 35.0
TEMP_FEVER = 38.0
TEMP_HIGH_FEVER = 39.5

# Ngưỡng SpO₂ (%)
SPO2_NORMAL = 95
SPO2_MILD_HYPOXIA = 91
SPO2_MODERATE_HYPOXIA = 88
SPO2_SEVERE_HYPOXIA = 85

# Ngưỡng huyết áp (mmHg)
SBP_HYPOTENSION = 90
SBP_SEVERE_HYPOTENSION = 85
DBP_HYPOTENSION = 60
MAP_HYPOTENSION = 65

# Khoảng giá trị vital signs cho từng mức triage
VITAL_RANGES = {
    1: {
        "age": (18, 90),
        "heart_rate": (140, 180),
        "respiratory_rate": (30, 45),
        "temperature": [(34.0, 35.0), (39.0, 41.0)],  # hạ thân nhiệt hoặc sốt cao
        "spo2": (70, 85),
        "systolic_bp": (60, 85),
        "diastolic_bp": (35, 55),
    },
    2: {
        "age": (18, 90),
        "heart_rate": (115, 145),
        "respiratory_rate": (24, 32),
        "temperature": (38.5, 39.5),
        "spo2": (85, 91),
        "systolic_bp": (85, 100),
        "diastolic_bp": (50, 70),
    },
    3: {
        "age": (18, 90),
        "heart_rate": (95, 120),
        "respiratory_rate": (20, 26),
        "temperature": (37.8, 38.8),
        "spo2": (91, 95),
        "systolic_bp": (95, 115),
        "diastolic_bp": (60, 80),
    },
    4: {
        "age": (18, 90),
        "heart_rate": (70, 95),
        "respiratory_rate": (14, 20),
        "temperature": (36.8, 37.8),
        "spo2": (95, 98),
        "systolic_bp": (110, 135),
        "diastolic_bp": (70, 85),
    },
    5: {
        "age": (18, 90),
        "heart_rate": (60, 80),
        "respiratory_rate": (12, 18),
        "temperature": (36.3, 37.2),
        "spo2": (98, 100),
        "systolic_bp": (115, 130),
        "diastolic_bp": (70, 85),
    },
}

# Phân bố mẫu mục tiêu cho synthetic data
TARGET_COUNTS = {
    1: 800,
    2: 1500,
    3: 2500,
    4: 1800,
    5: 1200,
}

# Tỷ lệ nhiễu
NOISE_RATIO = 0.05
