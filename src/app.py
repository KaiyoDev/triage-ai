"""Ứng dụng Streamlit dự đoán mức độ ưu tiên khám bệnh."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES, MODELS_DIR, TARGET


def load_model():
    """Load trained Random Forest model."""
    model_path = MODELS_DIR / "random_forest_model.pkl"
    if not model_path.exists():
        st.error("Chưa có model. Vui lòng chạy 02_random_forest.py trước.")
        st.stop()
    return joblib.load(model_path)


def main():
    st.title("🏥 Hệ thống AI hỗ trợ xếp thứ tự khám bệnh")
    st.markdown("Nhập thông tin bệnh nhân để dự đoán mức độ ưu tiên khám bệnh.")

    model = load_model()

    with st.form("patient_form"):
        st.subheader("Thông tin bệnh nhân")

        age = st.slider("Tuổi", 0, 120, 35)
        gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
        heart_rate = st.number_input("Nhịp tim (bpm)", min_value=0, max_value=250, value=80)
        respiratory_rate = st.number_input("Nhịp thở (lần/phút)", min_value=0, max_value=100, value=18)
        temperature = st.number_input("Nhiệt độ (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        spo2 = st.number_input("Độ bão hòa O2 (%)", min_value=0, max_value=100, value=98)
        systolic_bp = st.number_input("Huyết áp tâm thu (mmHg)", min_value=0, max_value=300, value=120)
        diastolic_bp = st.number_input("Huyết áp tâm trương (mmHg)", min_value=0, max_value=200, value=80)

        submitted = st.form_submit_button("Dự đoán")

    if submitted:
        input_dict = {
            "age": age,
            "heart_rate": heart_rate,
            "respiratory_rate": respiratory_rate,
            "temperature": temperature,
            "spo2": spo2,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
        }

        # One-hot gender theo format training
        input_dict["gender_Khác"] = 1 if gender == "Khác" else 0
        input_dict["gender_Nam"] = 1 if gender == "Nam" else 0
        input_dict["gender_Nữ"] = 1 if gender == "Nữ" else 0

        # Xây dựng DataFrame theo đúng feature model
        features = [f for f in FEATURES if f != "gender"] + [
            "gender_Khác",
            "gender_Nam",
            "gender_Nữ",
        ]
        X = pd.DataFrame([{f: input_dict.get(f, 0) for f in features}])

        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        st.success(f"Mức độ ưu tiên dự đoán: **{prediction}**")
        st.write("Xác suất theo từng mức:")
        proba_df = pd.DataFrame(
            {"Mức độ ưu tiên": model.classes_, "Xác suất": proba}
        ).sort_values("Xác suất", ascending=False)
        st.bar_chart(proba_df.set_index("Mức độ ưu tiên"))


if __name__ == "__main__":
    main()
