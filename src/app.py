"""Ứng dụng Streamlit dự đoán mức độ ưu tiên khám bệnh."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES, MODELS_DIR, TARGET


def load_model(model_name: str):
    """Load trained model by name."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        st.error(f"Chưa có model {model_name}. Vui lòng chạy training trước.")
        st.stop()
    return joblib.load(model_path)


def build_input_features(gender: str, vitals: dict) -> pd.DataFrame:
    """Build feature DataFrame from user input."""
    gender_map = {
        "Khác": "gender_Khác",
        "Nam": "gender_Nam",
        "Nữ": "gender_Nữ",
    }
    gender_onehot = {"gender_Khác": 0, "gender_Nam": 0, "gender_Nữ": 0}
    gender_onehot[gender_map[gender]] = 1

    features = [f for f in FEATURES if f != "gender"] + ["gender_Khác", "gender_Nam", "gender_Nữ"]
    return pd.DataFrame([{f: vitals.get(f, gender_onehot.get(f, 0)) for f in features}])


def predict_with_model(model, X: pd.DataFrame, model_name: str):
    """Return prediction and probability from a model."""
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return prediction, proba


def main():
    st.title("🏥 Hệ thống AI hỗ trợ xếp thứ tự khám bệnh")
    st.markdown("Nhập thông tin bệnh nhân để dự đoán mức độ ưu tiên khám bệnh.")

    # Load both models
    rf_model = load_model("random_forest")
    xgb_model = load_model("xgboost")

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
        vitals = {
            "age": age,
            "heart_rate": heart_rate,
            "respiratory_rate": respiratory_rate,
            "temperature": temperature,
            "spo2": spo2,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
        }

        X = build_input_features(gender, vitals)

        rf_pred, rf_proba = predict_with_model(rf_model, X, "random_forest")
        xgb_pred, xgb_proba = predict_with_model(xgb_model, X, "xgboost")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌲 Random Forest")
            st.success(f"Mức độ ưu tiên: **{rf_pred}**")
            proba_df = pd.DataFrame(
                {"Mức độ ưu tiên": rf_model.classes_, "Xác suất": rf_proba}
            ).sort_values("Xác suất", ascending=False)
            st.bar_chart(proba_df.set_index("Mức độ ưu tiên"))

        with col2:
            st.subheader("⚡ XGBoost")
            st.success(f"Mức độ ưu tiên: **{xgb_pred}**")
            proba_df = pd.DataFrame(
                {"Mức độ ưu tiên": xgb_model.classes_, "Xác suất": xgb_proba}
            ).sort_values("Xác suất", ascending=False)
            st.bar_chart(proba_df.set_index("Mức độ ưu tiên"))

        if rf_pred == xgb_pred:
            st.info(f"✅ Cả hai mô hình đều đồng thuận mức độ ưu tiên: **{rf_pred}**")
        else:
            st.warning(f"⚠️ RF dự đoán **{rf_pred}**, XGBoost dự đoán **{xgb_pred}**. Nên xem xét bác sĩ chuyên môn.")


if __name__ == "__main__":
    main()
