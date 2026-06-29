"""Ứng dụng Streamlit 5 trang cho Triage AI."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES, MODELS_DIR, REPORT_DIR

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Triage AI",
    page_icon="🏥",
    layout="wide",
)


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


# ---------------- Trang 1: Trang chủ ----------------
def home_page():
    st.title("🏥 Hệ thống AI hỗ trợ xếp thứ tự khám bệnh")
    st.markdown(
        """
        **Triage AI** là hệ thống phân loại mức độ ưu tiên khám bệnh dựa trên các chỉ số sinh tồn,
        sử dụng các thuật toán Machine Learning (Random Forest và XGBoost).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌲 Random Forest", "Accuracy 91.58%")
    col2.metric("⚡ XGBoost", "Accuracy 91.84%")
    col3.metric("📊 Tập train", "1,000,000 mẫu")
    col4.metric("🧪 Tập test", "199,998 mẫu")

    st.markdown("---")
    st.subheader("Mục tiêu")
    st.markdown(
        """
        - Giảm thời gian chờ khám bệnh.
        - Hỗ trợ nhân viên y tế đưa ra quyết định nhanh chóng.
        - Trực quan hóa và giải thích kết quả dự đoán.
        """
    )

    if st.button("🚀 Bắt đầu dự đoán", use_container_width=True):
        st.session_state.page = "Dự đoán bệnh nhân"
        st.rerun()


# ---------------- Trang 2: Dự đoán bệnh nhân ----------------
def predict_page():
    st.title("🩺 Dự đoán mức độ ưu tiên")

    rf_model = load_model("random_forest")
    xgb_model = load_model("xgboost")

    with st.form("patient_form"):
        st.subheader("Thông tin bệnh nhân")

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Tuổi", 0, 120, 35)
            gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
            heart_rate = st.number_input("Nhịp tim (bpm)", 0, 250, 80)
            respiratory_rate = st.number_input("Nhịp thở (lần/phút)", 0, 100, 18)
        with col2:
            temperature = st.number_input("Nhiệt độ (°C)", 30.0, 45.0, 37.0, 0.1)
            spo2 = st.number_input("Độ bão hòa O₂ (%)", 0, 100, 98)
            systolic_bp = st.number_input("Huyết áp tâm thu (mmHg)", 0, 300, 120)
            diastolic_bp = st.number_input("Huyết áp tâm trương (mmHg)", 0, 200, 80)

        submitted = st.form_submit_button("🔍 DỰ ĐOÁN", use_container_width=True)

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
        rf_pred = rf_model.predict(X)[0]
        xgb_pred = xgb_model.predict(X)[0]
        rf_proba = rf_model.predict_proba(X)[0].max()
        xgb_proba = xgb_model.predict_proba(X)[0].max()

        st.markdown("---")
        st.subheader("Kết quả dự đoán")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌲 Random Forest", f"Cấp {rf_pred}", f"{rf_proba*100:.0f}%")
        col2.metric("⚡ XGBoost", f"Cấp {xgb_pred}", f"{xgb_proba*100:.0f}%")

        consensus = "✔ Có" if rf_pred == xgb_pred else "✖ Không"
        col3.metric("Đồng thuận", consensus)


# ---------------- Trang 3: So sánh mô hình ----------------
def compare_page():
    st.title("📊 So sánh mô hình")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Random Forest": [0.9158, 0.9163, 0.9158, 0.9159, 0.9882],
        "XGBoost": [0.9184, 0.9189, 0.9184, 0.9186, 0.9896],
    })

    st.dataframe(metrics.set_index("Metric").style.format("{:.4f}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.plot(x="Metric", kind="bar", ax=ax, color=["#2ecc71", "#3498db"])
    ax.set_ylabel("Score")
    ax.set_title("So sánh hiệu năng Random Forest và XGBoost")
    ax.set_ylim(0.90, 0.995)
    ax.legend(loc="lower right")
    st.pyplot(fig)


# ---------------- Trang 4: Explainable AI ----------------
def explain_page():
    st.title("📈 Explainable AI")
    st.markdown("Giải thích đóng góp của các đặc trưng trong dự đoán mô hình.")

    summary_path = REPORT_DIR / "shap_summary.png"
    bar_path = REPORT_DIR / "shap_bar.png"

    if summary_path.exists():
        st.subheader("SHAP Summary Plot")
        st.image(str(summary_path), use_container_width=True)
    else:
        st.warning("Chưa có SHAP summary plot. Chạy `src/04_explainable_ai.py`.")

    if bar_path.exists():
        st.subheader("SHAP Bar Plot")
        st.image(str(bar_path), use_container_width=True)
    else:
        st.warning("Chưa có SHAP bar plot.")

    st.subheader("Feature Importance")
    rf_model = load_model("random_forest")
    importances = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importances, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    st.pyplot(fig)


# ---------------- Trang 5: Giới thiệu nhóm ----------------
def about_page():
    st.title("ℹ️ Giới thiệu")

    st.subheader("Thành viên nhóm")
    members = pd.DataFrame({
        "Họ và tên": ["Đặng Hoàng Ân", "Lê Hiền Đức", "Nguyễn Ngọc Phương", "Nguyễn Thị Thảo Trang"],
        "MSSV": ["0802....4982", "0682....2504", "0753....8872", "0403....8361"],
        "Nhiệm vụ": [
            "Kiến trúc hệ thống, Random Forest, giao diện",
            "Dataset và tiền xử lý dữ liệu",
            "XGBoost và đánh giá mô hình",
            "SHAP, kết luận và tổng hợp báo cáo",
        ],
    })
    st.dataframe(members, use_container_width=True, hide_index=True)

    st.subheader("Công nghệ sử dụng")
    st.markdown(
        """
        - Python, Pandas, NumPy
        - scikit-learn, XGBoost
        - SHAP, Streamlit
        """
    )


# ---------------- Main navigation ----------------
def main():
    pages = {
        "🏠 Trang chủ": home_page,
        "🩺 Dự đoán bệnh nhân": predict_page,
        "📊 So sánh mô hình": compare_page,
        "📈 Explainable AI": explain_page,
        "ℹ️ Giới thiệu": about_page,
    }

    # Sidebar navigation
    st.sidebar.title("Triage AI")
    selected = st.sidebar.radio("Điều hướng", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info("Đồ án môn Trí tuệ nhân tạo - Hệ thống hỗ trợ xếp thứ tự khám bệnh")

    pages[selected]()


if __name__ == "__main__":
    main()
