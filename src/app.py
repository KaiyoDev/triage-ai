"""Ứng dụng Streamlit 5 trang cho Triage AI."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Triage AI",
    page_icon="🏥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"
DATA_DIR = BASE_DIR / "data" / "processed"

FEATURES = [
    "age", "gender", "heart_rate", "respiratory_rate",
    "temperature", "spo2", "systolic_bp", "diastolic_bp",
]

TRIAGE_COLORS = {
    1: "#d32f2f",  # red
    2: "#f57c00",  # orange
    3: "#fbc02d",  # yellow
    4: "#388e3c",  # green
    5: "#1976d2",  # blue
}


def local_css():
    st.markdown(
        """
        <style>
        .main { background-color: #f8f9fa; }
        .stButton>button {
            background-color: #1976d2;
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        }
        .level-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 1.2rem;
        }
        h1, h2, h3 { color: #1565c0; }
        .stSidebar .stRadio label { font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


MODEL_FILE_MAP = {
    "random_forest": "random_forest_model",
    "xgboost": "xgboost_model",
}


@st.cache_resource
def load_model(model_name: str):
    file_name = MODEL_FILE_MAP.get(model_name, model_name)
    model_path = MODELS_DIR / f"{file_name}.pkl"
    if not model_path.exists():
        st.error(f"Chưa có model {file_name}. Vui lòng chạy training trước.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    path = DATA_DIR / "dataset_final.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_input_features(gender: str, vitals: dict) -> pd.DataFrame:
    gender_map = {"Khác": "gender_Khác", "Nam": "gender_Nam", "Nữ": "gender_Nữ"}
    onehot = {"gender_Khác": 0, "gender_Nam": 0, "gender_Nữ": 0}
    onehot[gender_map[gender]] = 1
    features = [f for f in FEATURES if f != "gender"] + ["gender_Khác", "gender_Nam", "gender_Nữ"]
    return pd.DataFrame([{f: vitals.get(f, onehot.get(f, 0)) for f in features}])


# ---------------- Trang 1: Trang chủ ----------------
def home_page():
    st.title("🏥 Hệ thống AI hỗ trợ xếp thứ tự khám bệnh")
    st.markdown(
        "**Triage AI** phân loại mức độ ưu tiên khám bệnh dựa trên chỉ số sinh tồn, "
        "sử dụng Random Forest và XGBoost."
    )

    cols = st.columns(4)
    metrics = [
        ("🌲 Random Forest", "Accuracy 91.58%"),
        ("⚡ XGBoost", "Accuracy 91.84%"),
        ("📊 Tập train", "1,000,000 mẫu"),
        ("🧪 Tập test", "199,998 mẫu"),
    ]
    for col, (title, value) in zip(cols, metrics):
        with col:
            st.markdown(f"<div class='metric-card'><h4>{title}</h4><h2>{value}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Mục tiêu")
        st.markdown(
            """
            - Giảm thời gian chờ khám bệnh.
            - Hỗ trợ nhân viên y tế đưa ra quyết định nhanh chóng.
            - Trực quan hóa và giải thích kết quả dự đoán.
            """
        )
    with c2:
        st.image("https://img.icons8.com/color/240/hospital.png", width=150)

    if st.button("🚀 Bắt đầu dự đoán", use_container_width=True):
        st.session_state.nav = "🩺 Dự đoán bệnh nhân"
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
            "age": age, "heart_rate": heart_rate, "respiratory_rate": respiratory_rate,
            "temperature": temperature, "spo2": spo2,
            "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        }
        X = build_input_features(gender, vitals)

        rf_pred = int(rf_model.predict(X)[0])
        xgb_pred = int(xgb_model.predict(X)[0])
        rf_proba = rf_model.predict_proba(X)[0].max()
        xgb_proba = xgb_model.predict_proba(X)[0].max()

        st.markdown("---")
        st.subheader("Kết quả dự đoán")

        cols = st.columns(3)
        rf_color = TRIAGE_COLORS.get(rf_pred, "#1976d2")
        xgb_color = TRIAGE_COLORS.get(xgb_pred, "#1976d2")
        agree = rf_pred == xgb_pred

        with cols[0]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 🌲 Random Forest")
            st.markdown(f"<span class='level-badge' style='background:{rf_color}'>Cấp {rf_pred}</span>", unsafe_allow_html=True)
            st.markdown(f"**Xác suất: {rf_proba*100:.0f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        with cols[1]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### ⚡ XGBoost")
            st.markdown(f"<span class='level-badge' style='background:{xgb_color}'>Cấp {xgb_pred}</span>", unsafe_allow_html=True)
            st.markdown(f"**Xác suất: {xgb_proba*100:.0f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        with cols[2]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### Đồng thuận")
            if agree:
                st.markdown("<span class='level-badge' style='background:#388e3c'>✔ Có</span>", unsafe_allow_html=True)
                st.markdown("Hai mô hình cho cùng kết quả")
            else:
                st.markdown("<span class='level-badge' style='background:#d32f2f'>✖ Không</span>", unsafe_allow_html=True)
                st.markdown("Nên tham khảo ý kiến bác sĩ")
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Trang 3: So sánh mô hình ----------------
def compare_page():
    st.title("📊 So sánh mô hình")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Random Forest": [0.9158, 0.9163, 0.9158, 0.9159, 0.9882],
        "XGBoost": [0.9184, 0.9189, 0.9184, 0.9186, 0.9896],
    }).set_index("Metric")

    st.dataframe(metrics.style.format("{:.4f}").highlight_max(axis=1, color="#90caf9"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db"])
    ax.set_ylabel("Score")
    ax.set_title("So sánh hiệu năng Random Forest và XGBoost")
    ax.set_ylim(0.90, 0.995)
    ax.legend(loc="lower right")
    st.pyplot(fig)


# ---------------- Trang 4: Thống kê dữ liệu ----------------
def data_page():
    st.title("📈 Thống kê dữ liệu")

    df = load_dataset()
    if df is None:
        st.warning("Chưa có dữ liệu. Chạy `src/01_preprocessing.py` trước.")
        return

    st.subheader("Tổng quan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số mẫu", f"{len(df):,}")
    c2.metric("Số đặc trưng", len(df.columns) - 1)
    c3.metric("Số mức triage", df["triage_level"].nunique())

    st.subheader("Phân bố mức độ ưu tiên")
    dist = df["triage_level"].value_counts().sort_index().reset_index()
    dist.columns = ["Mức độ ưu tiên", "Số lượng"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dist["Mức độ ưu tiên"], dist["Số lượng"], color=[TRIAGE_COLORS[x] for x in dist["Mức độ ưu tiên"]])
    ax.set_xlabel("Mức độ ưu tiên")
    ax.set_ylabel("Số lượng")
    ax.set_title("Phân bố các mức độ ưu tiên trong dataset")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width()/2, height), ha="center", va="bottom")
    st.pyplot(fig)

    st.subheader("Mô tả thống kê")
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

    st.subheader("Tương quan giữa các đặc trưng")
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ---------------- Trang 5: Explainable AI ----------------
def explain_page():
    st.title("🔍 Explainable AI")
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


# ---------------- Trang 6: Giới thiệu nhóm ----------------
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


# ---------------- Main ----------------
def main():
    local_css()

    pages = {
        "🏠 Trang chủ": home_page,
        "🩺 Dự đoán bệnh nhân": predict_page,
        "📊 So sánh mô hình": compare_page,
        "📈 Thống kê dữ liệu": data_page,
        "🔍 Explainable AI": explain_page,
        "ℹ️ Giới thiệu": about_page,
    }

    with st.sidebar:
        st.title("Triage AI")
        st.markdown("---")
        selected = st.radio("Điều hướng", list(pages.keys()))
        st.markdown("---")
        st.info("Đồ án môn Trí tuệ nhân tạo - Hỗ trợ xếp thứ tự khám bệnh")

    pages[selected]()


if __name__ == "__main__":
    main()
