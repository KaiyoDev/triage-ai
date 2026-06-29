"""Ứng dụng Streamlit 6 trang cho Triage AI."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Triage AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"
DATA_DIR = BASE_DIR / "data" / "processed"

FEATURES = [
    "age", "gender", "heart_rate", "respiratory_rate", "temperature", "spo2",
    "systolic_bp", "diastolic_bp", "pulse_pressure", "shock_index", "map",
    "tachycardia", "bradycardia", "hypotension", "hypoxia", "fever", "tachypnea",
]

# Medical palette
COLORS = {
    "primary": "#2563EB",
    "primary_dark": "#1E40AF",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "text": "#1E293B",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "white": "#FFFFFF",
}

TRIAGE_COLORS = {
    1: "#EF4444",
    2: "#F59E0B",
    3: "#FBBF24",
    4: "#22C55E",
    5: "#2563EB",
}

TRIAGE_LABELS = {
    1: "Cấp 1 - Nguy kịch",
    2: "Cấp 2 - Khẩn cấp",
    3: "Cấp 3 - Khẩn cấp chậm",
    4: "Cấp 4 - Không khẩn cấp",
    5: "Cấp 5 - Không cần khẩn cấp",
}


def local_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .main {{
            background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
            color: {COLORS["text"]};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3 {{
            color: {COLORS["text"]};
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .hero {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
            color: white;
            padding: 2.5rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
        }}
        .hero h1 {{
            color: white;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }}
        .hero p {{
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            margin: 0;
        }}
        .kpi-card {{
            background: {COLORS["surface"]};
            border-radius: 16px;
            padding: 1.5rem;
            text-align: left;
            box-shadow: 0 4px 12px rgba(30, 41, 59, 0.06);
            border-left: 5px solid {COLORS["primary"]};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(30, 41, 59, 0.1);
        }}
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 800;
            color: {COLORS["text"]};
            margin: 0;
        }}
        .kpi-label {{
            font-size: 0.9rem;
            color: {COLORS["muted"]};
            margin-top: 0.25rem;
        }}
        .prediction-card {{
            background: {COLORS["surface"]};
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 8px 24px rgba(30, 41, 59, 0.08);
            border-top: 5px solid {COLORS["primary"]};
        }}
        .level-badge {{
            display: inline-block;
            padding: 12px 28px;
            border-radius: 30px;
            color: white;
            font-weight: 800;
            font-size: 1.4rem;
            margin: 12px 0;
        }}
        .footer {{
            margin-top: 3rem;
            padding: 1.5rem 0;
            text-align: center;
            color: {COLORS["muted"]};
            font-size: 0.85rem;
            border-top: 1px solid {COLORS["border"]};
        }}
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
            color: white;
            border-radius: 12px;
            font-weight: 700;
            height: 52px;
            border: none;
            font-size: 1rem;
        }}
        .stButton>button:hover {{
            opacity: 0.9;
        }}
        .sidebar-logo {{
            text-align: center;
            padding: 1rem 0;
        }}
        .sidebar-logo h3 {{
            color: {COLORS["primary"]};
            margin: 0;
            font-weight: 800;
        }}
        .sidebar-logo p {{
            color: {COLORS["muted"]};
            font-size: 0.85rem;
            margin: 0;
        }}
        .stForm {{
            background: {COLORS["surface"]};
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(30, 41, 59, 0.06);
        }}
        .info-card {{
            background: {COLORS["surface"]};
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(30, 41, 59, 0.06);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


MODEL_FILE_MAP = {
    "random_forest": "random_forest_model",
    "xgboost": "xgboost_model",
}


def load_model(model_name: str):
    file_name = MODEL_FILE_MAP.get(model_name, model_name)
    model_path = MODELS_DIR / f"{file_name}.pkl"
    if not model_path.exists():
        st.error(f"Chưa có model {file_name}. Vui lòng chạy training trước.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    path = DATA_DIR / "dataset_final.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_scaler():
    scaler_path = DATA_DIR / "scaler.pkl"
    if not scaler_path.exists():
        st.error("Chưa có scaler. Vui lòng chạy preprocessing trước.")
        st.stop()
    return joblib.load(scaler_path)


def build_input_features(gender: str, vitals: dict, scaler) -> pd.DataFrame:
    pulse_pressure = vitals["systolic_bp"] - vitals["diastolic_bp"]
    shock_index = vitals["heart_rate"] / vitals["systolic_bp"]
    map_value = (vitals["systolic_bp"] + 2 * vitals["diastolic_bp"]) / 3
    tachycardia = int(vitals["heart_rate"] > 100)
    bradycardia = int(vitals["heart_rate"] < 60)
    hypotension = int(vitals["systolic_bp"] < 90)
    hypoxia = int(vitals["spo2"] < 90)
    fever = int(vitals["temperature"] >= 38)
    tachypnea = int(vitals["respiratory_rate"] > 20)

    row_df = pd.DataFrame([{
        "age": vitals["age"], "heart_rate": vitals["heart_rate"],
        "respiratory_rate": vitals["respiratory_rate"], "temperature": vitals["temperature"],
        "spo2": vitals["spo2"], "systolic_bp": vitals["systolic_bp"],
        "diastolic_bp": vitals["diastolic_bp"], "pulse_pressure": pulse_pressure,
        "shock_index": shock_index, "map": map_value, "tachycardia": tachycardia,
        "bradycardia": bradycardia, "hypotension": hypotension, "hypoxia": hypoxia,
        "fever": fever, "tachypnea": tachypnea,
    }])
    scaled = scaler.transform(row_df)[0]

    return pd.DataFrame([{
        "age": scaled[0], "heart_rate": scaled[1], "respiratory_rate": scaled[2],
        "temperature": scaled[3], "spo2": scaled[4], "systolic_bp": scaled[5],
        "diastolic_bp": scaled[6], "pulse_pressure": scaled[7], "shock_index": scaled[8],
        "map": scaled[9], "tachycardia": scaled[10], "bradycardia": scaled[11],
        "hypotension": scaled[12], "hypoxia": scaled[13], "fever": scaled[14],
        "tachypnea": scaled[15], "gender_Nam": 1 if gender == "Nam" else 0,
        "gender_Nữ": 1 if gender == "Nữ" else 0,
    }])


def kpi_card(label: str, value: str, accent: str = "primary") -> str:
    accent_color = COLORS.get(accent, COLORS["primary"])
    return f"""
    <div class="kpi-card" style="border-left-color: {accent_color};">
        <p class="kpi-value" style="color: {accent_color};">{value}</p>
        <p class="kpi-label">{label}</p>
    </div>
    """


def home_page():
    st.markdown(
        """
        <div class="hero">
            <h1>🏥 TRIAGE AI</h1>
            <p>Hệ thống AI hỗ trợ phân loại mức độ ưu tiên khám bệnh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_data = [
        ("Accuracy", "91.09%", "success"),
        ("Precision", "91.07%", "success"),
        ("Recall", "91.09%", "success"),
        ("F1-Score", "91.07%", "success"),
        ("ROC-AUC", "99.26%", "primary"),
        ("Dataset", "8,361 mẫu", "primary"),
    ]

    for i in range(0, len(kpi_data), 3):
        cols = st.columns(3)
        for col, (label, value, accent) in zip(cols, kpi_data[i : i + 3]):
            col.markdown(kpi_card(label, value, accent), unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Mục tiêu ứng dụng")
        st.markdown(
            """
            - Giảm thời gian chờ khám bệnh.
            - Hỗ trợ nhân viên y tế đưa ra quyết định nhanh chóng.
            - Trực quan hóa và giải thích kết quả dự đoán.
            """
        )
    with c2:
        st.image("https://img.icons8.com/color/240/hospital.png", width=160)

    st.markdown("---")
    if st.button("🚀 Bắt đầu dự đoán", use_container_width=True):
        st.session_state["nav"] = "🩺 Dự đoán bệnh nhân"
        st.rerun()


def predict_page():
    st.title("🩺 Dự đoán mức độ ưu tiên")

    rf_model = load_model("random_forest")
    xgb_model = load_model("xgboost")
    scaler = load_scaler()

    with st.form("patient_form"):
        st.subheader("Thông tin bệnh nhân")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Tuổi", 0, 120, 35)
            gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
            heart_rate = st.number_input("Nhịp tim (bpm)", 0, 250, 80)
            respiratory_rate = st.number_input("Nhịp thở (lần/phút)", 0, 100, 18)
        with col2:
            temperature = st.number_input("Nhiệt độ (°C)", 30.0, 45.0, 37.0, 0.1)
            spo2 = st.number_input("Độ bảo hòa O₂ (%)", 0, 100, 98)
            systolic_bp = st.number_input("Huyết áp tâm thu (mmHg)", 0, 300, 120)
            diastolic_bp = st.number_input("Huyết áp tâm trương (mmHg)", 0, 200, 80)

        submitted = st.form_submit_button("🔍 DỰ ĐOÁN", use_container_width=True)

    if not submitted:
        st.info("Nhập thông tin bệnh nhân và nhấn **DỰ ĐOÁN** để xem kết quả.")
        return

    vitals = {
        "age": age, "heart_rate": heart_rate, "respiratory_rate": respiratory_rate,
        "temperature": temperature, "spo2": spo2, "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
    }
    X = build_input_features(gender, vitals, scaler)

    rf_pred = int(rf_model.predict(X)[0])
    xgb_raw_pred = int(xgb_model.predict(X)[0])
    xgb_pred = xgb_raw_pred + 1

    rf_proba = dict(zip(rf_model.classes_, rf_model.predict_proba(X)[0]))
    xgb_classes = [int(c) + 1 for c in xgb_model.classes_]
    xgb_proba = dict(zip(xgb_classes, xgb_model.predict_proba(X)[0]))

    rf_max = rf_proba.get(rf_pred, 0)
    xgb_max = xgb_proba.get(xgb_pred, 0)
    agree = rf_pred == xgb_pred

    st.markdown("---")
    st.subheader("Kết quả dự đoán")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-top-color:{TRIAGE_COLORS.get(rf_pred, COLORS['primary'])};">
                <h4>🌲 Random Forest</h4>
                <div class="level-badge" style="background:{TRIAGE_COLORS.get(rf_pred, COLORS['primary'])};">
                    Cấp {rf_pred}
                </div>
                <p><strong>{TRIAGE_LABELS.get(rf_pred, '')}</strong></p>
                <p>Xác suất: <b>{rf_max*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(rf_max), text="Độ tin cậy RF")
    with c2:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-top-color:{TRIAGE_COLORS.get(xgb_pred, COLORS['primary'])};">
                <h4>⚡ XGBoost</h4>
                <div class="level-badge" style="background:{TRIAGE_COLORS.get(xgb_pred, COLORS['primary'])};">
                    Cấp {xgb_pred}
                </div>
                <p><strong>{TRIAGE_LABELS.get(xgb_pred, '')}</strong></p>
                <p>Xác suất: <b>{xgb_max*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(xgb_max), text="Độ tin cậy XGB")
    with c3:
        consensus_color = COLORS["success"] if agree else COLORS["danger"]
        consensus_text = "Đồng thuận" if agree else "Không đồng thuận"
        st.markdown(
            f"""
            <div class="prediction-card" style="border-top-color:{consensus_color};">
                <h4>🤝 {consensus_text}</h4>
                <div class="level-badge" style="background:{consensus_color};">
                    {'Có' if agree else 'Không'}
                </div>
                <p>{'Hai mô hình cho cùng kết quả.' if agree else 'Hai mô hình khác nhau, nên tham khảo ý kiến bác sĩ.'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Phân phối xác suất")
    levels = [1, 2, 3, 4, 5]
    proba_df = pd.DataFrame({
        "Mức độ ưu tiên": [f"Cấp {l}" for l in levels],
        "Random Forest": [rf_proba.get(l, 0) * 100 for l in levels],
        "XGBoost": [xgb_proba.get(l, 0) * 100 for l in levels],
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(levels))
    width = 0.35
    ax.bar(x - width / 2, proba_df["Random Forest"], width, label="Random Forest", color=COLORS["primary"], alpha=0.8)
    ax.bar(x + width / 2, proba_df["XGBoost"], width, label="XGBoost", color=COLORS["success"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(proba_df["Mức độ ưu tiên"])
    ax.set_ylabel("Xác suất (%)")
    ax.set_title("Xác suất dự đoán theo từng mức độ ưu tiên")
    ax.legend()
    ax.set_ylim(0, 100)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Yếu tố ảnh hưởng đến dự đoán")
    importance_df = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.barh(importance_df["Feature"], importance_df["Importance"], color=COLORS["primary"], alpha=0.85)
    ax2.set_xlabel("Tầm quan trọng")
    ax2.set_title("Feature Importance - Random Forest")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    st.pyplot(fig2)


def compare_page():
    st.title("📊 So sánh mô hình")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Random Forest": [0.9133, 0.9147, 0.9133, 0.9135, 0.9923],
        "XGBoost": [0.9109, 0.9107, 0.9109, 0.9107, 0.9926],
    }).set_index("Metric")

    st.dataframe(metrics.style.format("{:.4f}").highlight_max(axis=1, color="#90caf9"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics.plot(kind="bar", ax=ax, color=[COLORS["success"], COLORS["primary"]])
    ax.set_ylabel("Score")
    ax.set_title("So sánh hiệu năng Random Forest và XGBoost")
    ax.set_ylim(0.88, 1.0)
    ax.legend(loc="lower right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)


def data_page():
    st.title("📈 Thống kê dữ liệu")

    raw_path = BASE_DIR / "data" / "raw" / "augmented_ktas.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        df = load_dataset()
        if df is None:
            st.warning("Chưa có dữ liệu. Chạy preprocessing trước.")
            return

    st.subheader("Tổng quan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số mẫu", f"{len(df):,}")
    c2.metric("Số đặc trưng", len(df.columns) - 1)
    c3.metric("Số mức triage", df["triage_level"].nunique())
    c4.metric("Mức triage trung bình", f"{df['triage_level'].mean():.2f}")

    st.subheader("Phân bố mức độ ưu tiên")
    dist = df["triage_level"].value_counts().sort_index().reset_index()
    dist.columns = ["Mức độ ưu tiên", "Số lượng"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        dist["Mức độ ưu tiên"],
        dist["Số lượng"],
        color=[TRIAGE_COLORS[x] for x in dist["Mức độ ưu tiên"]],
    )
    ax.set_xlabel("Mức độ ưu tiên")
    ax.set_ylabel("Số lượng")
    ax.set_title("Phân bố các mức độ ưu tiên trong dataset")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)

    st.subheader("Mô tả thống kê")
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

    st.subheader("Tương quan giữa các đặc trưng")
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def explain_page():
    st.title("🔍 Explainable AI")
    st.markdown("Giải thích đóng góp của các đặc trưng trong dự đoán mô hình.")

    summary_path = REPORT_DIR / "shap_summary.png"
    bar_path = REPORT_DIR / "shap_bar.png"

    if summary_path.exists():
        st.subheader("SHAP Summary Plot")
        st.image(str(summary_path), width="stretch")
    else:
        st.warning("Chưa có SHAP summary plot. Chạy `src/04_explainable_ai.py`.")

    if bar_path.exists():
        st.subheader("SHAP Bar Plot")
        st.image(str(bar_path), width="stretch")
    else:
        st.warning("Chưa có SHAP bar plot.")

    st.subheader("Feature Importance")
    rf_model = load_model("random_forest")
    importances = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importances["Feature"], importances["Importance"], color=COLORS["primary"], alpha=0.85)
    ax.set_title("Feature Importance - Random Forest")
    ax.set_xlabel("Tầm quan trọng")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)


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


def render_sidebar():
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-logo">
            <h3>🏥 Triage AI</h3>
            <p>v1.0 · Đồ án AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    with st.expander("👥 Thông tin nhóm"):
        st.markdown(
            """
            - Đặng Hoàng Ân
            - Lê Hiền Đức
            - Nguyễn Ngọc Phương
            - Nguyễn Thị Thảo Trang
            """
        )
    st.markdown(
        "<div class='sidebar-footer'>"
        "Hệ thống hỗ trợ xếp thứ tự khám bệnh<br>"
        "© 2026 Triage AI"
        "</div>",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        "<div class='footer'>"
        "© 2026 Triage AI · Đồ án Trí tuệ nhân tạo · Phiên bản v1.0"
        "</div>",
        unsafe_allow_html=True,
    )


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
        render_sidebar()
        st.markdown("---")
        selected = st.radio("Điều hướng", list(pages.keys()))

    if "nav" in st.session_state and st.session_state["nav"] in pages:
        selected = st.session_state["nav"]
        st.session_state["nav"] = None

    pages[selected]()
    render_footer()


if __name__ == "__main__":
    main()
