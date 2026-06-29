"""Ứng dụng Streamlit — Đồ án phân loại ưu tiên khám bệnh (Triage AI)."""
from __future__ import annotations
import sys
sys.path = [p for p in sys.path if "python-packages" not in p.lower()]

from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns, streamlit as st
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Triage AI — Đồ án AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"
DATA_DIR = BASE_DIR / "data" / "processed"

# Deploy fallback: nếu data/processed không có, thử đọc từ raw
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

C = {"primary":"#1E40AF","primary_light":"#3B82F6","success":"#15803D","warning":"#B45309","danger":"#B91C1C","bg":"#F8FAFC","surface":"#FFFFFF","text":"#0F172A","muted":"#475569","border":"#E2E8F0"}
TRIAGE_COLORS = {1:"#B91C1C",2:"#B45309",3:"#F59E0B",4:"#15803D",5:"#1E40AF"}
TRIAGE_LABELS = {1:"Nguy kịch",2:"Khẩn cấp",3:"Khẩn cấp chậm",4:"Không khẩn cấp",5:"Không cần khẩn cấp"}
TRIAGE_DESC = {1:"Cần cấp cứu ngay, đe dọa tính mạng",2:"Cần xử trí khẩn cấp",3:"Cần xử trí nhưng chưa nguy kịch",4:"Tình trạng nhẹ, có thể chờ",5:"Không cần can thiệp y tế khẩn cấp"}

RF_METRICS = {"accuracy":0.9133,"precision":0.9147,"recall":0.9133,"f1":0.9135,"roc_auc":0.9923}
XGB_METRICS = {"accuracy":0.9109,"precision":0.9107,"recall":0.9109,"f1":0.9107,"roc_auc":0.9926}
RF_CM = np.array([[161,0,0,0,0],[0,308,13,8,0],[0,8,513,21,1],[0,8,11,326,51],[0,0,3,21,220]])
XGB_CM = np.array([[161,0,0,0,0],[0,305,16,8,0],[0,6,523,14,0],[0,10,14,332,40],[0,0,3,38,203]])
RF_FEATURES = [("shock_index",0.2473),("heart_rate",0.1485),("respiratory_rate",0.1468),("map",0.1191),("systolic_bp",0.0982),("tachypnea",0.0570),("spo2",0.0547),("diastolic_bp",0.0350),("tachycardia",0.0301),("temperature",0.0193),("hypotension",0.0131),("hypoxia",0.0125),("pulse_pressure",0.0079),("age",0.0060),("fever",0.0019),("gender_Nam",0.0011)]
FEATURES = ["age","gender","heart_rate","respiratory_rate","temperature","spo2","systolic_bp","diastolic_bp","pulse_pressure","shock_index","map","tachycardia","bradycardia","hypotension","hypoxia","fever","tachypnea"]


def local_css():
    st.markdown(f"""
    <style>
    html,body,[class*="css"]{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;color:{C["text"]};}}
    .main{{background:{C["bg"]};}}
    .block-container{{padding-top:1rem;padding-bottom:2rem;max-width:1200px;}}
    h1,h2,h3{{color:{C["text"]};font-weight:700;}}
    h1{{font-size:1.4rem;}}
    h2{{font-size:1.15rem;margin-top:0.75rem;}}
    .card{{background:{C["surface"]};border:1px solid {C["border"]};border-radius:10px;padding:1rem;margin-bottom:0.6rem;}}
    .card-accent{{border-left:4px solid {C["primary"]};}}
    .stat-number{{font-size:1.5rem;font-weight:700;color:{C["primary"]};margin:0;}}
    .footer{{margin-top:2rem;padding:0.75rem 0;text-align:center;color:{C["muted"]};font-size:0.72rem;border-top:1px solid {C["border"]};}}
    .stButton>button{{background:{C["primary"]};color:white;border-radius:8px;font-weight:600;height:40px;border:none;}}
    .stButton>button:hover{{background:{C["primary_light"]};}}
    .badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:0.7rem;font-weight:600;color:white;}}
    </style>
    """, unsafe_allow_html=True)


MODEL_FILE_MAP = {"random_forest":"random_forest_model","xgboost":"xgboost_model"}
def load_model(m):
    p = MODELS_DIR / f"{MODEL_FILE_MAP.get(m,m)}.pkl"
    if not p.exists(): st.error("Thiếu model"); st.stop()
    return joblib.load(p)
@st.cache_data
def load_dataset():
    p = DATA_DIR / "dataset_final.csv"
    return None if not p.exists() else pd.read_csv(p)
def load_scaler():
    # Ưu tiên models/, fallback data/processed/
    for p in [MODELS_DIR / "scaler.pkl", DATA_DIR / "scaler.pkl"]:
        if p.exists():
            return joblib.load(p)
    st.error("Thiếu scaler. Chạy pipeline preprocessing trước."); st.stop()

def build_input_features(gender,vitals,scaler):
    pp = vitals["systolic_bp"] - vitals["diastolic_bp"]
    si = vitals["heart_rate"] / vitals["systolic_bp"]
    mv = (vitals["systolic_bp"] + 2 * vitals["diastolic_bp"]) / 3
    row = pd.DataFrame([{
        "age":vitals["age"],"heart_rate":vitals["heart_rate"],
        "respiratory_rate":vitals["respiratory_rate"],"temperature":vitals["temperature"],
        "spo2":vitals["spo2"],"systolic_bp":vitals["systolic_bp"],"diastolic_bp":vitals["diastolic_bp"],
        "pulse_pressure":pp,"shock_index":si,"map":mv,
        "tachycardia":int(vitals["heart_rate"]>100),"bradycardia":int(vitals["heart_rate"]<60),
        "hypotension":int(vitals["systolic_bp"]<90),"hypoxia":int(vitals["spo2"]<90),
        "fever":int(vitals["temperature"]>=38),"tachypnea":int(vitals["respiratory_rate"]>20)}])
    s = scaler.transform(row)[0]
    return pd.DataFrame([{
        "age":s[0],"heart_rate":s[1],"respiratory_rate":s[2],"temperature":s[3],"spo2":s[4],
        "systolic_bp":s[5],"diastolic_bp":s[6],"pulse_pressure":s[7],"shock_index":s[8],"map":s[9],
        "tachycardia":s[10],"bradycardia":s[11],"hypotension":s[12],"hypoxia":s[13],"fever":s[14],
        "tachypnea":s[15],
        "gender_Nam":1 if gender=="Nam" else 0,"gender_Nữ":1 if gender=="Nữ" else 0}])

def fig_reset():
    fig,ax=plt.subplots(figsize=(10,4.5))
    for s in ax.spines.values(): s.set_visible(False)
    return fig,ax


# ════════════════════════════════════════════════════════════════
#  TRANG 1 — TỔNG QUAN
# ════════════════════════════════════════════════════════════════

def home_page():
    st.markdown(f"""
    <div class="card" style="padding:1.25rem 1.5rem;">
        <h1 style="font-size:1.6rem;margin-bottom:0.2rem;">Phân loại mức độ ưu tiên khám bệnh bằng AI</h1>
        <p style="color:{C["muted"]};font-size:0.85rem;margin-bottom:0.5rem;max-width:70ch;">Đồ án môn Trí tuệ nhân tạo — Sử dụng Random Forest và XGBoost để hỗ trợ phân luồng bệnh nhân tại cấp cứu.</p>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">
            <span class="badge" style="background:{C['primary']};">Random Forest</span>
            <span class="badge" style="background:#15803D;">XGBoost</span>
            <span class="badge" style="background:#B45309;">SHAP</span>
            <span class="badge" style="background:#6366F1;">Streamlit</span>
            <span style="font-size:0.72rem;color:{C['muted']};">Giảng viên: <b>Bùi Trọng Hiếu</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 BẮT ĐẦU DỰ ĐOÁN NGAY", use_container_width=True, type="primary"):
        st.session_state["nav"] = "🥇 Dự đoán bệnh nhân"
        st.rerun()

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='card card-accent'><p class='stat-number' style='font-size:1.8rem;'>91.33%</p><p style='color:{C['muted']};margin:0;font-size:0.78rem;'>Accuracy — Random Forest</p></div>",unsafe_allow_html=True)
    k2.markdown(f"<div class='card card-accent' style='border-left-color:#15803D;'><p class='stat-number' style='font-size:1.8rem;color:#15803D;'>99.26%</p><p style='color:{C['muted']};margin:0;font-size:0.78rem;'>ROC-AUC — XGBoost</p></div>",unsafe_allow_html=True)
    k3.markdown(f"<div class='card card-accent' style='border-left-color:#B45309;'><p class='stat-number' style='font-size:1.8rem;color:#B45309;'>8,361</p><p style='color:{C['muted']};margin:0;font-size:0.78rem;'>Tổng số mẫu dữ liệu</p></div>",unsafe_allow_html=True)
    k4.markdown(f"<div class='card card-accent' style='border-left-color:#6366F1;'><p class='stat-number' style='font-size:1.8rem;color:#6366F1;'>16</p><p style='color:{C['muted']};margin:0;font-size:0.78rem;'>Đặc trưng đầu vào</p></div>",unsafe_allow_html=True)

    st.markdown("<h2>Hành trình đồ án</h2>",unsafe_allow_html=True)
    journey=[("01","Tìm hiểu & Thu thập dữ liệu","Nghiên cứu thang KTAS, dữ liệu bệnh viện Hàn Quốc"),("02","Tiền xử lý & Làm sạch","Encoding cp1252, lọc outliers → 561 mẫu"),("03","Feature Engineering","Shock index, MAP, pulse pressure, binary flags"),("04","Sinh Synthetic","Rule-based Clinical Generator → 7,800 mẫu"),("05","Huấn luyện","RF 300 cây + XGBoost, Stratified K-Fold"),("06","Đánh giá & XAI","Confusion matrix, SHAP, Ensemble"),("07","Streamlit App","7 trang, biểu đồ, phân tích chi tiết")]
    for icon,title,desc in journey:
        st.markdown(f"<div style='border-left:3px solid {C['primary']};background:{C['surface']};border-radius:6px;padding:0.4rem 0.7rem;margin-bottom:0.3rem;'><div style='display:flex;gap:0.5rem;'><span style='font-weight:700;color:{C['primary']};min-width:24px;'>{icon}</span><div><p style='font-weight:600;margin:0;font-size:0.8rem;'>{title}</p><p style='margin:0;font-size:0.72rem;color:{C['muted']};'>{desc}</p></div></div></div>",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TRANG 2 — DỰ ĐOÁN (TO + GỌN + PHÂN TÍCH)
# ════════════════════════════════════════════════════════════════

def predict_page():
    st.title("Dự đoán mức độ ưu tiên")

    rf_model = load_model("random_forest")
    xgb_model = load_model("xgboost")
    scaler = load_scaler()

    with st.form("pf"):
        cols = st.columns(8)
        age = cols[0].number_input("Tuổi",0,120,35,key="age_in")
        gender = cols[1].selectbox("GT",["Nam","Nữ"],key="gender_in")
        hr = cols[2].number_input("Nhịp tim",0,250,80,key="hr_in")
        rr = cols[3].number_input("Nhịp thở",0,100,18,key="rr_in")
        temp = cols[4].number_input("Nhiệt độ",30.0,45.0,37.0,0.1,key="temp_in")
        spo2 = cols[5].number_input("SpO₂",0,100,98,key="spo2_in")
        sbp = cols[6].number_input("HA tâm thu",0,300,120,key="sbp_in")
        dbp = cols[7].number_input("HA tâm trương",0,200,80,key="dbp_in")
        submitted = st.form_submit_button("🩺 DỰ ĐOÁN",use_container_width=True)

    if not submitted:
        st.info("Nhập chỉ số phía trên và nhấn DỰ ĐOÁN.")
        return

    pp = sbp - dbp; si = hr/sbp; mv = (sbp+2*dbp)/3
    X = build_input_features(gender,{"age":age,"heart_rate":hr,"respiratory_rate":rr,"temperature":temp,"spo2":spo2,"systolic_bp":sbp,"diastolic_bp":dbp},scaler)
    rf_p = int(rf_model.predict(X)[0])
    xgb_p = int(xgb_model.predict(X)[0]) + 1
    rf_pr = dict(zip(rf_model.classes_,rf_model.predict_proba(X)[0]))
    xgb_cl = [int(c)+1 for c in xgb_model.classes_]
    xgb_pr = dict(zip(xgb_cl,xgb_model.predict_proba(X)[0]))
    rf_max = rf_pr.get(rf_p,0); xgb_max = xgb_pr.get(xgb_p,0)
    agree = rf_p == xgb_p

    # Kết quả 2 model — TO + ĐẸP
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:white;border-radius:14px;border:2px solid {TRIAGE_COLORS[rf_p]};padding:1.25rem 1rem;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.05);margin-bottom:0.5rem;">
            <p style="font-size:0.8rem;color:{C['muted']};margin:0;">Random Forest</p>
            <div style="font-size:3rem;font-weight:800;color:{TRIAGE_COLORS[rf_p]};margin:0.1rem 0;line-height:1.1;">{rf_p}</div>
            <div style="background:{TRIAGE_COLORS[rf_p]};color:white;border-radius:20px;padding:4px 16px;display:inline-block;font-weight:600;font-size:0.85rem;margin-bottom:0.3rem;">{TRIAGE_LABELS[rf_p]}</div>
            <div style="width:100%;background:#E2E8F0;border-radius:10px;height:7px;margin:0.35rem 0;"><div style="width:{rf_max*100}%;background:{TRIAGE_COLORS[rf_p]};height:7px;border-radius:10px;"></div></div>
            <p style="margin:0;font-size:0.8rem;color:{C['muted']};">Độ tin cậy <b>{rf_max*100:.1f}%</b></p>
        </div>""",unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:white;border-radius:14px;border:2px solid {TRIAGE_COLORS[xgb_p]};padding:1.25rem 1rem;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.05);margin-bottom:0.5rem;">
            <p style="font-size:0.8rem;color:{C['muted']};margin:0;">XGBoost</p>
            <div style="font-size:3rem;font-weight:800;color:{TRIAGE_COLORS[xgb_p]};margin:0.1rem 0;line-height:1.1;">{xgb_p}</div>
            <div style="background:{TRIAGE_COLORS[xgb_p]};color:white;border-radius:20px;padding:4px 16px;display:inline-block;font-weight:600;font-size:0.85rem;margin-bottom:0.3rem;">{TRIAGE_LABELS[xgb_p]}</div>
            <div style="width:100%;background:#E2E8F0;border-radius:10px;height:7px;margin:0.35rem 0;"><div style="width:{xgb_max*100}%;background:{TRIAGE_COLORS[xgb_p]};height:7px;border-radius:10px;"></div></div>
            <p style="margin:0;font-size:0.8rem;color:{C['muted']};">Độ tin cậy <b>{xgb_max*100:.1f}%</b></p>
        </div>""",unsafe_allow_html=True)

    final = rf_p; lc = TRIAGE_COLORS[final]; ac = C["success"] if agree else C["warning"]
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;background:{C['surface']};border:1px solid {C['border']};border-radius:10px;padding:0.5rem 1rem;margin:0.25rem 0 0.75rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <span style="background:{lc};color:white;border-radius:8px;padding:3px 12px;font-weight:700;">Cấp {final}</span>
            <span style="font-weight:600;font-size:0.85rem;">{TRIAGE_LABELS[final]}</span>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;">
            <span>SI <b>{si:.2f}</b></span><span>|</span>
            <span>MAP <b>{mv:.0f}</b></span><span>|</span>
            <span style="color:{ac};">{'✅ Đồng thuận' if agree else '⚠️ Khác kết quả'}</span>
        </div>
    </div>""",unsafe_allow_html=True)

    # ===== PHÂN TÍCH =====
    st.markdown("<h2>Phân tích & Giải thích</h2>",unsafe_allow_html=True)
    ca,cb = st.columns([1,1])

    with ca:
        levels = [1,2,3,4,5]
        pdf = pd.DataFrame({"Mức":[f"Cấp {l}" for l in levels],"RF":[rf_pr.get(l,0)*100 for l in levels],"XGB":[xgb_pr.get(l,0)*100 for l in levels]})
        fig,ax = fig_reset()
        x = np.arange(5); w = 0.35
        ax.bar(x-w/2,pdf["RF"],w,label="RF",color=C["primary"],alpha=0.85)
        ax.bar(x+w/2,pdf["XGB"],w,label="XGB",color="#15803D",alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(pdf["Mức"]); ax.set_ylabel("%")
        ax.set_title("Xác suất dự đoán"); ax.legend(); ax.set_ylim(0,100)
        st.pyplot(fig)

    with cb:
        rl = ["Nhịp tim","Nhịp thở","Nhiệt độ","SpO₂","HA thu","HA trương"]
        hrn=min(hr/180,1); rrn=min(rr/40,1); tn=(temp-34)/(42-34); s2n=1-(100-spo2)/100; sbn=min(sbp/250,1); dbn=min(dbp/160,1)
        tgt={1:(160,37,40,78,72,45),2:(130,28,39,88,92,60),3:(107,23,38.3,93,105,70),4:(82,17,37.3,96,122,77),5:(70,15,36.7,99,122,77)}
        t=tgt[final]; tv=[min(t[0]/180,1),min(t[1]/40,1),(t[2]-34)/(42-34),1-(100-t[3])/100,min(t[4]/250,1),min(t[5]/160,1)]
        ang=np.linspace(0,2*np.pi,6,False).tolist()+[0]
        pv=[hrn,rrn,tn,s2n,sbn,dbn]+[hrn]; tvf=tv+[tv[0]]
        fig,ax=plt.subplots(figsize=(4.5,4.5),subplot_kw={"projection":"polar"})
        ax.plot(ang,pv,"o-",lw=2,label="BN",color=C["danger"]); ax.fill(ang,pv,alpha=0.1,color=C["danger"])
        ax.plot(ang,tvf,"s--",lw=1.5,label=f"Cấp {final}",color=C["primary"]); ax.fill(ang,tvf,alpha=0.08,color=C["primary"])
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(rl,fontsize=7); ax.set_ylim(0,1.1); ax.set_yticks([])
        ax.legend(fontsize=7,loc="upper right"); ax.set_title("BN so với Cấp "+str(final),fontsize=8)
        st.pyplot(fig)

    # Yếu tố ảnh hưởng
    st.markdown("<h2>Yếu tố ảnh hưởng</h2>",unsafe_allow_html=True)
    f=[]
    if si>0.7: f.append(("🔴 Shock Index cao",f"SI = HR/SBP = {hr}/{sbp} = {si:.3f}. Nguy cơ sốc. Cần can thiệp ngay."))
    elif si>0.5: f.append(("🟡 Shock Index ranh giới",f"SI = {si:.3f}, theo dõi sát."))
    else: f.append(("✅ Shock Index bình thường",f"SI = {si:.3f} < 0.5"))
    if mv<60: f.append(("🔴 MAP tụt nguy kịch",f"MAP = ({sbp}+2×{dbp})/3 = {mv:.0f}. Tưới máu mô kém."))
    elif mv<65: f.append(("🔴 MAP thấp",f"MAP = {mv:.0f} < 65 mmHg."))
    elif mv<75: f.append(("🟡 MAP ranh giới",f"MAP = {mv:.0f}, theo dõi."))
    else: f.append(("✅ MAP ổn định",f"MAP = {mv:.0f} mmHg."))
    if hr>120: f.append(("🔴 Nhịp tim rất nhanh",f"HR = {hr} bpm, nguy cơ loạn nhịp/sốc."))
    elif hr>100: f.append(("🟡 Nhịp tim nhanh",f"HR = {hr} > 100 (tachycardia)."))
    elif hr<60: f.append(("🟡 Nhịp tim chậm",f"HR = {hr} < 60 (bradycardia)."))
    else: f.append(("✅ Nhịp tim bình thường",f"HR = {hr} bpm."))
    if rr>28: f.append(("🔴 Nhịp thở rất nhanh",f"RR = {rr}, suy hô hấp."))
    elif rr>20: f.append(("🟡 Nhịp thở nhanh",f"RR = {rr} > 20 (tachypnea)."))
    else: f.append(("✅ Nhịp thở bình thường",f"RR = {rr} lần/ph."))
    if spo2<85: f.append(("🔴 Thiếu oxy nặng",f"SpO₂ = {spo2}%, cần hỗ trợ hô hấp ngay."))
    elif spo2<90: f.append(("🔴 Thiếu oxy",f"SpO₂ = {spo2}% < 90%."))
    elif spo2<95: f.append(("🟡 SpO₂ ranh giới",f"SpO₂ = {spo2}% < 95%."))
    else: f.append(("✅ SpO₂ bình thường",f"SpO₂ = {spo2}%."))
    if temp>=39: f.append(("🔴 Sốt cao",f"T = {temp}°C, nguy cơ nhiễm trùng."))
    elif temp>=38: f.append(("🟡 Sốt",f"T = {temp}°C (fever)."))
    elif temp<=35: f.append(("🔴 Hạ thân nhiệt",f"T = {temp}°C."))
    else: f.append(("✅ Nhiệt độ bình thường",f"T = {temp}°C."))

    cols = st.columns(3)
    for i,(t,d) in enumerate(f):
        mark = "4px solid #B91C1C" if "🔴" in t else "4px solid #B45309" if "🟡" in t else "4px solid #15803D"
        cols[i%3].markdown(f"<div style='background:{C['surface']};border-left:{mark};border-radius:6px;padding:0.4rem 0.6rem;margin-bottom:0.35rem;'><p style='font-weight:600;margin:0;font-size:0.78rem;'>{t}</p><p style='margin:0;font-size:0.7rem;color:{C['muted']};'>{d}</p></div>",unsafe_allow_html=True)

    # Tổng hợp
    abn=[]
    if hr>100:abn.append(f"nhịp nhanh ({hr})")
    if hr<60:abn.append(f"nhịp chậm ({hr})")
    if rr>20:abn.append(f"thở nhanh ({rr})")
    if temp>=38:abn.append(f"sốt ({temp}°C)")
    if spo2<90:abn.append(f"thiếu oxy (SpO₂={spo2})")
    if spo2<95:abn.append(f"SpO₂ giảm ({spo2})")
    if sbp<90:abn.append(f"hạ HA ({sbp})")
    if si>0.7:abn.append(f"shock (SI={si:.2f})")
    if mv<65:abn.append(f"MAP tụt ({mv:.0f})")
    em={1:"Bệnh nhân nguy kịch, đe dọa tính mạng. Cần hồi sức ngay.",2:"Cần xử trí khẩn cấp. Chưa nguy kịch nhưng cần nhanh.",3:"Cần xử trí nhưng chưa nguy kịch. Có thể chờ.",4:"Đa số chỉ số bình thường. Có thể chờ.",5:"Tất cả bình thường. Không cần can thiệp khẩn cấp."}
    if abn:
        st.markdown(f"<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:0.6rem 1rem;margin-top:0.5rem;'><p style='font-weight:600;margin:0 0 0.15rem;'>📋 Tổng hợp</p><p style='margin:0;font-size:0.82rem;'>Phát hiện <b>{len(abn)}</b> dấu hiệu bất thường: {'; '.join(abn[:5])}.{f' +{len(abn)-5} dấu hiệu khác.' if len(abn)>5 else ''}</p><p style='margin:0.2rem 0 0;font-size:0.82rem;font-weight:500;color:{lc};'>→ {em[final]}</p></div>",unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:0.6rem 1rem;margin-top:0.5rem;'><p style='margin:0;font-size:0.82rem;'>Không phát hiện bất thường. {em[final]}</p></div>",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TRANG 3 — QUY TRÌNH
# ════════════════════════════════════════════════════════════════

def pipeline_page():
    st.title("Quy trình thực hiện")
    st.markdown(f"<div class='card'><p style='margin:0;font-size:0.82rem;'>Chi tiết các bước từ dữ liệu thô đến mô hình hoàn chỉnh.</p></div>",unsafe_allow_html=True)

    st.markdown("<h2>1. Dữ liệu KTAS</h2>",unsafe_allow_html=True)
    c1,c2 = st.columns([1,1])
    c1.markdown(f"<div class='card'><p><b>Nguồn:</b> Korean Triage and Acuity Scale — dữ liệu bệnh viện Hàn Quốc.</p><p><b>Dữ liệu thô:</b> ~21,500 dòng, encoding cp1252, separator <code>;</code>.</p><p><b>Vấn đề:</b> KTAS 1-7, outliers, mất cân bằng nghiêm trọng (Level 1: 2 mẫu, Level 5: 18).</p></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='card card-accent'><p style='font-weight:600;margin:0 0 0.3rem;'>Kết quả làm sạch</p><table style='width:100%;font-size:0.82rem;'><tr><td>Mẫu thô ban đầu</td><td><b>~21,500</b></td></tr><tr><td>KTAS hợp lệ (1-5)</td><td><b>572</b></td></tr><tr><td>Sau outliers</td><td><b>561</b></td></tr><tr><td>Gender mapping</td><td>1→Nam, 2→Nữ</td></tr></table></div>",unsafe_allow_html=True)

    st.markdown("<h2>2. Feature Engineering</h2>",unsafe_allow_html=True)
    st.markdown(f"<div class='card'><p>Từ <b>7 đặc trưng gốc</b> (tuổi, giới tính, HR, RR, nhiệt độ, SpO₂, SBP, DBP) xây dựng thêm <b>9 đặc trưng</b> có ý nghĩa lâm sàng → <b>16 đặc trưng</b> đầu vào.</p></div>",unsafe_allow_html=True)
    fg = [("Pulse Pressure","PP = SBP − DBP","Áp lực mạch"),("Shock Index","SI = HR / SBP","Chỉ số sốc"),("MAP","(SBP+2×DBP)/3","HA động mạch TB"),("Tachycardia","HR > 100","Cờ nhịp nhanh"),("Bradycardia","HR < 60","Cờ nhịp chậm"),("Hypotension","SBP < 90","Cờ hạ huyết áp"),("Hypoxia","SpO₂ < 90%","Cờ thiếu oxy"),("Fever","Nhiệt ≥ 38°C","Cờ sốt"),("Tachypnea","RR > 20","Cờ thở nhanh")]
    cols=st.columns(3)
    for i,(n,t,d) in enumerate(fg):
        cols[i%3].markdown(f"<div class='card' style='padding:0.6rem;min-height:85px;'><p style='font-weight:700;margin:0;font-size:0.82rem;'>{n}</p><p style='font-family:monospace;margin:2px 0;font-size:0.72rem;background:#F1F5F9;padding:1px 4px;border-radius:4px;display:inline-block;'>{t}</p><p style='color:{C['muted']};margin:0;font-size:0.7rem;'>{d}</p></div>",unsafe_allow_html=True)

    st.markdown("<h2>3. Synthetic Data</h2>",unsafe_allow_html=True)
    st.markdown(f"<div class='card'><p><b>Vấn đề:</b> 561 mẫu thực, Level 1: 2 mẫu → mất cân bằng. <b>Giải pháp:</b> Rule-based Clinical Generator với VITAL_RANGES riêng cho từng mức.</p></div>",unsafe_allow_html=True)
    ld=pd.DataFrame([(1,"Nguy kịch","140-180","30-45","34-35/39-41","70-85","60-85/35-55"),(2,"Khẩn cấp","115-145","24-32","38.5-39.5","85-91","85-100/50-70"),(3,"Khẩn cấp chậm","95-120","20-26","37.8-38.8","91-95","95-115/60-80"),(4,"Không khẩn cấp","70-95","14-20","36.8-37.8","95-98","110-135/70-85"),(5,"Không cần","60-80","12-18","36.3-37.2","98-100","115-130/70-85")],columns=["Cấp","Mức","HR (bpm)","RR","Nhiệt độ (°C)","SpO₂ (%)","HA (mmHg)"])
    st.dataframe(ld,use_container_width=True,hide_index=True)

    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown(f"<div class='card card-accent'><p style='font-weight:600;margin:0 0 0.3rem;'>Tổng hợp</p><table style='width:100%;font-size:0.82rem;'><tr><td>Dữ liệu thực</td><td><b>561</b></td></tr><tr><td>Synthetic</td><td><b>7,800</b></td></tr><tr><td style='font-weight:700;'>Tổng</td><td style='font-weight:700;'>8,361</td></tr><tr><td>Tỉ lệ synth:real</td><td>~14:1</td></tr></table></div>",unsafe_allow_html=True)
    with c2:
        fig,ax=fig_reset()
        b=ax.bar([1,2,3,4,5],[800,1500,2500,1800,1200],color=["#B91C1C","#B45309","#F59E0B","#15803D","#1E40AF"],width=0.6)
        ax.set_xticks([1,2,3,4,5]); ax.set_title("Phân bố synthetic mục tiêu")
        for bar in b: ax.annotate(f"{int(bar.get_height())}",(bar.get_x()+bar.get_width()/2,bar.get_height()+30),ha="center",fontsize=8)
        st.pyplot(fig)

    st.markdown("<h2>4. Huấn luyện</h2>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    c1.markdown(f"<div class='card card-accent'><p style='font-weight:700;'>Random Forest</p><table style='width:100%;font-size:0.82rem;'><tr><td>n_estimators</td><td>300</td></tr><tr><td>max_depth</td><td>12</td></tr><tr><td>class_weight</td><td>balanced</td></tr><tr><td>CV</td><td>Stratified 5-Fold</td></tr></table></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='card card-accent' style='border-left-color:#15803D;'><p style='font-weight:700;'>XGBoost</p><table style='width:100%;font-size:0.82rem;'><tr><td>n_estimators</td><td>300</td></tr><tr><td>max_depth</td><td>6</td></tr><tr><td>learning_rate</td><td>0.1</td></tr><tr><td>Nhãn</td><td>1-5 → 0-4 → dự đoán +1</td></tr></table></div>",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TRANG 4 — KẾT QUẢ & ĐÁNH GIÁ
# ════════════════════════════════════════════════════════════════

def evaluation_page():
    st.title("Kết quả & Đánh giá")
    st.markdown(f"<div class='card'><p style='margin:0;font-size:0.82rem;'>Test set: 1,673 mẫu.</p></div>",unsafe_allow_html=True)

    m=pd.DataFrame({"Chỉ số":["Accuracy","Precision","Recall","F1","ROC-AUC"],"RF":["91.33%","91.47%","91.33%","91.35%","99.23%"],"XGB":["91.09%","91.07%","91.09%","91.07%","99.26%"]})
    st.dataframe(m,use_container_width=True,hide_index=True)

    fig,ax=fig_reset()
    x=np.arange(5);w=0.3
    ax.bar(x-w/2,[0.9133,0.9147,0.9133,0.9135,0.9923],w,label="RF",color=C["primary"],alpha=0.85)
    ax.bar(x+w/2,[0.9109,0.9107,0.9109,0.9107,0.9926],w,label="XGB",color="#15803D",alpha=0.85)
    ax.set_xticks(x);ax.set_xticklabels(["Acc","Prec","Recall","F1","AUC"]);ax.set_ylim(0.88,1.0);ax.legend();ax.set_ylabel("Score")
    st.pyplot(fig)

    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(5.5,4.5))
        sns.heatmap(RF_CM,annot=True,fmt="d",cmap="Blues",xticklabels=["C1","C2","C3","C4","C5"],yticklabels=["C1","C2","C3","C4","C5"],cbar=False)
        ax.set_xlabel("Dự đoán");ax.set_ylabel("Thực tế");ax.set_title("Confusion Matrix — RF")
        st.pyplot(fig)
    with c2:
        fig,ax=plt.subplots(figsize=(5.5,4.5))
        sns.heatmap(XGB_CM,annot=True,fmt="d",cmap="Greens",xticklabels=["C1","C2","C3","C4","C5"],yticklabels=["C1","C2","C3","C4","C5"],cbar=False)
        ax.set_xlabel("Dự đoán");ax.set_ylabel("Thực tế");ax.set_title("Confusion Matrix — XGB")
        st.pyplot(fig)

    st.markdown("<h2>Nhận xét</h2>",unsafe_allow_html=True)
    st.markdown("""<div class='card'><ul style='margin:0;font-size:0.85rem;'><li>Cả 2 mô hình đạt <b>Accuracy > 91%</b>, ROC-AUC ~<b>99%</b>.</li><li><b>Level 1</b> đạt 100% Precision/Recall — phát hiện tuyệt đối ca nguy kịch.</li><li><b>Shock Index</b> là đặc trưng quan trọng nhất (RF: 24.7%, XGB: 45%).</li><li>Level 4 và 5 độ chính xác thấp hơn (~81-87%) do ranh giới mờ.</li><li><b>Ensemble</b> đạt Accuracy <b>91.80%</b>, cao hơn từng mô hình riêng.</li></ul></div>""",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TRANG 5 — DỮ LIỆU
# ════════════════════════════════════════════════════════════════

def data_page():
    st.title("Khám phá dữ liệu")
    df=load_dataset()
    rp=RAW_DATA_DIR/"augmented_ktas.csv"
    if df is None and rp.exists(): df=pd.read_csv(rp)
    if df is None: st.warning("Thiếu dữ liệu. Chạy pipeline preprocessing trước."); return

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Tổng mẫu",f"{len(df):,}"); c2.metric("Đặc trưng số",len(df.select_dtypes(include="number").columns)); c3.metric("Mức triage",df["triage_level"].nunique()); c4.metric("Trung bình",f"{df['triage_level'].mean():.2f}")

    dist=df["triage_level"].value_counts().sort_index().reset_index(); dist.columns=["Mức","Số lượng"]
    fig,ax=fig_reset()
    b=ax.bar(dist["Mức"],dist["Số lượng"],color=[TRIAGE_COLORS[x] for x in dist["Mức"]],width=0.6)
    ax.set_xticks(dist["Mức"]);ax.set_title("Phân bố mức độ ưu tiên")
    for bar in b: ax.annotate(f"{int(bar.get_height())}",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha="center",va="bottom",fontsize=9)
    st.pyplot(fig)

    st.markdown("<h2>Phân bố dấu hiệu sinh tồn</h2>",unsafe_allow_html=True)
    vc=["heart_rate","respiratory_rate","temperature","spo2","systolic_bp","diastolic_bp"]
    vl=["Nhịp tim (bpm)","Nhịp thở (lần/ph)","Nhiệt độ (°C)","SpO₂ (%)","HA tâm thu (mmHg)","HA tâm trương (mmHg)"]
    tabs=st.tabs(vl)
    for tab,col,label in zip(tabs,vc,vl):
        with tab:
            fig,ax=plt.subplots(figsize=(10,4))
            for l in sorted(df["triage_level"].unique()):
                d=df[df["triage_level"]==l][col].dropna()
                if len(d)>0: sns.kdeplot(d,ax=ax,label=f"Cấp {l}",color=TRIAGE_COLORS[l],fill=True,alpha=0.2)
            ax.set_xlabel(label);ax.set_title(label);ax.legend()
            st.pyplot(fig)

    st.markdown("<h2>Thống kê mô tả</h2>",unsafe_allow_html=True)
    st.dataframe(df.describe().style.format("{:.2f}"),use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TRANG 6 — XAI
# ════════════════════════════════════════════════════════════════

def explain_page():
    st.title("Giải thích mô hình (XAI)")
    st.markdown(f"<div class='card'><p style='margin:0;font-size:0.82rem;'><b>SHAP</b> (SHapley Additive exPlanations) dựa trên lý thuyết trò chơi, đo đóng góp của từng đặc trưng vào kết quả dự đoán.</p></div>",unsafe_allow_html=True)

    sp=REPORT_DIR/"shap_summary.png";bp=REPORT_DIR/"shap_bar.png"
    c1,c2=st.columns(2)
    with c1:
        if sp.exists(): st.image(str(sp),use_container_width=True)
        else: st.info("Chạy src/04_explainable_ai.py để tạo SHAP plots.")
    with c2:
        if bp.exists(): st.image(str(bp),use_container_width=True)
        else: st.info("Chưa có SHAP bar plot.")

    st.markdown("<h2>Feature Importance — RF</h2>",unsafe_allow_html=True)
    rf_model=load_model("random_forest")
    imp=pd.DataFrame({"Feature":rf_model.feature_names_in_,"Importance":rf_model.feature_importances_}).sort_values("Importance")
    fig,ax=fig_reset()
    ax.barh(imp["Feature"],imp["Importance"],color=C["primary"],alpha=0.85)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.markdown("<h2>Phân tích top yếu tố</h2>",unsafe_allow_html=True)
    for n,i in [("Shock Index",0.2473),("Heart Rate",0.1485),("Respiratory Rate",0.1468),("MAP",0.1191)]:
        st.markdown(f"<div style='border-left:3px solid {C['primary']};background:{C['surface']};border-radius:6px;padding:0.4rem 0.7rem;margin-bottom:0.3rem;'><div style='display:flex;align-items:flex-start;gap:0.5rem;'><span style='background:{C['primary']};color:white;border-radius:4px;padding:0 8px;font-weight:700;font-size:0.75rem;'>{i:.0%}</span><div><p style='font-weight:600;margin:0;font-size:0.82rem;'>{n}</p></div></div></div>",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TRANG 7 — GIỚI THIỆU
# ════════════════════════════════════════════════════════════════

def about_page():
    st.title("Giới thiệu đồ án")
    st.markdown(f"""
    <div class='card' style='padding:1.25rem;'>
    <h2 style='margin-top:0;'>Phân loại mức độ ưu tiên khám bệnh bằng AI</h2>
    <p><b>Môn học:</b> Trí tuệ nhân tạo</p>
    <p><b>Giảng viên:</b> Bùi Trọng Hiếu</p>
    <p><b>Học kỳ:</b> 2025-2026</p>
    <hr style='border:none;border-top:1px solid {C['border']};margin:0.5rem 0;'>
    <p style='text-align:justify;'>Đồ án xây dựng hệ thống hỗ trợ phân loại bệnh nhân dựa trên dấu hiệu sinh tồn bằng các mô hình Random Forest và XGBoost.</p>
    </div>""",unsafe_allow_html=True)

    st.markdown("<h2>Thành viên</h2>",unsafe_allow_html=True)
    for col,(n,msv,rl,det,cl) in zip(st.columns(4),[("Đặng Hoàng Ân","080206014982","Nhóm trưởng","Xây dựng toàn bộ hệ thống",C["primary"]),("Lê Hiền Đức","068206002504","Thành viên","Hỗ trợ xử lý dữ liệu KTAS","#15803D"),("Nguyễn Ngọc Phương","075306018872","Thành viên","Hỗ trợ đánh giá XGBoost","#B45309"),("Nguyễn Thị Thảo Trang","040306018361","Thành viên","Hỗ trợ SHAP, báo cáo","#6366F1")]):
        col.markdown(f"<div class='card' style='text-align:center;border-top:3px solid {cl};'><p style='font-weight:700;margin:0.25rem;font-size:0.85rem;'>{n}</p><p style='color:{C['muted']};font-size:0.7rem;margin:2px 0;'>{msv}</p><p style='font-size:0.75rem;margin:2px 0;'><span class='badge' style='background:{cl};'>{rl}</span></p><p style='color:{C['muted']};font-size:0.72rem;margin:0.25rem 0 0;'>{det}</p></div>",unsafe_allow_html=True)

    st.markdown("<h2>Kết luận</h2>",unsafe_allow_html=True)
    st.markdown("""<div class='card'><ul style='margin:0;font-size:0.85rem;'><li>Xây dựng thành công hệ thống phân loại ưu tiên với độ chính xác <b>91.33%</b>.</li><li>Rule-based Clinical Generator giải quyết mất cân bằng dữ liệu.</li><li>Feature Engineering tận dụng tối đa thông tin từ dấu hiệu sinh tồn.</li><li>SHAP giúp tăng độ tin cậy cho người dùng y tế.</li></ul></div>""",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SIDEBAR & FOOTER
# ════════════════════════════════════════════════════════════════

def render_sidebar():
    st.markdown("---")
    st.markdown("<div style='padding:0.15rem 0;'><h3 style='font-size:0.9rem;margin:0;'>Phân loại ưu tiên khám bệnh</h3><p style='font-size:0.7rem;margin:0;'>Đồ án AI 2025-2026</p><p style='font-size:0.65rem;margin:1px 0;'>Giảng viên: Bùi Trọng Hiếu</p></div>",unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("Thành viên"):
        st.markdown("- **Đặng Hoàng Ân** — Nhóm trưởng\n- Lê Hiền Đức\n- Nguyễn Ngọc Phương\n- Nguyễn Thị Thảo Trang")
    st.markdown("---")

def render_footer():
    st.markdown(f"<div class='footer'>© 2026 Đặng Hoàng Ân — KaiyoDang · Đồ án Trí tuệ nhân tạo · Giảng viên Bùi Trọng Hiếu</div>",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    local_css()
    pages = {"🏠 Tổng quan":home_page,"🥇 Dự đoán bệnh nhân":predict_page,"📋 Quy trình thực hiện":pipeline_page,"📊 Kết quả & Đánh giá":evaluation_page,"📈 Khám phá dữ liệu":data_page,"🔬 Giải thích (XAI)":explain_page,"👥 Giới thiệu":about_page}
    with st.sidebar:
        st.title("🏥 Triage AI")
        render_sidebar()
        st.markdown("---")
        selected = st.radio("Điều hướng",list(pages.keys()))
    if "nav" in st.session_state and st.session_state["nav"] in pages:
        selected = st.session_state["nav"]; st.session_state["nav"] = None
    pages[selected]()
    render_footer()

if __name__ == "__main__":
    main()