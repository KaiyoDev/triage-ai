# Báo cáo đồ án: Phân loại mức độ ưu tiên khám bệnh bằng AI (Triage AI)

**Môn học:** Trí tuệ nhân tạo  
**Giảng viên:** Bùi Trọng Hiếu  
**Học kỳ:** 2025–2026  
**Nhóm trưởng:** Đặng Hoàng Ân — MSSV 080206014982

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Dữ liệu và thu thập](#2-dữ-liệu-và-thu-thập)
3. [Tiền xử lý dữ liệu (7 bước)](#3-tiền-xử-lý-dữ-liệu-7-bước)
4. [Feature Engineering](#4-feature-engineering)
5. [Synthetic Data Generation](#5-synthetic-data-generation)
6. [Huấn luyện mô hình](#6-huấn-luyện-mô-hình)
7. [Đánh giá và so sánh](#7-đánh-giá-và-so-sánh)
8. [Ensemble Voting Classifier](#8-ensemble-voting-classifier)
9. [Giải thích mô hình với SHAP](#9-giải-thích-mô-hình-với-shap)
10. [Ứng dụng Streamlit](#10-ứng-dụng-streamlit)
11. [Kết luận và hướng phát triển](#11-kết-luận-và-hướng-phát-triển)

---

## 1. Tổng quan dự án

### 1.1 Giới thiệu

Phân loại mức độ ưu tiên khám bệnh (triage) là quy trình quan trọng tại khoa cấp cứu, giúp xác định bệnh nhân nào cần can thiệp y tế ngay lập tức dựa trên các dấu hiệu sinh tồn. Đồ án này xây dựng hệ thống AI hỗ trợ phân loại triage sử dụng hai mô hình Machine Learning: **Random Forest** và **XGBoost**.

### 1.2 Mục tiêu

- Tự động phân loại bệnh nhân thành 5 mức ưu tiên (từ Nguy kịch đến Không cần khẩn cấp)
- Đạt độ chính xác >90% trên tập kiểm tra
- Giải thích được kết quả dự đoán thông qua SHAP
- Cung cấp giao diện trực quan cho nhân viên y tế

### 1.3 Hệ thống phân loại KTAS

| Mức | Mô tả | Ý nghĩa lâm sàng |
|-----|-------|-------------------|
| 1 | Nguy kịch (Resuscitation) | Đe dọa tính mạng, cần cấp cứu ngay |
| 2 | Khẩn cấp (Emergency) | Cần xử trí khẩn cấp |
| 3 | Khẩn cấp chậm (Urgent) | Cần xử trí nhưng chưa nguy kịch |
| 4 | Không khẩn cấp (Less Urgent) | Tình trạng nhẹ, có thể chờ |
| 5 | Không cần (Non-Urgent) | Không cần can thiệp y tế khẩn cấp |

---

## 2. Dữ liệu và thu thập

### 2.1 Nguồn dữ liệu

Dữ liệu được lấy từ **Korean Triage and Acuity Scale (KTAS)** — thang phân loại triage chuẩn hóa tại Hàn Quốc, dựa trên CTAS (Canadian Triage and Acuity Scale).

- **File gốc:** `data.csv` (~21,500 dòng)
- **Encoding:** cp1252 (Windows-1252)
- **Separator:** `;` (dấu chấm phẩy)
- **Kích thước:** ~128 KB

### 2.2 Cấu trúc dữ liệu gốc

| Cột gốc | Ý nghĩa | Kiểu |
|----------|---------|------|
| Age | Tuổi | int |
| Sex | Giới tính (1=Nam, 2=Nữ) | int |
| HR | Nhịp tim (bpm) | float |
| RR | Nhịp thở (lần/phút) | float |
| BT | Nhiệt độ cơ thể (°C) | float |
| Saturation | Độ bão hòa oxy (%) | float |
| SBP | Huyết áp tâm thu (mmHg) | float |
| DBP | Huyết áp tâm trương (mmHg) | float |
| KTAS_expert | Mức triage (1–7) | int |

### 2.3 Thách thức

1. **Mất cân bằng dữ liệu nghiêm trọng:**
    - Level 1: chỉ 2 mẫu
    - Level 5: chỉ 18 mẫu
    - Level 3 chiếm đa số (~421 mẫu)
2. **Thang KTAS 1–7** trong khi đề tài yêu cầu 5 mức
3. **Nhiễu dữ liệu** do encoding cp1252, các giá trị không phải số trong cột Age
4. **561 mẫu sạch** — quá ít cho mô hình ML

---

## 3. Tiền xử lý dữ liệu (7 bước)

Script: `src/01_preprocessing.py`

### Bước 1: Thu thập và mapping cột

```python
# Ánh xạ tự động tên cột gốc sang chuẩn
mapping = _infer_required_columns(df)
df = _normalize_columns(df, mapping)
```

Hàm `_infer_required_columns()` quét tên cột theo lowercase, hỗ trợ đa dạng tên gọi:
- `Heart_Rate`, `HR`, `heartrate` → `heart_rate`
- `BT`, `body_temperature` → `temperature`
- `SBP`, `systolic blood pressure` → `systolic_bp`

### Bước 2: Làm sạch (Clean)

```python
df = clean_data(df)
```

- Xóa dòng trùng lặp
- Chuyển đổi kiểu dữ liệu số
- Lọc giá trị bất thường (outliers):
  - Tuổi: 0–120
  - SpO₂: ≤100%
  - Nhịp tim: ≤250 bpm
  - Nhiệt độ: ≤45°C

### Bước 3: Xử lý dữ liệu thiếu

```python
df = handle_missing(df)
```

- Cột thiếu ≤20%: điền bằng median
- Cột thiếu >20%: điền bằng KNN (k=5)
- Gender: điền bằng mode

### Bước 4: Chuẩn hóa

```python
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

Đưa tất cả đặc trưng số về miền [0, 1].

### Bước 5: Mã hóa

```python
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["gender"]])
# → gender_Nam, gender_Nữ
```

### Bước 6: Chọn đặc trưng

Giữ lại tất cả 16 đặc trưng đã qua xử lý. Pipeline sẵn sàng tích hợp SelectKBest trong tương lai.

### Bước 7: Xuất dữ liệu

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **Dataset cuối:** `dataset_final.csv` (8,361 mẫu đã scale + encode)
- **Train/Test:** 80/20, stratified theo triage_level
- **Scaler:** `scaler.pkl` (dùng lại cho inference trong app)

### Kết quả tiền xử lý

| Giai đoạn | Số mẫu |
|-----------|--------|
| Dữ liệu thô | ~21,500 |
| KTAS 1–5 hợp lệ | 572 |
| Sau outliers | 561 |
| KNN imputation | 561 |
| MinMaxScaler | 561 |
| + Synthetic | **8,361** |

---

## 4. Feature Engineering

Từ 7 dấu hiệu sinh tồn gốc, xây dựng thêm **9 đặc trưng** có ý nghĩa lâm sàng:

### 4.1 Các đặc trưng liên tục

| Đặc trưng | Công thức | Ý nghĩa lâm sàng |
|-----------|-----------|-------------------|
| **Pulse Pressure** | PP = SBP − DBP | Đo chênh lệch áp lực mạch, giảm trong sốc tim |
| **Shock Index** | SI = HR / SBP | Chỉ số sốc, SI > 0.7 báo hiệu sốc mất máu |
| **MAP (Mean Arterial Pressure)** | MAP = (SBP + 2×DBP) / 3 | Áp lực động mạch trung bình, MAP < 65 mmHg là nguy kịch |

### 4.2 Binary Flags

| Đặc trưng | Điều kiện | Ý nghĩa |
|-----------|-----------|---------|
| Tachycardia | HR > 100 bpm | Nhịp tim nhanh |
| Bradycardia | HR < 60 bpm | Nhịp tim chậm |
| Hypotension | SBP < 90 mmHg | Hạ huyết áp |
| Hypoxia | SpO₂ < 90% | Thiếu oxy máu |
| Fever | Nhiệt độ ≥ 38°C | Sốt |
| Tachypnea | RR > 20 lần/ph | Thở nhanh |

### 4.3 Đầu vào mô hình

**16 đặc trưng:** age, gender_Nam, gender_Nữ, heart_rate, respiratory_rate, temperature, spo2, systolic_bp, diastolic_bp, pulse_pressure, shock_index, map, tachycardia, bradycardia, hypotension, hypoxia, fever, tachypnea

---

## 5. Synthetic Data Generation

### 5.1 Tại sao cần synthetic data?

561 mẫu thực là không đủ, đặc biệt với:
- Level 1 chỉ có 2 mẫu
- Level 5 chỉ có 18 mẫu
→ Các mô hình sẽ không học được pattern của nhóm thiểu số

### 5.2 Rule-based Clinical Generator

Script: `src/synthetic_generator.py` + `src/clinical_rules.py`

Thiết kế khoảng giá trị lâm sàng riêng cho từng mức triage:

| Cấp | HR (bpm) | RR | Nhiệt độ (°C) | SpO₂ (%) | SBP/DBP (mmHg) |
|-----|----------|----|---------------|----------|----------------|
| 1 | 140–180 | 30–45 | 34–35 / 39–41 | 70–85 | 60–85 / 35–55 |
| 2 | 115–145 | 24–32 | 38.5–39.5 | 85–91 | 85–100 / 50–70 |
| 3 | 95–120 | 20–26 | 37.8–38.8 | 91–95 | 95–115 / 60–80 |
| 4 | 70–95 | 14–20 | 36.8–37.8 | 95–98 | 110–135 / 70–85 |
| 5 | 60–80 | 12–18 | 36.3–37.2 | 98–100 | 115–130 / 70–85 |

### 5.3 Quy trình sinh

```python
def generate_patient(level, rng):
    # Lấy mẫu từ VITAL_RANGES[level]
    # Thêm nhiễu ±5%
    # Tính feature engineering
    # Gán triage_level = level
```

- Nhiễu ±5% tạo đa dạng sinh học
- Level 1 có 2 khoảng nhiệt độ song song (hạ thân nhiệt/sốt cao)

### 5.4 Phân bố mục tiêu

| Level | Số lượng | % |
|-------|----------|---|
| 1 | 800 | 9.6% |
| 2 | 1,500 | 17.9% |
| 3 | 2,500 | 29.9% |
| 4 | 1,800 | 21.5% |
| 5 | 1,200 | 14.4% |
| **Tổng** | **7,800** | **100%** |

### 5.5 Dataset tổng hợp

| Nguồn | Số mẫu |
|-------|--------|
| Dữ liệu thực (KTAS cleaned) | 561 |
| Synthetic generator | 7,800 |
| **Tổng** | **8,361** |

Tỉ lệ synthetic:real ≈ 14:1. Toàn bộ dataset được trộn ngẫu nhiên, lưu tại `data/raw/augmented_ktas.csv`.

---

## 6. Huấn luyện mô hình

### 6.1 Random Forest

Script: `src/02_random_forest.py`

**Tham số:**

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

**Giải thích lựa chọn tham số:**
- `n_estimators=300`: cân bằng giữa độ chính xác và thời gian huấn luyện. Random Forest hội tụ về độ chính xác tối ưu tại ~200–300 cây.
- `max_depth=12`: ngăn overfitting. 16 đặc trưng, depth 12 giúp cây đủ sâu để nắm pattern phức tạp nhưng không quá sâu.
- `class_weight="balanced"`: tự động điều chỉnh trọng số theo tần suất lớp — quan trọng vì synthetic data vẫn duy trì imbalance.
- `random_state=42`: đảm bảo tái lập kết quả.

**Stratified K-Fold Cross Validation (5 folds):**

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Stratified đảm bảo tỉ lệ các lớp trên mỗi fold giống tỉ lệ tổng thể, tránh fold chỉ có level 3–4 không có level 1.

**Kết quả CV (Random Forest, 5 folds):**

| Fold | Accuracy |
|------|----------|
| 1 | ~0.912 |
| 2 | ~0.909 |
| 3 | ~0.915 |
| 4 | ~0.911 |
| 5 | ~0.913 |
| **Mean** | **~0.912** |

Độ lệch chuẩn thấp (~0.002) — mô hình ổn định qua các fold.

### 6.2 XGBoost

Script: `src/03_xgboost_evaluation.py`

**Label mapping:** XGBoost yêu cầu nhãn 0-based.

```python
label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
# Dự đoán xong cộng +1 để trả về nhãn gốc
```

**Tham số:**

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
)
```

**Giải thích lựa chọn tham số:**
- `max_depth=6`: XGBoost nhạy hơn RF với depth sâu. Depth 6 là điểm tối ưu — đủ sâu cho 16 features nhưng tránh overfitting tốt hơn RF.
- `learning_rate=0.1`: learning rate thấp, cần nhiều cây hơn để đạt hội tụ, nhưng mô hình tổng quát hóa tốt hơn.
- `eval_metric="mlogloss"`: phù hợp multi-class (5 lớp), đo cross-entropy loss.

---

## 7. Đánh giá và so sánh

### 7.1 Metrics

| Chỉ số | Random Forest | XGBoost |
|--------|---------------|---------|
| **Accuracy** | **91.33%** | 91.09% |
| **Precision** | **91.47%** | 91.07% |
| **Recall** | 91.33% | **91.09%** |
| **F1** | **91.35%** | 91.07% |
| **ROC-AUC** | 99.23% | **99.26%** |

RF nhỉnh hơn XGBoost trên các metrics chính (Accuracy, F1), nhưng XGBoost có ROC-AUC cao hơn — cho thấy khả năng xếp hạng xác suất tốt hơn.

### 7.2 Confusion Matrix — Random Forest

```
          Dự đoán
          C1   C2   C3   C4   C5
Thực C1 [161,  0,   0,   0,   0]
tế   C2 [0,   308, 13,  8,   0]
     C3 [0,   8,   513, 21,  1]
     C4 [0,   8,   11,  326, 51]
     C5 [0,   0,   3,   21,  220]
```

**Nhận xét:**
- Level 1 đạt **100%** — không có false negative nào
- Lỗi chủ yếu giữa Level 4 ↔ Level 5 (51 + 21 + 21 mẫu) — ranh giới mờ
- Level 3 ổn định với 513/543 đúng (~94.5%)

### 7.3 Confusion Matrix — XGBoost

```
          Dự đoán
          C1   C2   C3   C4   C5
Thực C1 [161,  0,   0,   0,   0]
tế   C2 [0,   305, 16,  8,   0]
     C3 [0,   6,   523, 14,  0]
     C4 [0,   10,  14,  332, 40]
     C5 [0,   0,   3,   38,  203]
```

**Nhận xét:**
- Level 1 cũng đạt 100%
- XGBoost nhỉnh hơn RF ở Level 3 (523 vs 513) và Level 4 (332 vs 326)
- RF nhỉnh hơn XGBoost ở Level 5 (220 vs 203)

### 7.4 Feature Importance — Random Forest

| Feature | Importance | Tích lũy |
|---------|-----------|----------|
| **Shock Index** | **24.73%** | 24.73% |
| **Heart Rate** | **14.85%** | 39.58% |
| **Respiratory Rate** | **14.68%** | 54.26% |
| **MAP** | **11.91%** | 66.17% |
| Systolic BP | 9.82% | 75.99% |
| Tachypnea | 5.70% | 81.69% |
| SpO₂ | 5.47% | 87.16% |
| Diastolic BP | 3.50% | 90.66% |
| Tachycardia | 3.01% | 93.67% |
| Temperature | 1.93% | 95.60% |
| Hypotension | 1.31% | 96.91% |
| Hypoxia | 1.25% | 98.16% |
| Pulse Pressure | 0.79% | 98.95% |
| Age | 0.60% | 99.55% |
| Fever | 0.19% | 99.74% |
| gender_Nam | 0.11% | 99.85% |

**Shock Index chiếm gần 25%** — đây là đặc trưng quan trọng nhất. Heart Rate và Respiratory Rate bổ sung thêm ~30%.

### 7.5 Phân tích chi tiết từng level

| Level | RF Precision | RF Recall | XGB Precision | XGB Recall |
|-------|-------------|-----------|--------------|------------|
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.9516 | 0.9362 | 0.9502 | 0.9271 |
| 3 | 0.9500 | 0.9448 | 0.9406 | 0.9632 |
| 4 | 0.8670 | 0.8232 | 0.8686 | 0.8384 |
| 5 | 0.8088 | 0.9016 | 0.8354 | 0.8320 |

- Level 1 hoàn hảo: 100% ở cả hai mô hình
- Level 2–3: RF và XGB đều ổn (~93–96%)
- Level 4–5: thấp hơn (81–90%) do overlap tự nhiên giữa nhẹ và rất nhẹ

---

## 8. Ensemble Voting Classifier

Script: `src/ensemble.py`

Kết hợp Random Forest và XGBoost qua **Soft Voting**:

```python
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)
```

Soft voting tính trung bình xác suất dự đoán, thường tốt hơn hard voting trong multi-class vì tận dụng độ tin cậy.

### Kết quả Ensemble

| Metric | RF | XGB | **Ensemble** |
|--------|-----|-----|-------------|
| Accuracy | 91.33% | 91.09% | **91.80%** |
| Precision | 91.47% | 91.07% | **91.85%** |
| Recall | 91.33% | 91.09% | **91.80%** |
| F1 | 91.35% | 91.07% | **91.81%** |
| ROC-AUC | 99.23% | 99.26% | **98.92%** |

Ensemble cải thiện ~0.5% trên Accuracy, Precision, Recall, F1 so với từng mô hình riêng lẻ. ROC-AUC giảm nhẹ (~0.3%) — có thể do softening phân phối xác suất.

---

## 9. Giải thích mô hình với SHAP

Script: `src/04_explainable_ai.py`

### 9.1 SHAP Summary Plot

`report/shap_summary.png`

Phân bố SHAP values cho từng đặc trưng trên tập test:
- Trục X: SHAP value (tác động lên đầu ra)
- Màu sắc: giá trị đặc trưng (đỏ = cao, xanh = thấp)
- Mỗi điểm là một bệnh nhân

### 9.2 SHAP Bar Plot

`report/shap_bar.png`

SHAP feature importance (mean |SHAP|) xác nhận:
1. **Heart Rate** và **Shock Index** dẫn đầu về tác động trung bình
2. **Respiratory Rate** và **MAP** theo sau
3. Gender và Fever gần như không ảnh hưởng

### 9.3 Giải thích y học cho SHAP top features

- **Shock Index (SI = HR/SBP):** Chỉ số sốc cao → tim đập nhanh bù trừ huyết áp tụt → dấu hiệu mất máu/sốc → triage level 1–2
- **Heart Rate:** Nhịp tim > 120 bpm gắn với sốc, loạn nhịp → level cao
- **Respiratory Rate:** Thở nhanh > 24 lần/ph → suy hô hấp, nhiễm toan → level cao
- **MAP:** MAP < 65 mmHg → tưới máu mô kém, nguy cơ suy tạng → level cao

---

## 10. Ứng dụng Streamlit

Script: `src/app.py`

### 10.1 Kiến trúc ứng dụng

Ứng dụng Streamlit 7 trang:

| Trang | Chức năng |
|-------|-----------|
| 🏠 Tổng quan | Hero dashboard, KPI badges, hành trình đồ án |
| 🥇 Dự đoán | Form nhập chỉ số, kết quả 2 model, biểu đồ, phân tích |
| 📋 Quy trình | Chi tiết preprocessing → feature engineering → training |
| 📊 Kết quả | Metrics, confusion matrix heatmaps, nhận xét |
| 📈 Khám phá | Phân bố dữ liệu, KDE plots, thống kê mô tả |
| 🔬 XAI | SHAP images, feature importance, phân tích top yếu tố |
| 👥 Giới thiệu | Thành viên, kết luận |

### 10.2 Chức năng Dự đoán chính

- Input: 8 chỉ số + giới tính (8-column inline form)
- Output:
  - 2 card lớn (RF + XGB) — level, label, confidence %, progress bar
  - Consensus bar (Shock Index, MAP, đồng thuận)
  - Bar chart xác suất 5 mức
  - Radar so sánh BN với triage level kỳ vọng
  - 8 factor analysis cards (🔴🟡✅ severity)
  - Summary phát hiện dấu hiệu bất thường

### 10.3 Công nghệ

- **Streamlit** — frontend framework
- **Matplotlib + Seaborn** — biểu đồ
- **Joblib** — load model/scaler
- **SHAP images** — XAI trực quan

---

## 11. Kết luận và hướng phát triển

### 11.1 Kết quả đạt được

- ✅ Xây dựng pipeline tiền xử lý 7 bước hoàn chỉnh
- ✅ Feature Engineering với 9 đặc trưng lâm sàng
- ✅ Rule-based Clinical Generator sinh 7,800 mẫu synthetic
- ✅ Random Forest đạt **91.33% accuracy**, XGBoost **91.09%**
- ✅ Ensemble Voting đạt **91.80% accuracy**
- ✅ Level 1 (nguy kịch) phát hiện 100%
- ✅ SHAP giải thích được feature importance
- ✅ Giao diện Streamlit đầy đủ 7 trang

### 11.2 Hạn chế

- ❌ Synthetic data chiếm tỉ lệ lớn (14:1) — chưa reflect đầy đủ phức tạp thực tế
- ❌ Level 4–5 chưa phân biệt tốt (Confusion: 51 mẫu C4→C5, 21 mẫu C5→C4)
- ❌ Dữ liệu chỉ từ 1 nguồn KTAS Hàn Quốc, khó generalize sang dân số Việt Nam
- ❌ Gender ít ảnh hưởng (0.11%) — có thể bỏ qua để giảm dimensionality

### 11.3 Hướng phát triển

1. **Tích hợp dữ liệu thực từ bệnh viện Việt Nam** — tăng tính ứng dụng thực tế
2. **Deep Learning** — thử LSTM/Transformers cho chuỗi thời gian vital signs
3. **Multi-modal** — kết hợp dấu hiệu sinh tồn + hình ảnh + bệnh sử
4. **Cải thiện Level 4–5** — thêm đặc trưng, fine-tune threshold
5. **Clinical Decision Support** — tích hợp gợi ý phác đồ xử trí
6. **Triển khai Web** — Streamlit Cloud / HuggingFace Spaces

---

## Thông tin nhóm

| Họ và tên | MSSV | Vai trò |
|-----------|------|---------|
| **Đặng Hoàng Ân** | 080206014982 | Nhóm trưởng — Kiến trúc hệ thống, RF, giao diện |
| Lê Hiền Đức | 068206002504 | Thành viên — Dataset, tiền xử lý |
| Nguyễn Ngọc Phương | 075306018872 | Thành viên — XGBoost, đánh giá |
| Nguyễn Thị Thảo Trang | 040306018361 | Thành viên — SHAP, báo cáo |

---

## Tài liệu tham khảo

1. Korean Triage and Acuity Scale (KTAS) — https://doi.org/10.3346/jkms.2021.36.e285
2. Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001).
3. Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD 2016.
4. Lundberg, S. & Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
5. Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830 (2011).

---

<div align="center">

**© 2026 Đặng Hoàng Ân — KaiyoDang**

Đồ án Trí tuệ nhân tạo — Giảng viên: Bùi Trọng Hiếu

</div>
