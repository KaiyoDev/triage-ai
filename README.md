# 🏥 HỆ THỐNG AI HỖ TRỢ XẾP THỨ TỰ KHÁM BỆNH (TRIAGE AI)

## 📖 Giới thiệu

Triage AI là đồ án môn **Trí tuệ nhân tạo** nhằm xây dựng một hệ thống hỗ trợ phân loại mức độ ưu tiên khám bệnh dựa trên các chỉ số sức khỏe của bệnh nhân.

Hệ thống sử dụng các thuật toán Machine Learning để dự đoán mức độ ưu tiên khám bệnh, giúp giảm thời gian chờ đợi, hỗ trợ nhân viên y tế xử lý các trường hợp khẩn cấp và nâng cao hiệu quả vận hành tại bệnh viện.

---

## 🎯 Mục tiêu đề tài

* Hỗ trợ phân loại bệnh nhân theo mức độ ưu tiên.
* Giảm thời gian chờ khám bệnh.
* Hỗ trợ nhân viên y tế đưa ra quyết định nhanh hơn.
* Xây dựng hệ thống AI có khả năng giải thích kết quả dự đoán.
* Trực quan hóa kết quả để dễ dàng theo dõi và đánh giá.

---

# 🏗️ Sơ đồ kiến trúc hệ thống

```text
                    HỆ THỐNG TRIAGE AI

┌────────────────────┐
│      BỆNH NHÂN     │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ NHẬP THÔNG TIN     │
│ SỨC KHỎE           │
└────────────────────┘
          │
          ▼
┌────────────────────────────────┐
│      TIỀN XỬ LÝ DỮ LIỆU        │
│                                │
│ • Làm sạch dữ liệu             │
│ • Xử lý dữ liệu thiếu          │
│ • Mã hóa dữ liệu               │
│ • Chuẩn hóa dữ liệu            │
│ • Chọn đặc trưng               │
└────────────────────────────────┘
          │
          ▼
┌────────────────────┐
│ CHIA TRAIN / TEST  │
│ 80% / 20%          │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ RANDOM FOREST      │
│ (MÔ HÌNH CHÍNH)    │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ XGBOOST            │
│ (MÔ HÌNH SO SÁNH)  │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ ĐÁNH GIÁ MÔ HÌNH   │
│                    │
│ • Accuracy         │
│ • Precision        │
│ • Recall           │
│ • F1-score         │
│ • ROC-AUC          │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ GIẢI THÍCH AI      │
│                    │
│ • SHAP             │
│ • Feature Importance│
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ HIỂN THỊ KẾT QUẢ   │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│ NHÂN VIÊN Y TẾ     │
└────────────────────┘
```

---

# 🔄 Quy trình hoạt động của hệ thống

```text
Dữ liệu bệnh nhân

↓

Tiền xử lý dữ liệu

↓

Chia tập Train/Test

↓

Huấn luyện mô hình AI

↓

Đánh giá hiệu suất

↓

Giải thích kết quả AI

↓

Hiển thị kết quả dự đoán
```

---

# 🧠 Thuật toán sử dụng

## Mô hình chính: Random Forest

Nhiệm vụ:

* Dự đoán mức độ ưu tiên khám bệnh.
* Tối ưu độ chính xác.
* Giảm hiện tượng quá khớp (Overfitting).

Tham số dự kiến:

```text
n_estimators = 300

max_depth = 12

min_samples_split = 5

min_samples_leaf = 2

random_state = 42
```

---

## Mô hình so sánh: XGBoost

Nhiệm vụ:

* So sánh hiệu suất với Random Forest.
* Kiểm tra khả năng cải thiện kết quả dự đoán.

Tham số dự kiến:

```text
n_estimators = 300

max_depth = 6

learning_rate = 0.1

random_state = 42
```

---

# 📊 Chỉ số đánh giá

Hệ thống sẽ được đánh giá thông qua các chỉ số:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

# 🛠️ Công nghệ sử dụng

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

---

# 👥 Thành viên thực hiện

| Họ và tên             | MSSV         | Nhiệm vụ                                     |
| --------------------- | ------------ | -------------------------------------------- |
| Đặng Hoàng Ân         | 080.....4982 | Kiến trúc hệ thống, Random Forest, giao diện |
| Lê Hiền Đức           | 068.....2504 | Dataset và tiền xử lý dữ liệu                |
| Nguyễn Ngọc Phương    | 075.....8872 | XGBoost và đánh giá mô hình                  |
| Nguyễn Thị Thảo Trang | 040.....8361 | SHAP, kết luận và tổng hợp báo cáo           |

---

# 📁 Cấu trúc dự án

```text
triage-ai/

README.md

requirements.txt

data/
│
├── raw/
└── processed/

src/
│
├── 01_preprocessing.py
├── 02_random_forest.py
├── 03_xgboost_evaluation.py
└── 04_explainable_ai.py

models/

images/

report/
```
