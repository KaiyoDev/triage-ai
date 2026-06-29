# 🏥 HỆ THỐNG AI HỖ TRỢ XẾP THỨ TỰ KHÁM BỆNH (TRIAGE AI)

## 📖 Giới thiệu

Triage AI là đồ án môn **Trí tuệ nhân tạo** nhằm xây dựng một hệ thống hỗ trợ phân loại mức độ ưu tiên khám bệnh dựa trên các chỉ số sức khỏe của bệnh nhân.

Hệ thống sử dụng các thuật toán Machine Learning để dự đoán mức độ ưu tiên khám bệnh, giúp giảm thời gian chờ đợi, hỗ trợ nhân viên y tế xử lý các trường hợp khẩn cấp và nâng cao hiệu quả vận hành tại bệnh viện.

---

# 🎯 Mục tiêu đề tài

* Hỗ trợ phân loại bệnh nhân theo mức độ ưu tiên.
* Giảm thời gian chờ khám bệnh.
* Hỗ trợ nhân viên y tế đưa ra quyết định nhanh hơn.
* Xây dựng hệ thống AI có khả năng giải thích kết quả dự đoán.
* Trực quan hóa kết quả để dễ dàng theo dõi và đánh giá.

---

# 🏗️ Sơ đồ kiến trúc tổng thể hệ thống

![Kiến trúc tổng thể hệ thống Triage AI](images/01_system_architecture.png)

---

# 🔄 Luồng hoạt động của hệ thống

![Luồng hoạt động của hệ thống](images/02_workflow.png)


---

# 🧠 Quy trình xử lý dữ liệu AI

![AI Pipeline](images/03_ai_pipeline.png)

---

# 🤖 Thuật toán sử dụng

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

# 🌲 Luồng hoạt động Random Forest

![Random Forest Workflow](images/04_random_forest.png)

---

# ⚡ Luồng hoạt động XGBoost

![Luồng hoạt động XGBoost](images/05_xgboost.png)

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

# 🩺 Phân loại mức độ ưu tiên khám bệnh

![Triage Level](images/06_triage_level.png)


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
| Đặng Hoàng Ân         | 0802....4982 | Kiến trúc hệ thống, Random Forest, giao diện |
| Lê Hiền Đức           | 0682....2504 | Dataset và tiền xử lý dữ liệu                |
| Nguyễn Ngọc Phương    | 0753....8872 | XGBoost và đánh giá mô hình                  |
| Nguyễn Thị Thảo Trang | 0403....8361 | SHAP, kết luận và tổng hợp báo cáo           |

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

---

# 📌 DANH SÁCH SƠ ĐỒ CẦN VẼ BẰNG DRAW.IO

| STT | Tên sơ đồ                   | Người phụ trách       |
| --- | --------------------------- | --------------------- |
| 01  | Kiến trúc tổng thể hệ thống | Đặng Hoàng Ân         |
| 02  | Luồng hoạt động hệ thống    | Đặng Hoàng Ân         |
| 03  | AI Pipeline                 | Đặng Hoàng Ân         |
| 04  | Random Forest Workflow      | Đặng Hoàng Ân         |
| 05  | XGBoost Workflow            | Đặng Hoàng Ân         |
| 06  | Triage Level                | Đặng Hoàng Ân         |

```
```
