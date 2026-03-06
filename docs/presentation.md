---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 22px;
  }
  h1 { color: #c0392b; font-size: 40px; }
  h2 { color: #2c3e50; border-bottom: 2px solid #c0392b; padding-bottom: 6px; }
  h3 { color: #c0392b; }
  table { font-size: 18px; width: 100%; }
  th { background: #c0392b; color: white; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
---

# Heart Disease Diagnosis
## Using Ensemble Learning

**AIO2025 Research Project | VietAI Learning Team**

> Chẩn đoán bệnh tim bằng học máy kết hợp nhiều mô hình

**Team:** Dũng · Anh · Vinh · Hằng · Huy

---

## Mục tiêu

- Xây dựng hệ thống **tự động chẩn đoán bệnh tim** dựa trên các chỉ số lâm sàng
- So sánh **9 thuật toán ML** để tìm mô hình tốt nhất
- Tối ưu hyperparameters tự động với **Optuna**
- Triển khai ứng dụng **web demo** với Streamlit

---

## Bộ dữ liệu

**Cleveland Heart Disease Dataset — UCI Machine Learning Repository**

| Thuộc tính | Chi tiết |
|---|---|
| Tổng mẫu | 303 bệnh nhân |
| Features | 13 đặc trưng lâm sàng |
| Target | 0 = Healthy, 1 = Disease |
| Phân chia | 80% train / 10% val / 10% test |
| Phân bố | ~54% Disease / ~46% Healthy |

**13 features:** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

---

## Pipeline tổng quan

```
Cleveland Dataset (303 mẫu)
        │
        ├── Train/Val/Test Split (80/10/10)
        │
        ├── Feature Engineering  ──→  Raw / FE / DT / FE-DT
        │
        ├── Preprocessing
        │     ├── Scaler (Standard / MinMax / Robust)
        │     └── OneHotEncoder
        │
        ├── Feature Selection
        │     (Variance / Correlation / KBest-MI / RFE / Boruta)
        │
        ├── Optuna Hyperparameter Tuning (100 trials × 5-fold CV)
        │
        └── Evaluation trên Test Set
```

---

## Các bộ dữ liệu tạo ra

| Dataset | Features | Mô tả |
|---|---|---|
| **raw** | 13 | Dữ liệu gốc, không xử lý thêm |
| **fe** | 13 | Feature Engineering: OHE + tạo `hr_ratio`, `chol_per_age`... |
| **dt** | 10 | Chọn 10/13 features quan trọng nhất bằng Decision Tree |
| **fe_dt** | 10 | FE trước → Decision Tree chọn 10 features quan trọng nhất |

Optuna thử tất cả 4 bộ → chọn bộ tốt nhất cho từng model

---

## Feature Engineering

Từ 13 features gốc → tạo thêm features mới:

| Feature mới | Công thức | Ý nghĩa |
|---|---|---|
| `hr_ratio` | `thalach / age` | Nhịp tim tối đa tương đối theo tuổi |
| `chol_per_age` | `chol / age` | Cholesterol tương đối theo tuổi |
| `thal_3.0`, `thal_7.0` | OHE(`thal`) | Loại thalassemia |
| `cp_3.0`, `cp_4.0` | OHE(`cp`) | Loại đau ngực |
| `exang_0.0`, `exang_1.0` | OHE(`exang`) | Đau thắt ngực khi gắng sức |

**Mục đích:** Giúp model học quan hệ phi tuyến, bổ sung thông tin lâm sàng

---

## Feature Selection

Loại bỏ features dư thừa sau FE:

| Phương pháp | Cách hoạt động | Đặc điểm |
|---|---|---|
| **Variance** | Loại features có phương sai thấp | Nhanh, đơn giản |
| **Correlation** | Loại features tương quan cao nhau | Giảm multicollinearity |
| **KBest-MI** | Chọn K features có Mutual Information cao nhất | Phổ biến, hiệu quả |
| **RFE-SVM** | Loại dần features kém quan trọng theo SVM | Chính xác, chậm |
| **Boruta** | So sánh features với "shadow features" ngẫu nhiên | Robust nhất |

---

## Tối ưu Hyperparameter

**Optuna (TPE — Tree-structured Parzen Estimator)**

```
Optuna (vòng ngoài — 100 trials)
│
├── Trial 1: thử {n_estimators=100, max_depth=3}
│       └── 5-fold Cross-Validation → AUC = 0.89
│
├── Trial 2: thử {n_estimators=200, max_depth=5}
│       └── 5-fold Cross-Validation → AUC = 0.91
│   ...
└── Trial 100: best AUC = 0.93 → lưu hyperparameters
```

- **Cross-validation:** đánh giá khách quan mỗi bộ tham số
- **Optuna:** quyết định thử tham số nào tiếp theo (thông minh hơn Grid/Random Search)
- **Metric tối ưu:** ROC-AUC

---

## 9 Mô hình được đánh giá

1. **Logistic Regression** — baseline tuyến tính
2. **Decision Tree** — cây quyết định đơn
3. **Random Forest** — ensemble cây ngẫu nhiên
4. **AdaBoost** — boosting tuần tự
5. **Gradient Boosting** — boosting gradient
6. **XGBoost** — gradient boosting tối ưu
7. **LightGBM** — gradient boosting nhanh
8. **Support Vector Machine** — phân loại siêu phẳng
9. **K-Nearest Neighbors** — dựa trên khoảng cách

---

## Kết quả

| Model | CV AUC | Test AUC | Accuracy | Recall | F1 |
|---|---|---|---|---|---|
| **Gradient Boosting** | 0.903 | **0.9545** | **91.8%** | **92.9%** | **91.2%** |
| Logistic Regression | 0.947 | 0.9567 | 88.5% | 92.9% | 88.1% |
| K-Nearest Neighbors | 0.922 | 0.9540 | 90.2% | 92.9% | 89.7% |
| XGBoost | 0.900 | 0.9437 | 90.2% | 92.9% | 89.7% |
| LightGBM | 0.905 | 0.9470 | 86.9% | 89.3% | 86.2% |
| SVM | 0.935 | 0.9556 | 83.6% | 82.1% | 82.1% |
| Random Forest | 0.939 | 0.9361 | 83.6% | 82.1% | 82.1% |
| AdaBoost | 0.904 | 0.9426 | 85.3% | 89.3% | 84.8% |
| Decision Tree | 0.856 | 0.8864 | 83.6% | 82.1% | 82.1% |

**Average Test AUC: 0.940**

---

## Metrics đánh giá

```
              Predicted
            Healthy   Disease
Actual  H  [  TN   ] [  FP   ]
        D  [  FN   ] [  TP   ]
```

| Metric | Công thức | Ý nghĩa trong y tế |
|---|---|---|
| **Accuracy** | (TP+TN)/Total | Tỷ lệ dự đoán đúng tổng thể |
| **Precision** | TP/(TP+FP) | Trong số báo bệnh, đúng bao nhiêu? |
| **Recall** | TP/(TP+FN) | Phát hiện được bao nhiêu % bệnh nhân? |
| **F1-Score** | 2×P×R/(P+R) | Cân bằng Precision & Recall |
| **ROC-AUC** | Area under ROC curve | Khả năng phân biệt 2 class |

> **Recall quan trọng nhất** trong bài toán y tế — bỏ sót bệnh nhân nguy hiểm hơn cảnh báo nhầm

---

## Feature Importance & SHAP

**SHAP (SHapley Additive exPlanations)** — giải thích từng dự đoán cụ thể

```
prediction = base_value + SHAP(thal) + SHAP(cp) + SHAP(ca) + ...
           = 0.45       + (+0.22)    + (+0.18)   + (+0.15)  + ...
           = 0.72  →  High Risk (72%)
```

| | Feature Importance | SHAP |
|---|---|---|
| Phạm vi | Toàn dataset | Từng bệnh nhân |
| Chiều tác động | Không có | Có (+/-) |
| Model-agnostic | Không | Có |

**Ứng dụng:** Bác sĩ biết được *tại sao* model dự đoán bệnh nhân cụ thể là High Risk

---

## Ứng dụng Web — Streamlit

**5 tính năng chính:**

1. **Patient Diagnosis** — nhập thông số → dự đoán từ 9 models + majority voting
2. **Model Analysis** — so sánh hiệu suất, hyperparameters
3. **Feature Importance** — SHAP cho từng model
4. **Experiment Tracking** — lịch sử 40 experiments, filter, export
5. **History & Reports** — lưu lịch sử bệnh nhân, xuất PDF

**Live Demo:** https://heart-disease-diagnosis-vietailearningteam.streamlit.app

---

## Kiến trúc hệ thống

```
heart-disease-diagnosis/
├── app/
│   └── streamlit_app.py        # Giao diện web
├── src/
│   ├── pipeline.py             # Load model, predict
│   ├── model_functions.py      # Feature Engineering classes
│   └── utils/app_utils.py      # SHAP, PDF report, history
├── scripts/
│   ├── train_models.py         # Optuna tuning
│   └── experiment_manager.py  # Tracking experiments
├── models/saved_models/latest/ # 10 file .pkl đã train
├── data/
│   ├── raw/                    # raw_train/val/test.csv
│   └── processed/              # fe, dt, fe_dt datasets
└── experiments/
    └── experiment_log.json     # 40 experiments logged
```

---

## Kết luận

**Đạt được:**
- Average Test AUC **0.940** trên 9 models
- Best model: **Gradient Boosting** (AUC 0.9545, Accuracy 91.8%, Recall 92.9%)
- Pipeline tự động: FE → Feature Selection → Optuna → Evaluation
- Ứng dụng web đầy đủ tính năng, deploy trên Streamlit Cloud

**Hạn chế:**
- Dataset nhỏ (n=303), chỉ từ Cleveland clinic
- Chưa validate trên dataset ngoài
- Không có dữ liệu temporal (xu hướng theo thời gian)

**Hướng phát triển:**
- Thêm SMOTE cho imbalanced data
- Stacking Ensemble thay vì chỉ Voting
- Validate trên Hungary, Switzerland datasets (cùng UCI repo)

---

## Cảm ơn

**VietAI Learning Team — AIO2025**

| | |
|---|---|
| Dataset | UCI Cleveland Heart Disease |
| Framework | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna (TPE, 100 trials/model) |
| UI | Streamlit |
| Explainability | SHAP |

> *"For Educational/Research Purposes Only.*
> *Always consult qualified healthcare professionals for medical decisions."*
