# BÁO CÁO NGHIÊN CỨU TĂNG CƯỜNG DỮ LIỆU Y TẾ

## Slide 1 — Nghiên cứu tăng cường dữ liệu cho dự đoán bệnh tim

**Nội dung trên slide**

- Cleveland Heart Disease Dataset
- Synthetic data • Hospital stress-test • Model selection
- Báo cáo kết quả nghiên cứu 2 tuần

**Thông điệp trình bày**

Mục tiêu là kiểm tra liệu dữ liệu tổng hợp có giúp mô hình chính xác và bền hơn khi gặp dữ liệu bệnh viện không hoàn hảo hay không.

---

## Slide 2 — Dataset nhỏ làm tăng rủi ro overfitting

**Nội dung trên slide**

| Thuộc tính | Giá trị |
|---|---:|
| Bệnh nhân | 303 |
| Feature | 13 |
| Không bệnh | 164 |
| Có bệnh | 139 |
| Missing values | 6 |

**Kết luận chính**

303 bệnh nhân chưa đủ đại diện cho độ phức tạp và sai lệch của dữ liệu bệnh viện thực tế.

**Thông điệp trình bày**

Dataset khá cân bằng, vì vậy mục tiêu augmentation không chỉ là cân bằng lớp mà còn là kiểm tra độ bền trước missing, outlier và distribution shift.

---

## Slide 3 — Thiết kế thí nghiệm ngăn data leakage

**Nội dung trên slide**

```text
303 bệnh nhân
      ↓
Stratified train/test split
      ↓
Train 80%                         Test thật 20%
      ↓                                  ↓
Fit synthesizer chỉ trên train     Clean / Severe test
      ↓
Original / x5 / x10
      ↓
Train và so sánh model
```

**Nguyên tắc**

- Test không tham gia sinh dữ liệu.
- Imputer, scaler và encoder chỉ fit trên train.
- Nhãn test không bị thay đổi khi tạo hospital noise.

---

## Slide 4 — Ba hướng tăng cường dữ liệu đã thử

**Nội dung trên slide**

### Gaussian Copula

- Nhanh, ổn định với dữ liệu nhỏ
- Mô hình hóa phân phối và tương quan

### CTGAN

- Học quan hệ phi tuyến
- Phù hợp dữ liệu numerical + categorical
- Chạy nhiều seed để đo độ ổn định

### Hospital corruption

- Missing values
- Outlier và sai đơn vị
- Làm tròn kết quả
- Sai categorical code
- Thay đổi tuổi và huyết áp

**Kết luận chính**

Synthetic data được đánh giá bằng fidelity, downstream utility và robustness — không chỉ bằng số lượng dòng.

---

## Slide 5 — Stress-test mô phỏng dữ liệu bệnh viện

**Nội dung trên slide**

| Mức | Missing | Outlier | Sai mã | Làm tròn |
|---|---:|---:|---:|---:|
| Mild | 3% | 1% | 0.5% | 5% |
| Medium | 8% | 3% | 1% | 10% |
| Severe | 15% | 5% | 2% | 20% |

**Chỉ số đánh giá**

- ROC-AUC và Recall
- False-negative rate
- Brier score
- Mức suy giảm từ clean sang severe

**Thông điệp trình bày**

Trong y tế, model tốt trên dữ liệu sạch nhưng suy giảm mạnh khi thiếu dữ liệu chưa phải model đáng tin cậy.

---

## Slide 6 — LightGBM tốt nhất trên dữ liệu thật

**Nội dung trên slide**

### Original data + LightGBM

| Chỉ số | Clean | Severe |
|---|---:|---:|
| ROC-AUC | **0.9643** | **0.9535** |
| Recall | **0.9643** | **0.8929** |
| Brier score | **0.0700** | **0.0856** |

**Kết luận chính**

LightGBM đạt hiệu năng và calibration tốt nhất khi huấn luyện trực tiếp trên dữ liệu thật.

**Thông điệp trình bày**

Trên clean test, model phát hiện xấp xỉ 27/28 trường hợp có bệnh; trên severe test còn khoảng 25/28.

---

## Slide 7 — Gaussian x5 tạo model nhẹ và robust

**Nội dung trên slide**

### Gaussian x5 dirty + Logistic Regression

| Chỉ số | Clean | Severe |
|---|---:|---:|
| ROC-AUC | 0.9491 | **0.9545** |
| Recall | 0.8929 | 0.8571 |
| Brier score | 0.0909 | 0.0931 |

- AUC drop: **−0.0054**
- Recall drop: **0.0357**
- Nhỏ, nhanh, dễ giải thích và dễ triển khai

**Kết luận chính**

Đây là phương án tốt nhất khi ưu tiên model gọn nhẹ và chịu nhiễu.

---

## Slide 8 — CTGAN x5 có ích nhưng chưa thắng baseline

**Nội dung trên slide**

### CTGAN x5 + Random Forest

| Chỉ số | Clean | Severe |
|---|---:|---:|
| ROC-AUC | 0.9318 ± 0.0136 | 0.9340 ± 0.0107 |
| Recall | 0.8810 | 0.8095 |
| Worst Recall | 0.8571 | 0.7500 |
| Brier score | 0.1311 | 0.1464 |

**Phát hiện**

- CTGAN x5 tương đối ổn định qua generator seeds.
- CTGAN x10 làm giảm hiệu năng trên phần lớn model.
- Chi phí và calibration chưa tốt bằng các phương án đơn giản hơn.

---

## Slide 9 — Synthetic x10 không đồng nghĩa với nhiều thông tin hơn

**Nội dung trên slide**

| CTGAN configuration | Clean AUC | Severe AUC |
|---|---:|---:|
| x5 + Random Forest | **0.9318** | **0.9340** |
| x10 + Random Forest | 0.9037 | 0.8907 |
| x5 + Logistic Regression | **0.9592** | 0.8589 |
| x10 + Logistic Regression | 0.9019 | 0.7897 |

**Kết luận chính**

Khi dữ liệu gốc chỉ có khoảng 242 train rows, synthetic x10 có thể lấn át tín hiệu thật và tạo pattern nhân tạo.

**Thông điệp trình bày**

Clean AUC cao chưa đủ: CTGAN x5 + Logistic Regression giảm hơn 0.10 AUC khi gặp severe test.

---

## Slide 10 — Đề xuất hai model theo hai mục tiêu

**Nội dung trên slide**

### Hiệu năng tổng thể

**Original + LightGBM**

- Clean AUC: 0.9643
- Severe AUC: 0.9535
- Severe Brier: 0.0856

### Gọn nhẹ và dễ giải thích

**Gaussian x5 dirty + Logistic Regression**

- Severe AUC: 0.9545
- AUC gần như không suy giảm
- Dễ tích hợp FastAPI

**Quyết định hiện tại**

LightGBM là empirical champion về hiệu năng; Logistic Regression là deployment candidate gọn nhẹ.

---

## Slide 11 — Kết luận và bước xác nhận tiếp theo

**Nội dung trên slide**

1. Augmentation không luôn cải thiện model.
2. x5 hợp lý hơn x10 trên Cleveland.
3. CTGAN chưa tạo đủ lợi ích để bù chi phí.
4. LightGBM thắng trên dữ liệu thật.
5. Chưa đủ cơ sở để tuyên bố SOTA lâm sàng.

**Bước tiếp theo**

- Chạy 20–30 hospital corruption seeds.
- Repeated cross-validation cho LightGBM và Logistic Regression.
- Tune threshold theo mục tiêu Recall.
- Kiểm tra calibration, fairness theo tuổi/giới tính.
- External validation bằng dữ liệu bệnh viện thật.

---

## Slide 12 — Tài liệu tham khảo

- Xu et al. **Modeling Tabular Data using Conditional GAN.** NeurIPS 2019.
- Choi et al. **Generating Multi-label Discrete Patient Records using GANs.** MLHC 2017.
- Che et al. **Boosting Deep Learning Risk Prediction with GANs for EHR.** 2017.
- Yan et al. **A Multifaceted Benchmarking of Synthetic EHR Generation Models.** 2022.
- Hernandez et al. **Synthetic Tabular Data Evaluation in the Health Domain.** 2023.

**Thông điệp kết thúc**

Đề xuất ưu tiên bằng chứng từ test thật và robustness, thay vì chọn model chỉ vì dữ liệu synthetic tạo ra điểm số cao.
