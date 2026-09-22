# SLIDE 1 — MÔ PHỎNG DỮ LIỆU BỆNH VIỆN VÀ RETRAIN MODEL

## Hospital Enrichment for Heart Disease Prediction

- Cleveland Heart Disease Dataset
- Synthetic data • Hospital noise • Model robustness
- Mục tiêu: giảm suy giảm hiệu năng khi dữ liệu thực tế không hoàn hảo

---

# SLIDE 2 — BÀI TOÁN

## Dataset gốc nhỏ và sạch hơn dữ liệu bệnh viện thực tế

| Thuộc tính | Giá trị |
|---|---:|
| Bệnh nhân thật | 303 |
| Feature đầu vào | 13 |
| Không bệnh | 164 |
| Có bệnh | 139 |
| Missing values | 6 |

### Vấn đề khi triển khai

- Dữ liệu bệnh viện có thể thiếu xét nghiệm.
- Có outlier, sai đơn vị và sai categorical code.
- Kết quả đo có thể bị làm tròn.
- Phân phối bệnh nhân khác nhau giữa các cơ sở.

> Model tốt trên dữ liệu sạch chưa chắc bền khi vận hành thực tế.

---

# SLIDE 3 — THIẾT KẾ THÍ NGHIỆM

## Chống data leakage từ bước đầu

```text
303 bệnh nhân thật
        ↓
Stratified train/test split
        │
        ├── Real train: khoảng 242 dòng
        │          ↓
        │    Gaussian Copula
        │          ↓
        │    Hospital enrichment
        │          ↓
        │    Enriched train: 3.000 dòng
        │
        └── Locked real test: khoảng 61 dòng
                   ↓
             Clean + hospital stress-test
```

### Nguyên tắc

- Generator chỉ fit trên real train.
- Locked test không tham gia sinh dữ liệu.
- Preprocessing chỉ fit trên training set.
- Test synthetic không dùng để báo cáo chất lượng lâm sàng.

---

# SLIDE 4 — MÔ PHỎNG H03 OUTPATIENT CLINIC

## Kịch bản dữ liệu phòng khám ngoại trú

| Loại nhiễu | Cấu hình |
|---|---:|
| Missing chung | 15% |
| Outlier numerical | 4% |
| Sai categorical code | 2% |
| Giá trị bị làm tròn | 20% |
| Thiếu riêng `ca/thal` | 35% |
| Population shift | Trẻ hơn 3 tuổi |

### Quy mô đánh giá

```text
5 model seeds
× 20 hospital corruption seeds
= 100 lượt đánh giá mỗi cấu hình
```

> Các tỷ lệ là giả định nghiên cứu, chưa phải thống kê của một bệnh viện thật.

---

# SLIDE 5 — DỮ LIỆU H03 LÀM MODEL CŨ SUY GIẢM

## LightGBM rất mạnh trên clean data nhưng nhạy với hospital shift

| Chỉ số | Clean | H03 | Thay đổi |
|---|---:|---:|---:|
| ROC-AUC | 0.9600 | 0.8915 | −0.0685 |
| Recall | 0.9643 | 0.7179 | −0.2464 |
| F1 | 0.9153 | 0.7674 | −0.1479 |
| Brier | 0.0806 | 0.1446 | +0.0640 |
| False negatives | 1.00 | 7.90 | +6.90 |

### Nhận xét

- Thiếu `ca/thal` làm mất tín hiệu quan trọng.
- Outlier và rounding làm thay đổi các ngưỡng chia cây.
- Recall giảm mạnh, dẫn đến nhiều ca bệnh bị bỏ sót hơn.

---

# SLIDE 6 — ENRICHMENT GIÚP LOGISTIC REGRESSION BỀN HƠN

## So sánh trên cùng H03 test

| Chỉ số | Real-only LR | Enriched LR | Cải thiện |
|---|---:|---:|---:|
| ROC-AUC | 0.8985 | **0.9225** | **+0.0240** |
| Recall | 0.8054 | **0.8536** | **+0.0482** |
| Worst Recall | 0.6786 | **0.7500** | **+0.0714** |
| F1 | 0.8105 | **0.8315** | **+0.0210** |
| Brier | 0.1268 | **0.1122** | **−0.0146** |
| False negatives | 5.45 | **4.10** | **−1.35** |

### Kết luận

Hospital enrichment giúp Logistic Regression:

- Giảm phụ thuộc vào một feature riêng lẻ.
- Quen với missing và invalid code.
- Giảm trung bình 1.35 ca false negative.
- Cho xác suất dự đoán ổn định hơn.

---

# SLIDE 7 — ENRICHMENT KHÔNG GIÚP LIGHTGBM

## Synthetic/noisy data ảnh hưởng khác nhau theo thuật toán

| Chỉ số H03 | Real-only LightGBM | Enriched LightGBM |
|---|---:|---:|
| ROC-AUC | **0.8915** | 0.8497 |
| Recall | **0.7179** | 0.7125 |
| Worst Recall | **0.5714** | 0.5357 |
| F1 | **0.7674** | 0.7400 |
| Brier | **0.1446** | 0.1576 |
| False negatives | **7.90** | 8.05 |

### Nguyên nhân có thể

- Synthetic chiếm hơn 90% enriched training set.
- Tree split học missing/noise pattern như shortcut signal.
- Outlier và rounding tạo các ngưỡng nhân tạo.
- Training noise không hoàn toàn đại diện cho test noise.

> Cùng một augmentation strategy không phù hợp cho mọi model.

---

# SLIDE 8 — KẾT QUẢ TỔNG HỢP QUA BA BỆNH VIỆN

## Enriched Logistic Regression robust nhất

| Train set + Model | Hospital AUC | Worst AUC | Recall | Worst Recall | Brier | FN |
|---|---:|---:|---:|---:|---:|---:|
| **Enriched + Logistic Regression** | **0.9388** | **0.9225** | **0.8798** | **0.7500** | **0.0997** | **3.37** |
| Real + Logistic Regression | 0.9266 | 0.8985 | 0.8708 | 0.6786 | 0.1087 | 3.62 |
| Real + LightGBM | 0.9247 | 0.8915 | 0.8423 | 0.5714 | 0.1154 | 4.42 |
| Enriched + LightGBM | 0.8833 | 0.8497 | 0.7857 | 0.5357 | 0.1313 | 6.00 |

### Phát hiện chính

```text
Hospital enrichment
→ cải thiện Logistic Regression
→ nhưng làm LightGBM suy giảm
```

---

# SLIDE 9 — MODEL ĐỀ XUẤT

## Chọn model theo điều kiện dữ liệu

### Dữ liệu sạch và được kiểm soát

```text
real-only + LightGBM
```

- Clean ROC-AUC: 0.9600
- Clean Recall: 0.9643
- False negative: 1

### Dữ liệu hospital-like

```text
enriched_hospital_3000 + Logistic Regression
```

- Hospital ROC-AUC: 0.9388
- Worst AUC: 0.9225
- Hospital Recall: 0.8798
- Worst Recall: 0.7500

### Đề xuất triển khai ban đầu

> Dùng enriched Logistic Regression làm robust deployment candidate; giữ real-only LightGBM làm performance benchmark/challenger.

---

# SLIDE 10 — OPS FLOW ĐỀ XUẤT

## Không chỉ trả prediction — phải báo chất lượng dữ liệu

```text
Patient input
      ↓
Schema validation
      ↓
Missing / outlier / invalid-code detection
      ↓
Imputation + encoding
      ↓
Logistic Regression
      ↓
Probability + data-quality warning
```

### Ví dụ response

```json
{
  "prediction": 1,
  "probability": 0.84,
  "data_quality": "low",
  "missing_features": ["ca", "thal"],
  "warning": "Prediction uses imputed clinical values"
}
```

---

# SLIDE 11 — HẠN CHẾ

## Chưa thể xem là bằng chứng lâm sàng

- H03 là cấu hình giả định, chưa dựa trên thống kê bệnh viện thật.
- Locked test chỉ khoảng 61 bệnh nhân.
- 3.000 dòng enriched vẫn được sinh từ khoảng 242 real train rows.
- Nhiều corruption seeds không làm tăng số bệnh nhân độc lập.
- Chưa có bác sĩ xác nhận clinical bounds và noise rules.
- Chưa thực hiện external validation.

> Kết quả là empirical evidence trong phạm vi Cleveland và simulation hiện tại.

---

# SLIDE 12 — KẾT LUẬN VÀ BƯỚC TIẾP THEO

## Kết luận

1. Dữ liệu hospital-like làm model clean suy giảm rõ rệt.
2. LightGBM có hiệu năng sạch tốt nhưng nhạy với missing và shift.
3. Enrichment giúp Logistic Regression robust hơn.
4. x3.000 synthetic không tự động giúp mọi thuật toán.
5. Model triển khai cần trả thêm data-quality warning.

## Bước tiếp theo

- Ablation từng loại nhiễu.
- Thử augmentation ratio 25%, 50%, 100%, 200%.
- Tune threshold để giảm false negative.
- Calibration và SHAP analysis.
- Thay giả định H03 bằng thống kê bệnh viện thật.
- External validation trên nguồn dữ liệu độc lập.

### Thông điệp cuối

> Mục tiêu của enrichment không phải làm dataset trông lớn hơn, mà giúp model ít suy giảm hơn khi gặp dữ liệu bệnh viện không hoàn hảo.
