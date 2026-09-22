# BÁO CÁO ẢNH HƯỞNG CỦA DỮ LIỆU BỆNH VIỆN MÔ PHỎNG ĐẾN MÔ HÌNH

## 1. Tóm tắt điều hành

Nghiên cứu mô phỏng môi trường phòng khám ngoại trú (`H03_outpatient_clinic`) với dữ liệu đầu vào thiếu và nhiễu hơn dữ liệu Cleveland gốc. Kịch bản gồm 15% missing, 4% outlier, 2% categorical code không hợp lệ, 20% giá trị bị làm tròn và 35% khả năng thiếu riêng hai feature `ca`/`thal`. Nhóm bệnh nhân cũng được mô phỏng trẻ hơn trung tâm tim mạch.

Kết quả chính:

- Dữ liệu H03 làm giảm hiệu năng của tất cả mô hình so với clean test.
- `real-only + LightGBM` giảm Recall mạnh nhất: từ 0.9643 xuống 0.7179.
- Hospital enrichment không cải thiện LightGBM; AUC và Brier đều xấu hơn.
- `enriched_hospital_3000 + Logistic Regression` chịu nhiễu tốt nhất: H03 AUC 0.9225, Recall 0.8536 và Brier 0.1122.
- So với Logistic Regression chỉ train trên dữ liệu thật, enrichment làm H03 AUC tăng 0.0240, Recall tăng 0.0482 và giảm trung bình 1.35 ca false negative.

Kết luận: hospital-like enrichment có ích cho Logistic Regression nhưng gây hại cho LightGBM. Logistic Regression trên enriched dataset được đề xuất làm robust candidate trong điều kiện dữ liệu đầu vào giống H03.

---

## 2. Mục tiêu

Mục tiêu thí nghiệm là trả lời ba câu hỏi:

1. Missing, outlier, sai categorical code, làm tròn và population shift ảnh hưởng thế nào đến model cũ?
2. Retrain model bằng 3.000 dòng hospital-enriched có giảm suy giảm hiệu năng không?
3. Logistic Regression hay LightGBM phù hợp hơn khi vận hành với dữ liệu không hoàn hảo?

---

## 3. Dữ liệu và thiết kế thí nghiệm

### 3.1. Dữ liệu nguồn

| Thành phần | Quy mô |
|---|---:|
| Cleveland gốc | 303 bệnh nhân |
| Real train | Khoảng 242 dòng |
| Locked real test | Khoảng 61 dòng |
| Enriched hospital train | 3.000 dòng |

Enriched dataset gồm khoảng 242 dòng real train và 2.758 dòng Gaussian synthetic. Locked real test được tách trước khi fit synthesizer và không tham gia sinh dữ liệu.

### 3.2. Training configurations

| Train set | Model |
|---|---|
| `real_only` | Logistic Regression |
| `real_only` | LightGBM |
| `enriched_hospital_3000` | Logistic Regression |
| `enriched_hospital_3000` | LightGBM |

### 3.3. Quy mô đánh giá

- 5 model seeds: `42, 52, 62, 72, 82`.
- 20 hospital corruption seeds: `100–119`.
- H03 có 100 lượt đánh giá cho mỗi cấu hình model/train set.
- Clean test dùng cùng locked real test để làm mốc so sánh.

---

## 4. Cấu hình mô phỏng H03 Outpatient Clinic

```python
H03_OUTPATIENT_CLINIC = {
    "missing": 0.15,
    "outlier": 0.04,
    "invalid_code": 0.02,
    "rounding": 0.20,
    "age_shift": -3,
    "bp_shift": 0,
    "shift_probability": 0.60,
    "ca_thal_missing": 0.35
}
```

### 4.1. Missing 15%

Mỗi ô trong 13 feature có 15% xác suất bị chuyển thành missing. Missing được áp dụng độc lập trong mô phỏng, sau đó pipeline sử dụng median/mode imputation và missing indicators.

Tác động kỳ vọng:

- Giảm lượng thông tin trực tiếp cho model.
- Làm probability kém chắc chắn hơn.
- Feature phụ thuộc nhiều vào `ca`/`thal` hoặc numerical measurement bị suy yếu.

### 4.2. Outlier 4%

4% giá trị numerical được nhân ngẫu nhiên với một hệ số trong:

```text
0.1, 0.5, 1.5 hoặc 10.0
```

Outlier mô phỏng lỗi nhập liệu, sai đơn vị hoặc thiết bị đo bất thường.

Tác động kỳ vọng:

- MinMaxScaler có thể đưa giá trị ra ngoài phạm vi train.
- Tree model có thể đi vào nhánh hiếm hoặc nhánh không được học ổn định.
- Logistic Regression bị ảnh hưởng nhưng quan hệ tuyến tính và regularization có thể làm suy giảm ít cực đoan hơn.

### 4.3. Sai categorical code 2%

2% giá trị categorical được thay bằng mã `99`. `OneHotEncoder(handle_unknown="ignore")` không làm pipeline lỗi, nhưng toàn bộ one-hot value của feature đó có thể bằng 0.

Tác động kỳ vọng:

- Model vẫn chạy được.
- Thông tin của feature bị mất tại dòng đó.
- API cần cảnh báo `invalid_code_features` thay vì âm thầm coi mã sai là hợp lệ.

### 4.4. Làm tròn 20%

| Feature | Bước làm tròn |
|---|---:|
| `trestbps` | 5 |
| `chol` | 10 |
| `thalach` | 5 |
| `oldpeak` | 0.5 |

Làm tròn mô phỏng khác biệt thiết bị, giao diện nhập liệu hoặc quy trình ghi nhận.

Tác động kỳ vọng:

- Giảm độ phân giải của numerical feature.
- Nhiều bệnh nhân có cùng giá trị hơn.
- Các ngưỡng chia cây có thể trở nên kém ổn định.

### 4.5. Thiếu riêng `ca`/`thal` 35%

Ngoài missing chung 15%, hai feature `ca` và `thal` có thêm 35% xác suất bị thiếu.

Đây là mô phỏng phòng khám ngoại trú chưa thực hiện đầy đủ xét nghiệm chuyên sâu. `ca` và `thal` là các feature có tín hiệu đáng kể trong Cleveland, nên việc thiếu chúng tạo ra distribution shift mạnh.

### 4.6. Bệnh nhân trẻ hơn trung tâm tim mạch

60% dòng H03 được áp dụng:

```python
age = age - 3
```

Trong khi H01 Cardiac Center sử dụng `age_shift=+5` với xác suất 70%. Do đó H03 được thiết kế có population profile trẻ hơn trung tâm tim mạch.

Lưu ý: đây là giả định nghiên cứu, chưa phải thống kê thực tế của một bệnh viện cụ thể.

---

## 5. Kết quả trên H03

| Train set | Model | AUC mean | AUC std | Recall mean | Recall min | F1 | Brier | FN mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Enriched 3.000 | **Logistic Regression** | **0.9225** | 0.0247 | **0.8536** | **0.7500** | **0.8315** | **0.1122** | **4.10** |
| Real-only | Logistic Regression | 0.8985 | 0.0333 | 0.8054 | 0.6786 | 0.8105 | 0.1268 | 5.45 |
| Real-only | LightGBM | 0.8915 | 0.0256 | 0.7179 | 0.5714 | 0.7674 | 0.1446 | 7.90 |
| Enriched 3.000 | LightGBM | 0.8497 | 0.0296 | 0.7125 | 0.5357 | 0.7400 | 0.1576 | 8.05 |

Enriched Logistic Regression đứng đầu ở mọi chỉ số quan trọng của H03.

---

## 6. Dữ liệu H03 làm model suy giảm thế nào?

### 6.1. Real-only Logistic Regression

| Chỉ số | Clean | H03 | Mức thay đổi |
|---|---:|---:|---:|
| ROC-AUC | 0.9567 | 0.8985 | −0.0582 |
| Recall | 0.9286 | 0.8054 | −0.1232 |
| F1 | 0.8814 | 0.8105 | −0.0709 |
| Brier | 0.0884 | 0.1268 | +0.0384 |
| False negatives | 2.00 | 5.45 | +3.45 |

Model vẫn hoạt động nhưng bỏ sót nhiều bệnh nhân hơn khi `ca/thal` và các measurement bị thiếu hoặc sai.

### 6.2. Enriched Logistic Regression

| Chỉ số | Clean | H03 | Mức thay đổi |
|---|---:|---:|---:|
| ROC-AUC | 0.9556 | 0.9225 | −0.0331 |
| Recall | 0.8929 | 0.8536 | −0.0393 |
| F1 | 0.8621 | 0.8315 | −0.0306 |
| Brier | 0.0868 | 0.1122 | +0.0254 |
| False negatives | 3.00 | 4.10 | +1.10 |

Hospital enrichment làm mức suy giảm nhỏ hơn đáng kể. Model đã tiếp xúc với missing, outlier, rounding và invalid code trong training nên ít bất ngờ hơn khi gặp H03.

### 6.3. Real-only LightGBM

| Chỉ số | Clean | H03 | Mức thay đổi |
|---|---:|---:|---:|
| ROC-AUC | 0.9600 | 0.8915 | −0.0685 |
| Recall | 0.9643 | 0.7179 | −0.2464 |
| F1 | 0.9153 | 0.7674 | −0.1479 |
| Brier | 0.0806 | 0.1446 | +0.0640 |
| False negatives | 1.00 | 7.90 | +6.90 |

LightGBM là model tốt nhất trên clean data nhưng suy giảm rất mạnh khi input distribution thay đổi.

### 6.4. Enriched LightGBM

| Chỉ số | Clean | H03 | Mức thay đổi |
|---|---:|---:|---:|
| ROC-AUC | 0.9134 | 0.8497 | −0.0637 |
| Recall | 0.8929 | 0.7125 | −0.1804 |
| F1 | 0.8772 | 0.7400 | −0.1372 |
| Brier | 0.1028 | 0.1576 | +0.0548 |
| False negatives | 3.00 | 8.05 | +5.05 |

Enrichment không cứu được LightGBM. Synthetic rows chiếm hơn 90% enriched train và chứa các noise pattern do mô phỏng; tree boosting có khả năng học các artifact này như split signal.

---

## 7. Hospital enrichment cải thiện Logistic Regression bao nhiêu?

So sánh trên cùng H03:

| Chỉ số | Real-only LR | Enriched LR | Lợi ích enrichment |
|---|---:|---:|---:|
| ROC-AUC | 0.8985 | **0.9225** | **+0.0240** |
| Recall | 0.8054 | **0.8536** | **+0.0482** |
| Worst Recall | 0.6786 | **0.7500** | **+0.0714** |
| F1 | 0.8105 | **0.8315** | **+0.0210** |
| Brier | 0.1268 | **0.1122** | **−0.0146** |
| False negatives | 5.45 | **4.10** | **−1.35** |

Hospital enrichment làm Logistic Regression:

- Xếp hạng nguy cơ tốt hơn.
- Phát hiện thêm bệnh nhân có bệnh.
- Giảm trung bình 1.35 false negative trên mỗi H03 test run.
- Giảm biến thiên AUC từ 0.0333 xuống 0.0247.
- Cho xác suất tốt hơn, thể hiện qua Brier thấp hơn.

---

## 8. Hospital enrichment ảnh hưởng LightGBM thế nào?

So sánh trên cùng H03:

| Chỉ số | Real-only LightGBM | Enriched LightGBM | Thay đổi |
|---|---:|---:|---:|
| ROC-AUC | **0.8915** | 0.8497 | −0.0418 |
| Recall | **0.7179** | 0.7125 | −0.0054 |
| Worst Recall | **0.5714** | 0.5357 | −0.0357 |
| F1 | **0.7674** | 0.7400 | −0.0274 |
| Brier | **0.1446** | 0.1576 | +0.0130 |
| False negatives | **7.90** | 8.05 | +0.15 |

Đối với LightGBM, enriched data làm tất cả chỉ số chính xấu đi. Đây là bằng chứng rằng cùng một augmentation strategy không phù hợp cho mọi thuật toán.

---

## 9. Giải thích nguyên nhân

### Logistic Regression hưởng lợi

- Mối quan hệ tuyến tính và regularization làm model ít nhạy với artifact nhỏ.
- Missing indicators giúp model nhận biết trạng thái thiếu dữ liệu.
- One-hot `handle_unknown="ignore"` ngăn invalid code làm pipeline lỗi.
- Việc tiếp xúc với nhiều missing/noisy examples giúp decision boundary ít phụ thuộc vào một feature cụ thể.

### LightGBM suy giảm

- Tree split có thể học những ngưỡng outlier hoặc rounding do mô phỏng.
- Synthetic chiếm tỷ trọng quá cao so với 242 dòng thật.
- Missing pattern giả định có thể trở thành shortcut signal.
- Khi test noise khác chính xác training noise, các split synthetic không generalize.

Đây là suy luận từ kết quả thí nghiệm; cần feature importance/SHAP và ablation study để xác nhận cơ chế.

---

## 10. Đề xuất model

### Khi dữ liệu đầu vào sạch và được kiểm soát

```text
real-only + LightGBM
```

- Clean AUC: 0.9600.
- Clean Recall: 0.9643.
- Chỉ 1 false negative trên locked real test.

### Khi dữ liệu đầu vào giống phòng khám H03

```text
enriched_hospital_3000 + Logistic Regression
```

- H03 AUC: 0.9225.
- H03 Recall: 0.8536.
- Worst Recall: 0.7500.
- Brier: 0.1122.
- False negatives: 4.10.

Nếu chỉ chọn một model cho môi trường dữ liệu không đồng nhất, enriched Logistic Regression là lựa chọn robust hơn.

---

## 11. Khuyến nghị vận hành

Inference response nên trả cả dự đoán và cảnh báo chất lượng:

```json
{
  "prediction": 1,
  "probability": 0.84,
  "data_quality": "low",
  "missing_features": ["ca", "thal"],
  "outlier_features": [],
  "invalid_code_features": [],
  "warning": "Prediction uses imputed clinical values"
}
```

Các hồ sơ có chất lượng thấp không nên được xử lý giống hồ sơ đầy đủ. Có thể thiết lập quy tắc yêu cầu bổ sung xét nghiệm hoặc human review khi:

- Thiếu đồng thời `ca` và `thal`.
- Có nhiều hơn hai feature thiếu.
- Có outlier hoặc categorical code không hợp lệ.
- Xác suất nằm gần decision threshold.

---

## 12. Hạn chế

- H03 là cấu hình giả định, chưa dựa trên aggregate statistics của một phòng khám thật.
- Locked test chỉ khoảng 61 bệnh nhân.
- 20 corruption seeds tăng độ tin cậy về noise variability nhưng không tăng số bệnh nhân độc lập.
- Synthetic data không chứa kiến thức y khoa ngoài Cleveland.
- `CLINICAL_BOUNDS` và noise rates chưa được bác sĩ phê duyệt.
- Chưa thực hiện external validation.
- Chưa có ablation để tách ảnh hưởng riêng của missing, outlier, invalid code, rounding và `ca/thal` missing.

---

## 13. Công việc tiếp theo

1. Chạy ablation study từng loại nhiễu riêng biệt.
2. Thử augmentation ratio 25%, 50%, 100%, 200% thay vì để synthetic chiếm hơn 90%.
3. Tune threshold để giảm false negative.
4. Vẽ calibration curve và thực hiện probability calibration.
5. Phân tích feature importance/SHAP trước và sau enrichment.
6. Thay H03 assumptions bằng thống kê từ bệnh viện/phòng khám thực tế.
7. External validation trên một nguồn dữ liệu tim mạch độc lập.

---

## 14. Kết luận

Kịch bản H03 với 15% missing, 4% outlier, 2% sai categorical code, 20% rounding, 35% thiếu riêng `ca/thal` và population trẻ hơn đã làm giảm đáng kể hiệu năng các model huấn luyện trên clean real data. Tác động mạnh nhất xuất hiện ở Recall và false negatives.

Hospital enrichment giúp Logistic Regression chống chịu tốt hơn: AUC tăng từ 0.8985 lên 0.9225, Recall tăng từ 0.8054 lên 0.8536 và false negatives giảm từ 5.45 xuống 4.10. Ngược lại, LightGBM không hưởng lợi và có dấu hiệu học artifact từ synthetic/noisy training data.

Đề xuất hiện tại:

```text
Performance benchmark trên clean data:
real-only + LightGBM

Robust candidate trên hospital-like data:
enriched_hospital_3000 + Logistic Regression
```

Kết quả này là empirical evidence trong phạm vi Cleveland và simulation hiện tại, không phải bằng chứng về hiệu quả lâm sàng.
