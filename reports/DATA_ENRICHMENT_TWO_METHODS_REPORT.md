# BÁO CÁO Ý TƯỞNG LÀM GIÀU DỮ LIỆU BỆNH TIM

## So sánh dữ liệu thật đa trung tâm và dữ liệu synthetic hospital-enriched

## 1. Tóm tắt điều hành

Nghiên cứu bắt đầu từ Cleveland Heart Disease Dataset gồm 303 bệnh nhân và 13 feature. Hai hướng làm giàu dữ liệu đã được thực hiện:

1. **Phương pháp 1 — Dữ liệu thật đa trung tâm:** hợp nhất Cleveland, Hungarian, Switzerland và Long Beach VA thành 920 hồ sơ bệnh nhân thật; giữ đủ 13 feature và xử lý missing bằng imputation trong Machine Learning Pipeline.
2. **Phương pháp 2 — Dữ liệu synthetic hospital-enriched:** fit bộ sinh dữ liệu trên real training set, tạo training set 3.000 dòng và thêm missing, outlier, rounding, categorical error cùng population shift để mô phỏng dữ liệu bệnh viện không hoàn hảo.

Kết quả chính:

- Dữ liệu thật đa trung tâm cung cấp thêm bệnh nhân độc lập và cho phép kiểm tra khả năng tổng quát hóa sang một bệnh viện chưa từng thấy.
- Logistic Regression đạt ROC-AUC trung bình khoảng **0.7978** và worst-site AUC **0.7005** trong đánh giá Leave-One-Center-Out; cao hơn LightGBM lần lượt khoảng **0.7772** và **0.6584**.
- Trên hospital stress-test mô phỏng, training bằng 3.000 dòng enriched giúp Logistic Regression tăng AUC từ **0.8985 lên 0.9225**, tăng Recall từ **0.8054 lên 0.8536** và giảm false negatives từ **5.45 xuống 4.10**.
- Cùng bộ enriched data lại làm LightGBM giảm AUC từ **0.8915 xuống 0.8497**.
- Synthetic data có ích như augmentation và robustness training, nhưng không thay thế bệnh nhân thật và không tự động cải thiện mọi thuật toán.

> Kết luận đề xuất: dùng dữ liệu thật đa trung tâm làm nền tảng đánh giá chính; dùng synthetic data như augmentation có kiểm soát bên trong từng training fold. Logistic Regression hiện là robustness candidate; LightGBM tiếp tục là clean-data performance benchmark.

---

## 2. Bài toán nghiên cứu

Dataset Cleveland ban đầu có những hạn chế:

| Thuộc tính | Giá trị |
|---|---:|
| Bệnh nhân thật | 303 |
| Feature đầu vào | 13 |
| Không bệnh | 164 |
| Có bệnh | 139 |
| Missing cells | 6 |

Dataset nhỏ và gần như sạch có thể làm kết quả random split quá lạc quan. Khi triển khai thực tế, bệnh viện có thể khác nhau về:

- Dân số bệnh nhân và tỷ lệ mắc bệnh.
- Feature được thu thập.
- Mức độ missing.
- Thiết bị, quy trình đo và cách nhập dữ liệu.
- Outlier, sai mã categorical và giá trị bị làm tròn.

Câu hỏi nghiên cứu tổng quát:

> Làm giàu dữ liệu bằng bệnh nhân thật đa trung tâm và bằng dữ liệu synthetic ảnh hưởng thế nào đến hiệu năng và độ bền của mô hình dự đoán bệnh tim?

---

## 3. Phương pháp 1 — Hợp nhất dữ liệu thật đa trung tâm

### 3.1. Nguồn dữ liệu

Bốn file processed của UCI được hợp nhất:

| Cohort | Số bệnh nhân |
|---|---:|
| Cleveland | 303 |
| Hungarian | 294 |
| Switzerland | 123 |
| Long Beach VA | 200 |
| **Tổng cộng** | **920** |

Schema thống nhất:

```text
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal, num
```

- 13 cột đầu là feature.
- `num` từ 0 đến 4 là target gốc.
- Target binary: `target = 1` nếu `num > 0`, ngược lại bằng 0.
- `site` được thêm để audit và chia dữ liệu theo bệnh viện, không dùng làm model feature.

### 3.2. Chất lượng dữ liệu thật

```text
920 bệnh nhân × 13 feature = 11.960 ô dữ liệu
```

| Chỉ số | Giá trị |
|---|---:|
| Missing cells | 1.759 |
| Missing rate | 14,71% |
| Missing rate sau khi xem `chol=0`, `trestbps=0` là sentinel | 16,15% |
| Numerical outlier theo IQR, sau xử lý sentinel | 68 ô |
| Outlier rate trên 5 numerical feature | 1,48% |
| Bệnh nhân có ít nhất một numerical outlier | 64/920 — 6,96% |
| Categorical code ngoài miền hợp lệ | 0% |

Missing không phân bố đều:

| Feature | Missing | Tỷ lệ |
|---|---:|---:|
| `ca` | 611 | 66,41% |
| `thal` | 486 | 52,83% |
| `slope` | 309 | 33,59% |
| `fbs` | 90 | 9,78% |
| `oldpeak` | 62 | 6,74% |
| `trestbps` | 59 | 6,41% |
| `thalach` | 55 | 5,98% |
| `exang` | 55 | 5,98% |
| `chol` | 30 | 3,26% |

Missing cũng khác nhau mạnh theo bệnh viện:

| Site | Missing rate | Dòng có ít nhất một missing |
|---|---:|---:|
| Cleveland | 0,15% | 1,98% |
| Hungarian | 20,46% | 99,66% |
| Switzerland | 17,07% | 100% |
| VA | 26,85% | 99,50% |

Điều này cho thấy dữ liệu thật có **site-dependent missingness**, không phải missing ngẫu nhiên đồng đều.

### 3.3. Xử lý dữ liệu

Giữ đủ 13 feature và đặt preprocessing trong Pipeline:

```text
Numerical
→ median imputation
→ missing indicator
→ StandardScaler

Categorical
→ most-frequent imputation
→ missing indicator
→ OneHotEncoder(handle_unknown="ignore")
```

Nguyên tắc chống leakage:

- Không impute toàn bộ dataset trước khi chia dữ liệu.
- Imputer và encoder chỉ fit trên training hospitals.
- Không đưa `site` vào model.
- Bệnh viện test bị khóa hoàn toàn.

### 3.4. Thiết kế đánh giá Leave-One-Center-Out

```text
Fold 1: Train Hungarian + Switzerland + VA → Test Cleveland
Fold 2: Train Cleveland + Switzerland + VA → Test Hungarian
Fold 3: Train Cleveland + Hungarian + VA → Test Switzerland
Fold 4: Train Cleveland + Hungarian + Switzerland → Test VA
```

Đây là external-site evaluation: model phải dự đoán tại một trung tâm chưa xuất hiện trong training.

### 3.5. Kết quả theo bệnh viện

| Model | Test site | AUC | Recall | Specificity | Brier | False negatives |
|---|---|---:|---:|---:|---:|---:|
| LightGBM | Cleveland | 0.8707 | 0.7698 | 0.8171 | 0.1433 | 32 |
| LightGBM | Hungarian | 0.8557 | 0.7358 | 0.8138 | 0.1501 | 28 |
| LightGBM | Switzerland | 0.7239 | 0.7478 | 0.6250 | 0.1860 | 29 |
| LightGBM | VA | 0.6584 | 0.6779 | 0.5686 | 0.2423 | 48 |
| Logistic Regression | Cleveland | 0.8694 | 0.6043 | 0.9329 | 0.1572 | 55 |
| Logistic Regression | Hungarian | 0.8997 | 0.7547 | 0.8883 | 0.1187 | 26 |
| Logistic Regression | Switzerland | 0.7217 | 0.7739 | 0.6250 | 0.1774 | 26 |
| Logistic Regression | VA | 0.7005 | 0.8658 | 0.3137 | 0.1832 | 20* |

\* False negatives tại VA được suy ra từ 149 ca dương và Recall 0.8658; cần đối chiếu output đầy đủ của notebook trước khi dùng như số liệu chính thức trong bài báo.

### 3.6. Tổng hợp kết quả đa trung tâm

| Model | AUC trung bình | Worst-site AUC | Worst-site Recall | Worst-site Specificity |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.7978** | **0.7005** | 0.6043 | 0.3137 |
| LightGBM | 0.7772 | 0.6584 | **0.6779** | **0.5686** |

Nhận xét:

- Logistic Regression tổng quát hóa tốt hơn về mean AUC và worst-site AUC.
- LightGBM giữ cân bằng Recall–Specificity tốt hơn trong một số site.
- Threshold 0.5 của Logistic Regression không ổn định giữa các bệnh viện: Recall thấp tại Cleveland nhưng Specificity rất thấp tại VA.
- VA là site khó nhất, có missing rate cao nhất và distribution khác training hospitals.

---

## 4. Phương pháp 2 — Làm giàu thành 3.000 dòng synthetic hospital-like

### 4.1. Mục tiêu

Phương pháp này không nhằm biến 242 bệnh nhân thành 3.000 bệnh nhân độc lập. Mục tiêu là:

- Tăng độ đa dạng của training input.
- Cho model tiếp xúc với missing và noise trước khi triển khai.
- Stress-test mức suy giảm trên các hospital profile.
- Nghiên cứu thuật toán nào hưởng lợi từ robustness training.

### 4.2. Thiết kế chống leakage

```text
303 bệnh nhân Cleveland
        ↓
Stratified train/test split
        │
        ├── Real train khoảng 242 dòng
        │        ↓
        │   Fit Gaussian Copula / generator
        │        ↓
        │   Sinh và làm nhiễu training data
        │        ↓
        │   Enriched training set: 3.000 dòng
        │
        └── Locked real test khoảng 61 dòng
                 ↓
            Clean test + hospital stress-test
```

- Generator chỉ fit trên real train.
- Locked test không tham gia sinh dữ liệu.
- Preprocessing chỉ fit trên training set.
- Synthetic test không được dùng để khẳng định hiệu năng lâm sàng.

### 4.3. Hospital corruption profile H03

| Loại nhiễu | Cấu hình mô phỏng |
|---|---:|
| Missing chung | 15% |
| Numerical outlier | 4% |
| Sai categorical code | 2% |
| Giá trị bị làm tròn | 20% |
| Khả năng thiếu riêng `ca/thal` | 35% |
| Population shift | Trẻ hơn 3 tuổi |

Đánh giá được lặp bằng:

```text
5 model seeds × 20 corruption seeds
= 100 lượt đánh giá mỗi cấu hình
```

### 4.4. Mức độ tương đồng với dữ liệu thật đa trung tâm

| Vấn đề | UCI multicenter thật | Synthetic 3.000 |
|---|---:|---:|
| Missing chung | 14,71%; 16,15% nếu xử lý zero sentinel | Cấu hình 15% |
| Missing `ca` | 66,41% | Thêm xác suất 35% |
| Missing `thal` | 52,83% | Thêm xác suất 35% |
| Numerical outlier | 1,48% số ô numerical theo IQR | Chèn 4% |
| Invalid categorical code | 0% được phát hiện | Chèn 2% |
| Rounding corruption | Không xác định được | Chèn 20% |
| Population shift | Có thật giữa bốn site | Mô phỏng bằng rule |

Nhận xét:

- Mức missing chung 15% khá gần dữ liệu thật.
- Simulation chưa tái hiện đúng cấu trúc missing: thực tế thiếu tập trung vào `ca`, `thal`, `slope` và khác nhau theo site.
- Outlier, invalid code và rounding trong synthetic chủ yếu là stress condition, không phải ước lượng tỷ lệ lỗi thật tại UCI.

### 4.5. Kết quả trên H03 stress-test

| Training set | Model | H03 AUC | Recall | Worst Recall | F1 | Brier | FN trung bình |
|---|---|---:|---:|---:|---:|---:|---:|
| **Enriched 3.000** | **Logistic Regression** | **0.9225** | **0.8536** | **0.7500** | **0.8315** | **0.1122** | **4.10** |
| Real-only | Logistic Regression | 0.8985 | 0.8054 | 0.6786 | 0.8105 | 0.1268 | 5.45 |
| Real-only | LightGBM | 0.8915 | 0.7179 | 0.5714 | 0.7674 | 0.1446 | 7.90 |
| Enriched 3.000 | LightGBM | 0.8497 | 0.7125 | 0.5357 | 0.7400 | 0.1576 | 8.05 |

### 4.6. Ảnh hưởng đối với Logistic Regression

| Chỉ số H03 | Real-only LR | Enriched LR | Thay đổi |
|---|---:|---:|---:|
| ROC-AUC | 0.8985 | **0.9225** | **+0.0240** |
| Recall | 0.8054 | **0.8536** | **+0.0482** |
| Worst Recall | 0.6786 | **0.7500** | **+0.0714** |
| F1 | 0.8105 | **0.8315** | **+0.0210** |
| Brier | 0.1268 | **0.1122** | **−0.0146** |
| False negatives | 5.45 | **4.10** | **−1.35** |

Hospital enrichment giúp Logistic Regression bền hơn trước đúng corruption profile đã dùng trong nghiên cứu.

### 4.7. Ảnh hưởng đối với LightGBM

| Chỉ số H03 | Real-only LightGBM | Enriched LightGBM | Thay đổi |
|---|---:|---:|---:|
| ROC-AUC | **0.8915** | 0.8497 | −0.0418 |
| Recall | **0.7179** | 0.7125 | −0.0054 |
| Worst Recall | **0.5714** | 0.5357 | −0.0357 |
| F1 | **0.7674** | 0.7400 | −0.0274 |
| Brier | **0.1446** | 0.1576 | +0.0130 |
| False negatives | **7.90** | 8.05 | +0.15 |

Các nguyên nhân có thể:

- Synthetic chiếm hơn 90% enriched training set.
- Tree split học artifact của generator và noise như shortcut signal.
- Outlier và rounding tạo các ngưỡng nhân tạo.
- Training corruption không hoàn toàn đại diện cho test distribution.

---

## 5. So sánh hai phương pháp làm giàu

| Tiêu chí | 920 dòng real multicenter | 3.000 dòng synthetic enriched |
|---|---|---|
| Bản chất | Bệnh nhân thật độc lập | Mẫu sinh từ real training set |
| Số nguồn | 4 bệnh viện/cohort | Một nguồn ban đầu và các profile mô phỏng |
| Missing | Missing thật, phụ thuộc feature/site | Missing do cấu hình |
| Population shift | Quan sát được giữa site thật | Mô phỏng bằng rule |
| Giá trị khoa học chính | External-site generalization | Augmentation và robustness stress-test |
| Rủi ro leakage | Impute trước split; trộn site | Generator fit cả dataset; synthetic hóa test |
| Rủi ro bias | Bias khác nhau giữa cohort | Khuếch đại bias và artifact của generator |
| Kết quả nổi bật | LR có mean/worst AUC tốt hơn LightGBM | Enrichment giúp LR nhưng làm hại LightGBM |
| Có thể thay thế external validation? | Có giá trị như external-site test nội bộ | Không |

### 5.1. Không so sánh trực tiếp hai giá trị AUC như cùng một benchmark

Không được kết luận rằng:

```text
H03 AUC 0.9225 > LOCO AUC 0.7978
→ synthetic tốt hơn dữ liệu thật
```

Hai kết quả đến từ hai protocol khác nhau:

- `0.7978`: trung bình qua bốn bệnh viện thật bị held-out.
- `0.9225`: H03 stress-test được tạo từ locked Cleveland test theo corruption profile mô phỏng.

LOCO khó hơn vì chứa thay đổi thật về population, prevalence, measurement và missingness. So sánh hợp lệ phải huấn luyện các cấu hình khác nhau rồi test trên **cùng một held-out real hospital**.

---

## 6. Kết luận

### 6.1. Kết luận về dữ liệu

1. Việc hợp nhất bốn cohort tạo ra 920 bệnh nhân thật và có giá trị khoa học cao hơn việc chỉ nhân số dòng synthetic.
2. Dữ liệu multicenter chứng minh Cleveland sạch không đại diện cho Hungarian, Switzerland và VA.
3. Missing thật có cấu trúc theo site và feature; đặc biệt nghiêm trọng ở `ca`, `thal`, `slope`.
4. Bộ 3.000 dòng hữu ích để mô phỏng lỗi vận hành, nhưng không tạo thêm 3.000 bệnh nhân độc lập.

### 6.2. Kết luận về model

1. LightGBM là performance benchmark tốt trên Cleveland sạch nhưng suy giảm khi gặp hospital shift.
2. Logistic Regression có mean AUC và worst-site AUC tốt hơn trong đánh giá đa trung tâm.
3. Enriched training cải thiện Logistic Regression trên H03 nhưng gây hại cho LightGBM.
4. Không có augmentation strategy duy nhất phù hợp cho mọi model.
5. Threshold 0.5 chưa ổn định giữa các bệnh viện; cần tune threshold và calibration không leakage.

### 6.3. Model đề xuất tại thời điểm hiện tại

```text
Robustness candidate:
Logistic Regression
+ fold-safe imputation
+ missing indicators
+ threshold tuning
+ calibration
```

```text
Performance challenger:
LightGBM train trên dữ liệu thật
```

Chưa nên gọi bất kỳ model nào là SOTA lâm sàng vì chưa có external validation trên dữ liệu bệnh viện mới ngoài bốn cohort UCI.

---

## 7. Hướng nghiên cứu tiếp theo

### 7.1. Thí nghiệm quan trọng nhất

Trong mỗi Leave-One-Center-Out fold:

```text
Ba bệnh viện training
        ├── Real only
        ├── Real + Gaussian 25%, 50%, 100%
        ├── Real + CTGAN 25%, 50%, 100%
        └── Real + site-aware corruption
                    ↓
        Test trên bệnh viện thật thứ tư
```

Generator chỉ được fit trên training hospitals. Held-out hospital không được dùng để:

- Fit generator.
- Fit imputer.
- Tune hyperparameter.
- Tune threshold.
- Chọn augmentation ratio.

### 7.2. Site-aware synthetic generation

Thay cấu hình missing đồng đều bằng profile học từ dữ liệu thật:

```text
Cleveland-like     ~0,15% missing
Hungarian-like    ~20,46% missing
Switzerland-like  ~17,07% missing
VA-like           ~26,85% missing
```

Missing probability cần phụ thuộc vào feature, đặc biệt:

```text
ca      66,41%
thal    52,83%
slope   33,59%
```

### 7.3. Feature availability ablation

So sánh ba bộ feature:

1. Đủ 13 feature.
2. Bỏ `ca`, `thal`.
3. Chỉ giữ common features có missing thấp ở cả bốn site.

Mục tiêu là tìm cân bằng giữa predictive performance và khả năng thu thập feature nhất quán tại nhiều bệnh viện.

### 7.4. Threshold và calibration

- Chọn threshold bằng internal validation trên training hospitals.
- So sánh threshold 0.5, threshold tối ưu F1 và threshold đạt Recall mục tiêu.
- Báo cáo Specificity, false positives và false negatives tương ứng.
- Thử Platt scaling hoặc isotonic calibration.

### 7.5. Tiêu chí chọn model cuối cùng

Không chọn model chỉ bằng AUC trung bình. Model cuối cần được đánh giá theo:

- Mean và worst-site ROC-AUC.
- Mean và worst-site Recall.
- Specificity.
- False negatives.
- Brier score và calibration.
- Độ ổn định qua seed.
- Thời gian inference và kích thước model.
- Khả năng giải thích.

---

## 8. Thông điệp cuối

> Làm giàu dữ liệu không có nghĩa là chỉ tăng từ 303 lên 3.000 dòng. Giá trị quan trọng nhất là tăng độ đa dạng của bệnh nhân thật, mô phỏng đúng cấu trúc dữ liệu bệnh viện và chứng minh model ít suy giảm hơn trên một bệnh viện thật chưa từng thấy.

Hai phương pháp không cạnh tranh để loại bỏ nhau:

```text
Real multicenter data
→ nền tảng huấn luyện và đánh giá tính tổng quát

Synthetic hospital enrichment
→ công cụ augmentation và robustness training
```

Hướng nghiên cứu hợp lý nhất là kết hợp cả hai, nhưng mọi cải thiện từ synthetic data phải được xác nhận trên held-out real hospital.
