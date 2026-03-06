# Project Workflow - Heart Disease Diagnosis

**AIO2025 Research Project | VietAI Learning Team**

---

## 📋 Overview

This document describes the main workflow and purpose of key files in the project.

---

## 🔄 Main Workflow

```
1. Data Preparation
   ↓
2. Model Training (Jupyter Notebook)
   ↓
3. Hyperparameter Tuning (Optuna)
   ↓
4. Save Models
   ↓
5. Deploy Web App
```

---

## 📂 Key Files & Their Roles

### 1️⃣ **Data Preparation**

#### Input Data
```
📁 data/raw/
├── raw_train.csv    # Training data (original)
├── raw_val.csv      # Validation data
└── raw_test.csv     # Test data
```

#### Processed Data
```
📁 data/processed/
├── raw_train.csv, raw_val.csv, raw_test.csv          # Raw features
├── fe_train.csv, fe_val.csv, fe_test.csv              # Feature engineering
├── dt_train.csv, dt_val.csv, dt_test.csv              # Decision tree features
├── fe_dt_train.csv, fe_dt_val.csv, fe_dt_test.csv    # Combined FE + DT
└── master_boruta_train/val/test.csv                   # Boruta selection
```

**Purpose:** Different feature engineering strategies for model comparison

---

### 2️⃣ **Model Training**

#### Main Training Notebook
```
📓 notebooks/latest.ipynb
```

**Purpose:** Train all 8 models on all 5 dataset variations

**Process:**
1. Load preprocessed data from `data/processed/`
2. Train 8 models:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors
   - Decision Tree
   - Gradient Boosting
   - Naive Bayes
   - SVM
   - Ensemble (Voting Classifier)
3. Evaluate with cross-validation
4. Save models to `models/saved_models/latest/`

**Output:**
```
📁 models/saved_models/latest/
├── best_lr_model_pipeline.pkl
├── best_rf_model_pipeline.pkl
├── best_knn_model_pipeline.pkl
├── best_dt_model_pipeline.pkl
├── best_gb_model_pipeline.pkl
├── best_nb_model_pipeline.pkl
├── best_svm_model_pipeline.pkl
├── best_ensemble_model_pipeline.pkl
└── best_models_summary.json    # Performance metrics
```

---

### 3️⃣ **Hyperparameter Tuning**

#### Script
```
📜 scripts/train_models.py
```

**Purpose:** Optimize hyperparameters using Optuna

**How to run:**
```bash
python scripts/train_models.py
```

**Process:**
1. Load training data
2. For each model (8 models):
   - Run 100 Optuna trials
   - Find best hyperparameters
   - Save to JSON
3. Total: 800 trials across all models

**Output:**
```
📁 experiments/optimized_params/
├── best_params_raw.json              # Best params for raw data
├── best_params_fe.json               # Best params for FE data
├── best_params_dt.json               # Best params for DT data
├── best_params_fe_dt.json            # Best params for FE+DT data
├── best_params_master_boruta.json    # Best params for Boruta data
└── best_params_all.json              # All combined
```

---

### 4️⃣ **Experiment Tracking**

#### Manager
```
📜 scripts/experiment_manager.py
```

**Purpose:** Track all experiments systematically

**Features:**
- Set global seed (reproducibility)
- Log all experiments
- Compare results
- Generate reports

**Output:**
```
📁 experiments/
├── experiment_log.json       # 40 experiments logged
├── logs/                     # Training logs
│   └── experiment_*.log
└── results/                  # Metrics & predictions
```

**How to use:**
```python
from scripts.experiment_manager import ExperimentManager

manager = ExperimentManager()
manager.set_global_seed(42)
manager.log_experiment(...)
df = manager.compare_experiments()
```

---

### 5️⃣ **Model Pipeline**

#### Core Module
```
📜 src/pipeline.py
```

**Purpose:** Load models and make predictions

**Key Class:** `HeartDiseasePipeline`

**Methods:**
- `load_models()` - Load 8 trained models from `models/saved_models/latest/`
- `predict(patient_data)` - Predict using all models
- `get_majority_vote()` - Ensemble prediction

**Used by:** `app/streamlit_app.py`

---

### 6️⃣ **Feature Engineering**

#### Module
```
📜 src/model_functions.py
```

**Purpose:** Feature engineering functions

**Functions:**
- `fe_basic()` - Basic feature engineering
- `fe_enhanced()` - Advanced features
- `fe_poly_only()` - Polynomial features only

**Used by:** `src/pipeline.py` when loading models

---

### 7️⃣ **Web Application**

#### Main App
```
📜 app/streamlit_app.py
```

**Purpose:** Interactive web interface for predictions

**Features:**
- **Tab 1:** Patient input & prediction (8 models + majority vote)
- **Tab 2:** Model performance analysis
- **Tab 3:** Feature importance visualization
- **Tab 4:** Experiment tracking dashboard
- **Tab 5:** Prediction history & PDF reports

**How to run:**
```bash
./scripts/run_app.sh
```

**Dependencies:**
```
📜 src/utils/app_utils.py      # Helper functions
├── PatientHistoryManager      # Save/load predictions
├── create_patient_report_pdf  # Generate PDF reports
├── create_feature_importance_plot
└── get_preset_examples        # Example patients
```

---

### 8️⃣ **Deployment**

#### Live App
```
🌐 https://heart-disease-diagnosis-vietai.streamlit.app
```

**Deployment:** Streamlit Cloud (auto-deploy from GitHub)

**Configuration:**
```
📜 requirements.txt    # Python dependencies
📜 packages.txt        # System packages (empty)
📜 .streamlit/config.toml
```

---

## 🎯 Main Use Cases

### Use Case 1: Train New Models

```bash
# 1. Prepare data (if needed)
# Already in data/processed/

# 2. Open training notebook
jupyter notebook notebooks/latest.ipynb

# 3. Run all cells
# Models saved to models/saved_models/latest/
```

### Use Case 2: Optimize Hyperparameters

```bash
# Run Optuna optimization
python scripts/train_models.py

# Results saved to experiments/optimized_params/
```

### Use Case 3: Run Web App

```bash
# Local
./scripts/run_app.sh

# Or visit live app
# https://heart-disease-diagnosis-vietai.streamlit.app
```

### Use Case 4: Make Predictions

```python
from src.pipeline import pipeline

# Initialize
pipeline.initialize()

# Predict
patient_data = {...}  # 13 features
results, predictions = pipeline.predict(patient_data)
final_pred, votes, total = pipeline.get_majority_vote(predictions)
```

---

## 🔄 Complete Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                         │
│    data/processed/*.csv (5 variations)                      │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MODEL TRAINING                                           │
│    notebooks/latest.ipynb                                   │
│    → 8 models × 5 datasets                                  │
│    → Save to models/saved_models/latest/*.pkl               │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. HYPERPARAMETER TUNING (Optional)                         │
│    scripts/train_models.py                                  │
│    → Optuna optimization (100 trials/model)                 │
│    → Save to experiments/optimized_params/*.json            │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EXPERIMENT TRACKING                                      │
│    scripts/experiment_manager.py                            │
│    → Log all experiments                                    │
│    → Save to experiments/experiment_log.json                │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. WEB APPLICATION                                          │
│    app/streamlit_app.py                                     │
│    → Load models from models/saved_models/latest/           │
│    → Use src/pipeline.py for predictions                    │
│    → Deploy to Streamlit Cloud                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Summary: Essential Files Only

### Must-Have Files (Core Workflow)

```
✅ data/processed/*.csv                    # Preprocessed data
✅ notebooks/latest.ipynb                  # Model training
✅ models/saved_models/latest/*.pkl        # Trained models
✅ scripts/train_models.py                 # Hyperparameter tuning
✅ scripts/experiment_manager.py           # Experiment tracking
✅ src/pipeline.py                         # Model loading & prediction
✅ src/model_functions.py                  # Feature engineering
✅ src/utils/app_utils.py                  # App utilities
✅ app/streamlit_app.py                    # Web interface
✅ experiments/optimized_params/*.json     # Best hyperparameters
✅ experiments/experiment_log.json         # Experiment history
```

### Nice-to-Have (Documentation)

```
✅ README.md                               # Project overview
✅ docs/DEPLOYMENT.md                      # Deployment guide
✅ requirements.txt                        # Dependencies
```
---

**Last Updated:** September 30, 2025  
**Team:** VietAI Learning Team - AIO2025
