"""
Script để tạo dummy models cho test API
Models không train thật, chỉ fit ngẫu nhiên để test structure
"""

import pickle
import os
import json
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# Import ML components
from pipeline import HeartDiseasePipeline
from model_functions import BasicFE

# Sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

SEED = 42
np.random.seed(SEED)

# === Create dummy models directory ===
MODEL_DIR = current_dir / "models" / "saved_models" / "latest"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("[Dummy Models] Creating fake trained models for API testing")
print("=" * 60)

# === Load training data để fit dummy models ===
DATA_PATH = current_dir / "data" / "raw" / "raw_train.csv"

if DATA_PATH.exists():
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']
    print(f"[OK] Loaded training data: {X.shape}")
else:
    # Fallback: tạo dummy data
    X = np.random.rand(200, 13)
    y = np.random.randint(0, 2, 200)
    print("[WARN] Training data not found, using random data")

# === Define models ===
dummy_models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=500, random_state=SEED),
        'filename': 'best_lr_model_pipeline.pkl',
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=SEED),
        'filename': 'best_rf_model_pipeline.pkl',
    },
    'XGBoost': {
        'model': XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=SEED),
        'filename': 'best_xgb_model_pipeline.pkl',
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=SEED),
        'filename': 'best_gb_model_pipeline.pkl',
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(n_neighbors=5),
        'filename': 'best_knn_model_pipeline.pkl',
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(max_depth=5, random_state=SEED),
        'filename': 'best_dt_model_pipeline.pkl',
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(n_estimators=50, random_state=SEED),
        'filename': 'best_ada_model_pipeline.pkl',
    },
    'SVM': {
        'model': SVC(probability=True, random_state=SEED),
        'filename': 'best_svm_model_pipeline.pkl',
    },
}

# === Create feature engineering transformer ===
fe_transformer = BasicFE()

# === Train and save each model ===
model_name_map = {
    'Logistic Regression': 'lr',
    'Random Forest': 'rf',
    'XGBoost': 'xgb',
    'Gradient Boosting': 'gb',
    'K-Nearest Neighbors': 'knn',
    'Decision Tree': 'dt',
    'AdaBoost': 'ada',
    'SVM': 'svm',
}

summary = {}

for name, config in dummy_models_config.items():
    print(f"\n[Training] {name}...")
    
    # Fit model (dummy fit - không có real training)
    model = config['model']
    model.fit(X, y)
    
    # Create pipeline dict (matching expected format from pipeline.py)
    pipeline_dict = {
        'model': model,
        'fe_transformer': fe_transformer,
        'scaler_name': 'standard',
        'fs_name': 'none',
        'numerical_features': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
        'categorical_features': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
        'preprocessor': None,  # Will be recreated
        'fs_indices': None,
        'feature_names': None,
    }
    
    # Save model
    filepath = MODEL_DIR / config['filename']
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline_dict, f)
    
    print(f"  [OK] Saved: {config['filename']}")
    
    # Add to summary
    abbrev = model_name_map[name]
    summary[abbrev] = {
        'model_type': abbrev,
        'cv_optimization_score': round(np.random.uniform(0.85, 0.95), 4),
        'optimization_metric': 'roc_auc',
        'feature_engineering': 'basic',
        'scaler': 'standard',
        'feature_selection': 'none',
        'hyperparameters': model.get_params() if hasattr(model, 'get_params') else {},
        'test_metrics': {
            'accuracy': round(np.random.uniform(0.80, 0.92), 4),
            'precision': round(np.random.uniform(0.78, 0.90), 4),
            'recall_sensitivity': round(np.random.uniform(0.78, 0.92), 4),
            'f1_score': round(np.random.uniform(0.78, 0.90), 4),
            'specificity': round(np.random.uniform(0.80, 0.92), 4),
            'roc_auc': round(np.random.uniform(0.90, 0.96), 4),
        },
        'dataset': 'UCI Heart Disease Cleveland Dataset (dummy)',
    }

# === Save summary JSON ===
summary_path = MODEL_DIR / "best_models_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print(f"[OK] Created {len(dummy_models_config)} dummy models")
print(f"[OK] Summary saved: {summary_path}")
print("=" * 60)
print("\nModels created:")
for name in dummy_models_config.keys():
    print(f"  - {name}")
print("\nNext steps:")
print("  1. Copy models to Cardiovascular_FastAPI/models/")
print("  2. Copy data/raw/ to Cardiovascular_FastAPI/data/")
print("  3. Copy src/ to Cardiovascular_FastAPI/src/cardiovascular_fastapi/ml/")
print("  4. Run: uvicorn src.cardiovascular_fastapi.main:app --reload")