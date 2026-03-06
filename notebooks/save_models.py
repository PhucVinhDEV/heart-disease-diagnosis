"""
Script để train và lưu các models từ notebook Model-Training-and-Evalution
Tạo file .pkl cho Streamlit app
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Set seed
SEED = 42
np.random.seed(SEED)

def read_csv(filepath):
    """Đọc CSV và trả về X, y"""
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def find_optimal_rf(X_train, y_train, n_estimators_range=range(50, 501, 50), cv_splits=3):
    """Tìm Random Forest tối ưu"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    for n in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=5, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt', bootstrap=True,
            n_jobs=-1, random_state=SEED
        )
        cv_score = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        scores.append(cv_score.mean())
    
    best_n = n_estimators_range[np.argmax(scores)]
    best_model = RandomForestClassifier(
        n_estimators=best_n, max_depth=5, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True,
        n_jobs=-1, random_state=SEED
    )
    best_model.fit(X_train, y_train)
    return best_model, best_n, max(scores)

def find_optimal_ada(X_train, y_train, n_estimators_range=range(50, 501, 50), cv_splits=3):
    """Tìm AdaBoost tối ưu"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    for n in n_estimators_range:
        ada = AdaBoostClassifier(
            n_estimators=n, learning_rate=0.1, algorithm='SAMME.R',
            random_state=SEED
        )
        cv_score = cross_val_score(ada, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        scores.append(cv_score.mean())
    
    best_n = n_estimators_range[np.argmax(scores)]
    best_model = AdaBoostClassifier(
        n_estimators=best_n, learning_rate=0.1, algorithm='SAMME.R',
        random_state=SEED
    )
    best_model.fit(X_train, y_train)
    return best_model, best_n, max(scores)

def find_optimal_gb(X_train, y_train, n_estimators_range=range(50, 501, 50), cv_splits=3):
    """Tìm Gradient Boosting tối ưu"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    for n in n_estimators_range:
        gb = GradientBoostingClassifier(
            n_estimators=n, learning_rate=0.1, max_depth=3,
            min_samples_split=2, min_samples_leaf=1, subsample=1.0,
            random_state=SEED
        )
        cv_score = cross_val_score(gb, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        scores.append(cv_score.mean())
    
    best_n = n_estimators_range[np.argmax(scores)]
    best_model = GradientBoostingClassifier(
        n_estimators=best_n, learning_rate=0.1, max_depth=3,
        min_samples_split=2, min_samples_leaf=1, subsample=1.0,
        random_state=SEED
    )
    best_model.fit(X_train, y_train)
    return best_model, best_n, max(scores)

def find_optimal_xgb(X_train, y_train, n_estimators_range=range(50, 501, 50), cv_splits=3):
    """Tìm XGBoost tối ưu"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    for n in n_estimators_range:
        xgb = XGBClassifier(
            n_estimators=n, learning_rate=0.1, max_depth=3,
            min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
            random_state=SEED, eval_metric='logloss'
        )
        cv_score = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        scores.append(cv_score.mean())
    
    best_n = n_estimators_range[np.argmax(scores)]
    best_model = XGBClassifier(
        n_estimators=best_n, learning_rate=0.1, max_depth=3,
        min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
        random_state=SEED, eval_metric='logloss'
    )
    best_model.fit(X_train, y_train)
    return best_model, best_n, max(scores)

def main():
    """Main function để train và lưu models"""
    print("🚀 Bắt đầu training và lưu models...")
    
    # Tạo thư mục models nếu chưa có
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Đọc datasets
    datasets = {
        'raw': 'data/processed/raw_train.csv',
        'fe': 'data/processed/fe_train.csv', 
        'dt': 'data/processed/dt_train.csv',
        'fe_dt': 'data/processed/fe_dt_train.csv'
    }
    
    models_info = {}
    
    for dataset_name, filepath in datasets.items():
        print(f"\n📊 Training models trên dataset: {dataset_name}")
        
        # Đọc data
        X_train, y_train = read_csv(filepath)
        print(f"   Shape: {X_train.shape}")
        
        # Train Random Forest
        print("   🌲 Training Random Forest...")
        rf_model, rf_n, rf_score = find_optimal_rf(X_train, y_train)
        models_info[f'{dataset_name}_rf'] = {'model': rf_model, 'n_estimators': rf_n, 'score': rf_score}
        
        # Train AdaBoost
        print("   🚀 Training AdaBoost...")
        ada_model, ada_n, ada_score = find_optimal_ada(X_train, y_train)
        models_info[f'{dataset_name}_ada'] = {'model': ada_model, 'n_estimators': ada_n, 'score': ada_score}
        
        # Train Gradient Boosting
        print("   📈 Training Gradient Boosting...")
        gb_model, gb_n, gb_score = find_optimal_gb(X_train, y_train)
        models_info[f'{dataset_name}_gb'] = {'model': gb_model, 'n_estimators': gb_n, 'score': gb_score}
        
        # Train XGBoost
        print("   ⚡ Training XGBoost...")
        xgb_model, xgb_n, xgb_score = find_optimal_xgb(X_train, y_train)
        models_info[f'{dataset_name}_xgb'] = {'model': xgb_model, 'n_estimators': xgb_n, 'score': xgb_score}
    
    # Lưu models
    print("\n💾 Lưu models...")
    for model_name, info in models_info.items():
        filename = f'models/saved_models/{model_name}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(info['model'], f)
        print(f"   ✅ Đã lưu {filename} (n_estimators={info['n_estimators']}, score={info['score']:.4f})")
    
    # Lưu metadata
    metadata = {
        'models_info': models_info,
        'datasets': datasets,
        'seed': SEED
    }
    
    with open('models/saved_models/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n🎉 Hoàn thành! Đã lưu {len(models_info)} models vào models/saved_models/")
    print("📁 Files được tạo:")
    for model_name in models_info.keys():
        print(f"   - {model_name}.pkl")
    print("   - metadata.pkl")

if __name__ == "__main__":
    main()


