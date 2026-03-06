"""
Hyperparameter Tuning với Optuna
Tối ưu hóa hyperparameters cho 7 models trên 5 dataset types
"""

import optuna
import numpy as np
import pandas as pd
import joblib
import json
import warnings
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer

# Import experiment_manager from same directory
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from experiment_manager import ExperimentManager

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    Tối ưu hóa hyperparameters cho các models sử dụng Optuna
    """
    
    def __init__(self, n_trials=50, cv=5, random_state=42):
        """
        Initialize tuner
        
        Args:
            n_trials (int): Số lần thử cho mỗi model
            cv (int): Number of cross-validation folds
            random_state (int): Random seed
        """
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.scorer = make_scorer(f1_score, average='weighted')
        
        # Initialize experiment manager
        self.exp_manager = ExperimentManager()
        self.exp_manager.set_global_seed(random_state)
        
        # Results storage
        self.best_params = {}
        self.best_scores = {}
        
    def optimize_logistic_regression(self, X_train, y_train, dataset_name=""):
        """Optimize Logistic Regression"""
        print(f"\n🔍 Optimizing Logistic Regression on {dataset_name}...")
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.random_state
            }
            
            # Handle solver-penalty compatibility
            if params['penalty'] == 'l1' and params['solver'] == 'saga':
                params['solver'] = 'liblinear'
            
            model = LogisticRegression(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"lr_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_random_forest(self, X_train, y_train, dataset_name=""):
        """Optimize Random Forest"""
        print(f"\n🌲 Optimizing Random Forest on {dataset_name}...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            model = RandomForestClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"rf_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_knn(self, X_train, y_train, dataset_name=""):
        """Optimize K-Nearest Neighbors"""
        print(f"\n👥 Optimizing KNN on {dataset_name}...")
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                'p': trial.suggest_int('p', 1, 3)
            }
            
            model = KNeighborsClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"knn_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_decision_tree(self, X_train, y_train, dataset_name=""):
        """Optimize Decision Tree"""
        print(f"\n🌳 Optimizing Decision Tree on {dataset_name}...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'random_state': self.random_state
            }
            
            model = DecisionTreeClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"dt_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_adaboost(self, X_train, y_train, dataset_name=""):
        """Optimize AdaBoost"""
        print(f"\n🚀 Optimizing AdaBoost on {dataset_name}...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
                'random_state': self.random_state
            }
            
            model = AdaBoostClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"ada_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_gradient_boosting(self, X_train, y_train, dataset_name=""):
        """Optimize Gradient Boosting"""
        print(f"\n⚡ Optimizing Gradient Boosting on {dataset_name}...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
            
            model = GradientBoostingClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"gb_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_xgboost(self, X_train, y_train, dataset_name=""):
        """Optimize XGBoost"""
        print(f"\n💫 Optimizing XGBoost on {dataset_name}...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            model = XGBClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"xgb_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_naive_bayes(self, X_train, y_train, dataset_name=""):
        """Optimize Naive Bayes (Gaussian)"""
        print(f"\n🎯 Optimizing Naive Bayes on {dataset_name}...")
        
        def objective(trial):
            params = {
                'var_smoothing': trial.suggest_float('var_smoothing', 1e-10, 1e-5, log=True)
            }
            
            model = GaussianNB(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"nb_{dataset_name}")
        study.optimize(objective, n_trials=min(self.n_trials, 20), show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def optimize_svm(self, X_train, y_train, dataset_name=""):
        """Optimize SVM"""
        print(f"\n🎲 Optimizing SVM on {dataset_name}...")
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'random_state': self.random_state
            }
            
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            
            model = SVC(**params, probability=True)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, 
                                   scoring=self.scorer, n_jobs=-1).mean()
            return score
        
        study = optuna.create_study(direction='maximize', 
                                   study_name=f"svm_{dataset_name}")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Best F1 Score: {study.best_value:.4f}")
        return study.best_params, study.best_value
    
    def run_full_optimization(self, data_dir="data/processed", output_dir="experiments/optimized_params"):
        """
        Chạy optimization cho tất cả models trên tất cả datasets
        
        Args:
            data_dir (str): Thư mục chứa processed data
            output_dir (str): Thư mục lưu best params
        """
        print("=" * 80)
        print("🚀 STARTING FULL HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset types
        dataset_types = ['raw', 'fe', 'dt', 'fe_dt', 'master_boruta']
        
        # Models to optimize
        models = {
            'lr': self.optimize_logistic_regression,
            'rf': self.optimize_random_forest,
            'knn': self.optimize_knn,
            'dt': self.optimize_decision_tree,
            # 'ada': self.optimize_adaboost,  # TODO: Uncomment when AdaBoost model is trained
            'gb': self.optimize_gradient_boosting,
            'xgb': self.optimize_xgboost,
            'nb': self.optimize_naive_bayes,
            'svm': self.optimize_svm
        }
        
        all_results = {}
        
        # Loop through datasets
        for dataset_type in dataset_types:
            print(f"\n{'=' * 80}")
            print(f"📊 DATASET: {dataset_type.upper()}")
            print(f"{'=' * 80}")
            
            # Load data
            train_file = Path(data_dir) / f"{dataset_type}_train.csv"
            
            if not train_file.exists():
                print(f"⚠️  Warning: {train_file} not found. Skipping...")
                continue
            
            df_train = pd.read_csv(train_file)
            
            # Separate features and target
            if 'target' in df_train.columns:
                X_train = df_train.drop('target', axis=1)
                y_train = df_train['target']
            elif 'num' in df_train.columns:
                X_train = df_train.drop('num', axis=1)
                y_train = (df_train['num'] > 0).astype(int)
            else:
                print(f"⚠️  Warning: No target column found in {train_file}. Skipping...")
                continue
            
            dataset_results = {}
            
            # Loop through models
            for model_name, optimize_func in models.items():
                try:
                    best_params, best_score = optimize_func(X_train, y_train, dataset_type)
                    
                    dataset_results[model_name] = {
                        'best_params': best_params,
                        'best_cv_f1': float(best_score),
                        'dataset': dataset_type
                    }
                    
                    # Log to experiment manager
                    self.exp_manager.log_experiment(
                        model_name=model_name,
                        dataset_type=dataset_type,
                        params=best_params,
                        metrics={'f1': float(best_score)},
                        additional_info={'optimization': 'optuna', 'n_trials': self.n_trials}
                    )
                    
                except Exception as e:
                    print(f"❌ Error optimizing {model_name} on {dataset_type}: {str(e)}")
                    continue
            
            all_results[dataset_type] = dataset_results
            
            # Save intermediate results
            intermediate_file = output_path / f"best_params_{dataset_type}.json"
            with open(intermediate_file, 'w') as f:
                json.dump(dataset_results, f, indent=2)
            print(f"\n💾 Saved results to {intermediate_file}")
        
        # Save all results
        final_file = output_path / "best_params_all.json"
        with open(final_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'=' * 80}")
        print(f"✅ OPTIMIZATION COMPLETE!")
        print(f"💾 All results saved to {final_file}")
        print(f"{'=' * 80}")
        
        # Generate summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results):
        """Print optimization summary"""
        print("\n" + "=" * 80)
        print("📊 OPTIMIZATION SUMMARY")
        print("=" * 80)
        
        for dataset_type, dataset_results in results.items():
            print(f"\n📁 Dataset: {dataset_type.upper()}")
            print("-" * 80)
            
            # Sort by score
            sorted_models = sorted(dataset_results.items(), 
                                 key=lambda x: x[1]['best_cv_f1'], 
                                 reverse=True)
            
            for rank, (model_name, info) in enumerate(sorted_models, 1):
                print(f"  {rank}. {model_name.upper():8s} - F1: {info['best_cv_f1']:.4f}")
        
        print("\n" + "=" * 80)


# Main execution
if __name__ == "__main__":
    # Initialize tuner
    tuner = HyperparameterTuner(n_trials=50, cv=5, random_state=42)
    
    # Run full optimization
    results = tuner.run_full_optimization(
        data_dir="data/processed",
        output_dir="experiments/optimized_params"
    )
    
    # Generate experiment report
    tuner.exp_manager.generate_report(output_format='html')
    tuner.exp_manager.export_to_csv()
    
    print("\n✅ Done! Check experiments/ folder for results.")
