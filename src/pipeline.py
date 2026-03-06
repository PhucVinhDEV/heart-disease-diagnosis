"""
Preprocessing Pipeline for Heart Disease Prediction
Cập nhật để load models từ folder latest/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add src directory to path FIRST before any imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import feature engineering functions needed for unpickling models
from model_functions import fe_basic, fe_enhanced, fe_poly_only

# Also make the classes available for unpickling
from model_functions import BasicFE, EnhancedFE, PolyFE

# Make these available in __main__ namespace for pickle compatibility
import __main__

__main__.fe_basic = fe_basic
__main__.fe_enhanced = fe_enhanced
__main__.fe_poly_only = fe_poly_only
__main__.BasicFE = BasicFE
__main__.EnhancedFE = EnhancedFE
__main__.PolyFE = PolyFE

# Also make them available in the global namespace of this module
globals()["fe_basic"] = fe_basic
globals()["fe_enhanced"] = fe_enhanced
globals()["fe_poly_only"] = fe_poly_only
globals()["BasicFE"] = BasicFE
globals()["EnhancedFE"] = EnhancedFE
globals()["PolyFE"] = PolyFE

# Import additional sklearn components for feature selection
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    f_classif,
    SelectFromModel,
    RFE,
    RFECV,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Feature selection functions (copied from notebook)
def fs_variance(X_pre, y, threshold=0.01):
    """Remove features with low variance"""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_pre)
    return selector.transform(X_pre), np.where(selector.get_support())[0]


def fs_correlation(X_pre, y, corr_threshold=0.9):
    """Remove highly correlated features"""
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()
    df_pre = pd.DataFrame(X_pre)
    df_pre.columns = [f"feat_{i}" for i in range(X_pre.shape[1])]
    corr = df_pre.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper.columns if any(upper[column] > corr_threshold)
    ]
    keep = [col for col in df_pre.columns if col not in to_drop]
    indices = np.array([int(col.split("_")[1]) for col in keep])
    return df_pre[keep].values, indices


class KBestMISelector:
    """
    Select K best features based on Mutual Information
    
    Args:
        k: Number or proportion of features to select
           If int: select k features
           If float (0-1): select k * n_features
    """
    def __init__(self, k=0.8):
        self.k = k

    def fit(self, X, y):
        """Fit the selector on training data"""
        k = self.k if isinstance(self.k, int) else int(X.shape[1] * self.k)
        self.selector = SelectKBest(mutual_info_classif, k=k)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        """Transform X to selected features"""
        return self.selector.transform(X)

    def get_support(self):
        """Get boolean mask of selected features"""
        return self.selector.get_support()


def fs_kbest_mi(X_pre, y, k=0.8):
    """Select K best features using mutual information"""
    selector = KBestMISelector(k=k)
    selector.fit(X_pre, y)
    return selector.transform(X_pre), np.where(selector.get_support())[0]


def fs_rfe_svm(X_pre, y):
    """Recursive Feature Elimination with SVM using cross-validation"""
    from sklearn.model_selection import StratifiedKFold
    model = SVC(kernel="linear", random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    selector = RFECV(estimator=model, step=0.2, min_features_to_select=1, cv=cv)
    selector.fit(X_pre, y)
    return X_pre[:, selector.support_], np.where(selector.support_)[0]


def fs_select_model_lr(X_pre, y):
    """Select features based on Logistic Regression coefficients"""
    model = LogisticRegression(random_state=42, max_iter=500, solver="liblinear")
    selector = SelectFromModel(model, threshold="median")
    selector.fit(X_pre, y)
    return X_pre[:, selector.get_support()], np.where(selector.get_support())[0]


def fs_rfe_lr(X_pre, y):
    """Recursive Feature Elimination with Logistic Regression using cross-validation"""
    from sklearn.model_selection import StratifiedKFold
    model = LogisticRegression(random_state=42, max_iter=500)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    selector = RFECV(estimator=model, step=0.2, min_features_to_select=1, cv=cv)
    selector.fit(X_pre, y)
    return X_pre[:, selector.support_], np.where(selector.support_)[0]


def fs_boruta(X_pre, y, max_iter=50, alpha=0.05, random_state=42):
    """
    Boruta feature selection algorithm
    
    Args:
        X_pre: Preprocessed features
        y: Target variable
        max_iter: Maximum iterations for boruta
        alpha: Significance level
        random_state: Random seed for reproducibility
    """
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()
    X = pd.DataFrame(X_pre)
    X.columns = [f"feat_{i}" for i in range(X.shape[1])]
    num_feat = X.shape[1]
    hits = np.zeros(num_feat)
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(random_state)
    
    for iter_ in range(max_iter):
        # Create shadow features with controlled randomness
        shadow = X.apply(lambda col: rng.permutation(col.values))
        shadow.columns = [f"shadow_{i}" for i in range(num_feat)]
        X_boruta = pd.concat([X, shadow], axis=1)
        rf = RandomForestClassifier(n_jobs=1, max_depth=None, random_state=random_state)
        rf.fit(X_boruta, y)
        feat_imp_X = rf.feature_importances_[:num_feat]
        feat_imp_shadow = rf.feature_importances_[num_feat:]
        hits += feat_imp_X > feat_imp_shadow.max()
    criteria = hits / max_iter > 1 - alpha
    return X_pre[:, criteria], np.where(criteria)[0]


# Feature selection options mapping
fs_options = {
    "variance": fs_variance,
    "correlation": fs_correlation,
    "kbest_mi": fs_kbest_mi,
    "rfe_svm": fs_rfe_svm,
    "select_lr": fs_select_model_lr,
    "rfe_lr": fs_rfe_lr,
    "boruta": fs_boruta,
    "none": lambda X, y: (X, np.arange(X.shape[1])),  # No feature selection
}


class HeartDiseasePipeline:
    """
    Preprocessing pipeline for heart disease prediction models.
    Handles data loading, preprocessing, and model prediction.
    Load models từ models/saved_models/latest/
    """

    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.metrics = {}
        self.is_fitted = False
        self.feature_order = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        self.X_train = None  # Training data for fitting preprocessors
        self.y_train = None

    def _apply_feature_engineering(self, X):
        """
        Apply basic feature engineering (không dùng polynomial vì models đã được train với basic FE)

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data sau khi FE
        """
        X = X.copy()

        # Đảm bảo tất cả features có mặt
        for col in self.feature_order:
            if col not in X.columns:
                X[col] = 0

        # Sắp xếp theo đúng thứ tự
        X = X[self.feature_order]

        return X

    def load_models(self, models_dir="models/saved_models/latest"):
        """
        Load all trained models từ folder latest/
        Support cả pipeline models (từ notebook với preprocessing) và simple models
        """
        # Convert to absolute path relative to project root
        if not os.path.isabs(models_dir):
            # Get project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            models_dir = project_root / models_dir

        try:
            # Model mapping (theo naming convention trong app)
            model_files = {
                "Logistic Regression": "best_lr_model_pipeline.pkl",
                "Random Forest": "best_rf_model_pipeline.pkl",
                "K-Nearest Neighbors": "best_knn_model_pipeline.pkl",
                "Decision Tree": "best_dt_model_pipeline.pkl",
                "AdaBoost": "best_ada_model_pipeline.pkl",
                "Gradient Boosting": "best_gb_model_pipeline.pkl",
                "XGBoost": "best_xgb_model_pipeline.pkl",
                "LightGBM": "best_lgbm_model_pipeline.pkl",
                "SVM": "best_svm_model_pipeline.pkl",
                "Ensemble": "best_ensemble_model_pipeline.pkl",
            }

            # Load models
            for model_name, model_file in model_files.items():
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        # Load model/pipeline
                        loaded_obj = joblib.load(model_path)

                        # Check if it's a pipeline (dict with 'model' key) or simple model
                        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                            # Pipeline format từ notebook - contains full preprocessing info
                            self.models[model_name] = loaded_obj
                            print(f"[OK] Loaded {model_name} (pipeline format)")
                        else:
                            # Simple model format từ save_models.py
                            # Tạo pipeline giả với model đơn giản
                            self.models[model_name] = {
                                "model": loaded_obj,
                                "preprocessor": None,  # Không có preprocessor
                                "pre_pipe": None,  # Không có preprocessing pipeline
                                "fs_indices": None,  # Không có feature selection
                                "feature_names": None,
                            }
                            print(f"[OK] Loaded {model_name} (simple format)")

                    except Exception as e:
                        print(f"[WARN]  Skipping {model_name}: {str(e)[:80]}...")
                        # Uncomment below for full error details during debugging
                        # import traceback
                        # traceback.print_exc()
                        continue
                else:
                    print(f"[WARN]  {model_name} not found at {model_path}")

            if len(self.models) > 0:
                self.is_fitted = True
                print(f"\n[OK] Successfully loaded {len(self.models)} models")
                return True
            else:
                print("[ERROR] No models loaded")
                return False

        except Exception as e:
            print(f"[ERROR] Error loading models: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def load_metrics(self, models_dir="models/saved_models/latest"):
        """Load model performance metrics từ best_models_summary.json với cấu trúc mới"""
        # Convert to absolute path relative to project root
        if not os.path.isabs(models_dir):
            # Get project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            models_dir = project_root / models_dir

        try:
            summary_path = os.path.join(models_dir, "best_models_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary_data = json.load(f)

                    # Map model abbreviations to full names (theo naming trong app)
                    model_name_map = {
                        "lr": "Logistic Regression",
                        "rf": "Random Forest",
                        "knn": "K-Nearest Neighbors",
                        "dt": "Decision Tree",
                        "ada": "AdaBoost",
                        "gb": "Gradient Boosting",
                        "xgb": "XGBoost",
                        "lgbm": "LightGBM",
                        "nb": "Naive Bayes",
                        "svm": "SVM",
                        "ensemble": "Ensemble",
                    }

                    # Load metrics từ cấu trúc mới
                    for abbrev, info in summary_data.items():
                        full_name = model_name_map.get(abbrev, abbrev)
                        self.metrics[full_name] = {
                            "cv_optimization_score": info.get(
                                "cv_optimization_score", 0
                            ),
                            "optimization_metric": info.get(
                                "optimization_metric", "roc_auc"
                            ),
                            "test_metrics": info.get("test_metrics", {}),
                            "feature_engineering": info.get(
                                "feature_engineering", "N/A"
                            ),
                            "scaler": info.get("scaler", "N/A"),
                            "feature_selection": info.get("feature_selection", "N/A"),
                            "hyperparameters": info.get("hyperparameters", {}),
                        }

                    print(f"[OK] Loaded metrics for {len(self.metrics)} models")
                    return True
            else:
                print(f"[WARN]  Metrics file not found: {summary_path}")
                return False
        except Exception as e:
            print(f"[ERROR] Error loading metrics: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def predict(self, input_data):
        """
        Make predictions using all loaded models
        Each model uses its own preprocessing pipeline recreated from stored parameters

        Recreates the exact preprocessing from train.ipynb:
        1. Apply FE transformer (stored and fitted)
        2. Recreate and fit preprocessor (scaling + OHE) on training data
        3. Apply feature selection using fs_name and training data
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call initialize() first.")

        if len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")

        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Training data not loaded. Pipeline needs training data to recreate preprocessing."
            )

        try:
            results = []
            predictions = []

            for model_name, model_data in self.models.items():
                try:
                    # Check if this is a pipeline model (from notebook) or simple model
                    if isinstance(model_data, dict) and "model" in model_data:
                        # Pipeline model - recreate preprocessing from stored parameters
                        model = model_data["model"]
                        fe_transformer = model_data.get("fe_transformer")
                        scaler_name = model_data.get("scaler_name")
                        fs_name = model_data.get("fs_name")
                        numerical_features = model_data.get("numerical_features")
                        categorical_features = model_data.get("categorical_features")

                        # Step 1: Apply FE to input data
                        if fe_transformer is not None:
                            X_input_fe = fe_transformer.transform(input_data)
                        else:
                            X_input_fe = input_data.copy()

                        # Step 2: Use stored preprocessor or create new one
                        stored_preprocessor = model_data.get("preprocessor")
                        
                        if stored_preprocessor is not None:
                            # Use fitted preprocessor from training (BEST - ensures consistency)
                            X_input_pre = stored_preprocessor.transform(X_input_fe)
                        else:
                            # Fallback: Recreate and fit preprocessor on training data
                            # (only if preprocessor not stored - for backward compatibility)
                            if fe_transformer is not None:
                                X_train_fe = fe_transformer.transform(self.X_train)
                            else:
                                X_train_fe = self.X_train.copy()
                            
                            if scaler_name == "standard":
                                scaler = StandardScaler()
                            elif scaler_name == "robust":
                                scaler = RobustScaler()
                            else:  # minmax
                                scaler = MinMaxScaler()

                            try:
                                ohe = OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                )
                            except TypeError:
                                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ("num", scaler, numerical_features),
                                    ("cat", ohe, categorical_features),
                                ]
                            )
                            
                            # Fit on training data, transform input
                            X_train_pre = preprocessor.fit_transform(X_train_fe)
                            X_input_pre = preprocessor.transform(X_input_fe)

                        # Convert to dense if sparse
                        if hasattr(X_input_pre, "toarray"):
                            X_input_pre = X_input_pre.toarray()
                        
                        # Convert X_train_pre if it exists (fallback path)
                        if 'X_train_pre' in locals() and hasattr(X_train_pre, "toarray"):
                            X_train_pre = X_train_pre.toarray()

                        # Step 3: Apply feature selection
                        # Use stored fs_indices if available (trained once), otherwise compute
                        fs_indices_stored = model_data.get("fs_indices")
                        
                        if fs_indices_stored is not None:
                            # Use pre-computed indices from training (consistent results)
                            selected_features = X_input_pre[:, fs_indices_stored]
                        elif fs_name and fs_name in fs_options:
                            # Fallback: compute FS (only if no stored indices)
                            # This requires X_train_pre which should exist in fallback path
                            if 'X_train_pre' not in locals():
                                # Need to compute X_train_pre for FS
                                if fe_transformer is not None:
                                    X_train_fe = fe_transformer.transform(self.X_train)
                                else:
                                    X_train_fe = self.X_train.copy()
                                X_train_pre = stored_preprocessor.transform(X_train_fe) if stored_preprocessor else X_train_fe
                                if hasattr(X_train_pre, "toarray"):
                                    X_train_pre = X_train_pre.toarray()
                            
                            _, fs_indices = fs_options[fs_name](
                                X_train_pre, self.y_train
                            )
                            selected_features = X_input_pre[:, fs_indices]
                        else:
                            selected_features = X_input_pre

                    else:
                        # Simple model - basic preprocessing only
                        model = model_data
                        X_fe = self._apply_feature_engineering(input_data)
                        selected_features = X_fe.values

                    # Make prediction
                    prediction = model.predict(selected_features)[0]
                    predictions.append(prediction)

                    # Get probability/confidence
                    if hasattr(model, "predict_proba"):
                        prediction_proba = model.predict_proba(selected_features)[0]
                        confidence = (
                            prediction_proba[prediction]
                            if prediction < len(prediction_proba)
                            else prediction_proba[0]
                        )
                    elif hasattr(model, "decision_function"):
                        # For SVM and other models with decision_function
                        decision_scores = model.decision_function(selected_features)[0]
                        # Convert to probability-like confidence (sigmoid)
                        prob_class_1 = 1 / (1 + np.exp(-decision_scores))
                        # Get confidence for the predicted class
                        confidence = prob_class_1 if prediction == 1 else (1 - prob_class_1)
                    else:
                        # Fallback for models without probability methods
                        confidence = 1.0 if prediction == 1 else 0.0

                    results.append(
                        {
                            "Model": model_name,
                            "Prediction": (
                                "High Risk" if prediction == 1 else "Low Risk"
                            ),
                            "Confidence": float(confidence),
                        }
                    )

                except Exception as e:
                    print(f"[ERROR] Error with {model_name}: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    continue

            return results, predictions

        except Exception as e:
            print(f"[ERROR] Error making predictions: {str(e)}")
            import traceback

            traceback.print_exc()
            return [], []

    def get_majority_vote(self, predictions):
        """Get majority vote from all predictions"""
        high_risk_votes = sum(p == 1 for p in predictions)
        low_risk_votes = sum(p == 0 for p in predictions)

        if high_risk_votes > low_risk_votes:
            return "High Risk", high_risk_votes, len(predictions)
        else:
            return "Low Risk", low_risk_votes, len(predictions)

    def initialize(self, models_dir="models/saved_models/latest"):
        """
        Initialize the complete pipeline
        Load models từ folder latest/ and training data for preprocessing
        """
        # Convert to absolute path relative to project root
        if not os.path.isabs(models_dir):
            # Get project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            models_dir = project_root / models_dir

        print("=" * 80)
        print("[HEART] Initializing Heart Disease Prediction Pipeline...")
        print("=" * 80)

        # Load training data (needed to recreate preprocessing)
        try:
            project_root = Path(__file__).parent.parent
            train_data_path = project_root / "data" / "raw" / "raw_train.csv"

            if train_data_path.exists():
                train_df = pd.read_csv(train_data_path)
                self.X_train = train_df.drop("target", axis=1)
                self.y_train = train_df["target"]
                print(f"[OK] Loaded training data: {self.X_train.shape}")
            else:
                print(f"[WARN]  Training data not found at {train_data_path}")
                print(
                    "   Pipeline will work but preprocessing may not match training exactly"
                )
        except Exception as e:
            print(f"[WARN]  Could not load training data: {e}")

        # Load models (models đã có preprocessor riêng, không cần fit thêm)
        if not self.load_models(models_dir):
            print("[ERROR] Failed to load models")
            return False

        # Load metrics
        self.load_metrics(models_dir)

        # Mark as fitted
        self.is_fitted = True

        print("=" * 80)
        print(f"[OK] Pipeline initialized successfully with {len(self.models)} models")
        print("=" * 80)
        return True


# Global pipeline instance
pipeline = HeartDiseasePipeline()
