"""
Feature Engineering Functions for Model Pipeline
Các functions này cần thiết để unpickle models từ notebook
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer


def fe_basic(X):
    """
    Basic feature engineering - no advanced transformations
    Just returns original features separated into numerical and categorical

    Args:
        X: DataFrame with original features

    Returns:
        tuple: (X_transformed, numerical_features, categorical_features)
    """
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    return X.copy(), numerical_features, categorical_features


def fe_enhanced(X):
    """
    Enhanced feature engineering with polynomial and manual features
    """
    # This is complex - for now just return basic
    # The actual implementation would need to match the notebook's EnhancedFE
    return fe_basic(X)


def fe_poly_only(X):
    """
    Polynomial features only - creates degree 2 polynomial features from numerical columns
    """
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_array = poly.fit_transform(X[numerical_features])
    poly_feature_names = poly.get_feature_names_out(numerical_features)

    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=X.index)

    # Combine with categorical features
    X_transformed = pd.concat([X[categorical_features], poly_df], axis=1)

    return X_transformed, list(poly_feature_names), categorical_features


# Create classes to match what models expect - sklearn-compatible transformers
class BasicFE:
    def __init__(self):
        self.num_features = None
        self.cat_features = None

    def fit(self, X, y=None):
        """Fit method for sklearn compatibility"""
        _, self.num_features, self.cat_features = fe_basic(X)
        return self

    def transform(self, X):
        """Transform method for sklearn compatibility"""
        X_transformed, _, _ = fe_basic(X)
        return X_transformed

    def __call__(self, X):
        """Legacy callable interface for backward compatibility"""
        return fe_basic(X)


class EnhancedFE:
    def __init__(self):
        self.num_features = None
        self.cat_features = None

    def fit(self, X, y=None):
        """Fit method for sklearn compatibility"""
        _, self.num_features, self.cat_features = fe_enhanced(X)
        return self

    def transform(self, X):
        """Transform method for sklearn compatibility"""
        X_transformed, _, _ = fe_enhanced(X)
        return X_transformed

    def __call__(self, X):
        """Legacy callable interface for backward compatibility"""
        return fe_enhanced(X)


class PolyFE:
    def __init__(self):
        self.num_features = None
        self.cat_features = None
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        # Set default values that may be overridden during fit
        self.numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.categorical_cols = [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ]

    def fit(self, X, y=None):
        """Fit method for sklearn compatibility"""
        # Ensure attributes exist (for backward compatibility with old pickled objects)
        if not hasattr(self, "numerical_cols"):
            self.numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        if not hasattr(self, "categorical_cols"):
            self.categorical_cols = [
                "sex",
                "cp",
                "fbs",
                "restecg",
                "exang",
                "slope",
                "ca",
                "thal",
            ]
        if not hasattr(self, "poly"):
            self.poly = PolynomialFeatures(degree=2, include_bias=False)

        # Fit polynomial transformer on numerical features
        self.poly.fit(X[self.numerical_cols])
        self.num_features = list(self.poly.get_feature_names_out(self.numerical_cols))
        self.cat_features = self.categorical_cols.copy()
        return self

    def transform(self, X):
        """Transform method for sklearn compatibility"""
        # Ensure attributes exist (for backward compatibility with old pickled objects)
        if not hasattr(self, "numerical_cols"):
            self.numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        if not hasattr(self, "categorical_cols"):
            self.categorical_cols = [
                "sex",
                "cp",
                "fbs",
                "restecg",
                "exang",
                "slope",
                "ca",
                "thal",
            ]
        if not hasattr(self, "poly"):
            self.poly = PolynomialFeatures(degree=2, include_bias=False)
            # If poly wasn't fitted, we need to fit it
            if not hasattr(self.poly, "n_features_in_"):
                self.poly.fit(X[self.numerical_cols])
                self.num_features = list(
                    self.poly.get_feature_names_out(self.numerical_cols)
                )
                self.cat_features = self.categorical_cols.copy()

        # Create polynomial features
        poly_array = self.poly.transform(X[self.numerical_cols])
        poly_df = pd.DataFrame(poly_array, columns=self.num_features, index=X.index)

        # Drop original numerical features and add polynomial features
        X_transformed = X.drop(self.numerical_cols, axis=1).copy()
        X_transformed = pd.concat([X_transformed, poly_df], axis=1)

        return X_transformed

    def __call__(self, X):
        """Legacy callable interface for backward compatibility"""
        return fe_poly_only(X)


# Make sure these functions are available when unpickling
__all__ = ["fe_basic", "fe_enhanced", "fe_poly_only", "BasicFE", "EnhancedFE", "PolyFE"]
