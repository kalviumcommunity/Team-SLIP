"""
Preprocessing pipeline builder for the Loan Default Predictor.

Builds a ColumnTransformer that handles numerical and categorical features
separately. The transformer is returned UNFITTED — fitting happens inside
the training pipeline to prevent data leakage.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted ColumnTransformer for preprocessing.

    Numerical features:
        1. SimpleImputer(strategy='median') — fills NaN with median
        2. StandardScaler() — zero mean, unit variance

    Categorical features:
        1. SimpleImputer(strategy='most_frequent') — fills NaN with mode
        2. OneHotEncoder(handle_unknown='ignore') — binary columns per category

    Returns
    -------
    ColumnTransformer
        Unfitted transformer. Must be used inside a Pipeline with a model
        so that .fit() is called only on training data.

    Notes
    -----
    - handle_unknown='ignore' is critical: at inference time, a new loan
      might have a category not seen during training. Without this, the
      model would crash.
    - sparse_output=False ensures the output is a dense numpy array,
      compatible with all sklearn estimators. Requires sklearn >= 1.2.
    """
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor
