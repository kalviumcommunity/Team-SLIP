"""
Model training utilities for the Loan Default Predictor.

Combines a preprocessor and a model into a single sklearn Pipeline,
then fits it on training data. The fitted pipeline can be serialized
as a single pickle file.
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: BaseEstimator,
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    Build a Pipeline from a preprocessor and model, then fit on training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (raw, unprocessed). The preprocessor handles
        imputation, scaling, and encoding.
    y_train : pd.Series
        Training labels (0 = No Default, 1 = Default).
    model : BaseEstimator
        Any sklearn-compatible classifier (e.g., LogisticRegression,
        DecisionTreeClassifier, KNeighborsClassifier).
    preprocessor : ColumnTransformer
        Unfitted ColumnTransformer from build_preprocessor().

    Returns
    -------
    Pipeline
        Fitted pipeline. Calling .predict() or .predict_proba() on this
        pipeline will first transform the input, then classify.

    Notes
    -----
    - The preprocessor is fitted ONLY on X_train inside the pipeline.
      This prevents data leakage from the test set.
    - The returned pipeline is the object that should be pickled.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    print(f"Pipeline fitted with {model.__class__.__name__}")
    print(f"Training samples: {X_train.shape[0]:,}, Features: {X_train.shape[1]}")

    return pipeline
