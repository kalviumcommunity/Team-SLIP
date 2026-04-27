"""
Inference utilities for the Loan Default Predictor.

Handles loading a trained pipeline and making predictions on new data.
Used by the Streamlit app and by evaluation notebooks.
"""

import pickle

import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    """
    Load a trained pipeline from a pickle file.

    Parameters
    ----------
    model_path : str
        Path to the .pkl file containing the fitted Pipeline.

    Returns
    -------
    Pipeline
        Fitted pipeline ready for prediction.
    """
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return pipeline


def predict(pipeline: Pipeline, input_df: pd.DataFrame) -> dict:
    """
    Make a prediction on a single input or batch of inputs.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline (preprocessor + classifier).
    input_df : pd.DataFrame
        Input features as a DataFrame. Column names must match
        the training features exactly (NUMERICAL_FEATURES + CATEGORICAL_FEATURES).

    Returns
    -------
    dict
        For single-row input:
            {'prediction': int, 'probability': float}
        Where prediction is 0 (No Default) or 1 (Default),
        and probability is the predicted probability of class 1 (default).

    Examples
    --------
    >>> result = predict(pipeline, new_loan_df)
    >>> print(result)
    {'prediction': 1, 'probability': 0.73}
    """
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    return {
        'prediction': int(prediction),
        'probability': round(float(probability), 4)
    }
