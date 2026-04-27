"""
Data loading utilities for the Loan Default Predictor.

Handles reading CSV files and returning raw DataFrames.
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file. Supports .csv and .csv.gz formats.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed DataFrame as read from the file.

    Examples
    --------
    >>> from src.data_loader import load_data
    >>> df = load_data('data/raw/Loan_default.csv')
    >>> print(df.shape)
    (255347, 18)
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns from {filepath}")
    return df
