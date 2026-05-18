"""
data_processing/preprocess.py

Shared preprocessing utilities for the Retraction Watch analysis pipeline.
Handles loading, cleaning, multi-value field splitting, and categorical encoding.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "retraction_watch.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Retraction Watch CSV."""
    return pd.read_csv(path)


def load_with_dates(path: str = DATA_PATH) -> pd.DataFrame:
    """Load data and parse RetractionDate as datetime."""
    df = pd.read_csv(path)
    df["RetractionDate"] = pd.to_datetime(df["RetractionDate"], errors="coerce")
    return df


def explode_semicolon_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Split a semicolon-delimited column into individual rows.
    Strips whitespace from each value after splitting.
    """
    df = df.copy()
    df[col] = df[col].str.split(";")
    df = df.explode(col)
    df[col] = df[col].str.strip()
    return df


def explode_multiple_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Sequentially explode multiple semicolon-delimited columns."""
    for col in cols:
        df = explode_semicolon_column(df, col)
    return df


def encode_features(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode a list of categorical feature columns.

    Returns:
        df: DataFrame with new `<col>_encoded` columns added
        encoders: dict mapping column name -> fitted LabelEncoder
    """
    encoders = {}
    for col in features:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def build_binary_target(df: pd.DataFrame, target_col: str = "RetractionNature") -> pd.DataFrame:
    """
    Create a binary target column: 1 if RetractionNature == 'Retraction', else 0.
    Drops rows where target_col is null.
    """
    df = df[df[target_col].notna()].copy()
    df["Target"] = df[target_col].apply(lambda x: 1 if x == "Retraction" else 0)
    return df
