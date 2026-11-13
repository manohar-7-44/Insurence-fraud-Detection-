"""
Data Ingestion and Quality Validation Module.
Handles raw dataset ingestion, '?' missing value parsing, schema checks,
and leak-proof Stratified Train/Test splitting.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        RAW_DATA_PATH,
        TARGET_COL,
        TARGET_MAPPING,
        DROP_COLS,
        DATE_COLS,
        RANDOM_STATE,
    )
except ImportError:
    from config import (
        RAW_DATA_PATH,
        TARGET_COL,
        TARGET_MAPPING,
        DROP_COLS,
        DATE_COLS,
        RANDOM_STATE,
    )


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw claims data, stripping column whitespace and converting
    the dataset-specific '?' placeholder into genuine NaN missing values.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Claims dataset not found at {data_path}")

    # Explicitly treat '?' as NaN so missing indicators are captured
    df = pd.read_csv(data_path, na_values=["?"], keep_default_na=True)
    df.columns = df.columns.str.strip()

    # Drop technical artifact columns like empty _c39 if present
    if "_c39" in df.columns:
        df = df.drop(columns=["_c39"])

    return df


def parse_dates_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date features into datetime objects and map binary target 'Y'/'N' to 1/0.
    """
    df = df.copy()

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if TARGET_COL in df.columns:
        # Encode target: Y -> 1, N -> 0
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)

    return df


def get_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a strict Stratified Split into Train and Test partitions BEFORE
    any imputation, scaling, or feature engineering to guarantee ZERO data leakage.
    
    Default test_size is 0.20 (200 records: 49 frauds, 151 non-frauds).
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not present in dataframe.")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_df = load_raw_data()
    print(f"Dataset Loaded Successfully! Shape: {raw_df.shape}")
    clean_df = parse_dates_and_target(raw_df)
    print(f"Target Distribution:\n{clean_df[TARGET_COL].value_counts(normalize=True).round(3)}")
    X_tr, X_te, y_tr, y_te = get_stratified_split(clean_df)
    print(f"Train Shape: {X_tr.shape}, Test Shape: {X_te.shape}")
    print(f"Train Fraud Rate: {y_tr.mean():.3f}, Test Fraud Rate: {y_te.mean():.3f}")
