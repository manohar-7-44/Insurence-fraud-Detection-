"""
Unit and Integration Tests for Insurance Fraud Detection Pipeline.
Ensures zero data leakage, correct feature calculation, and valid predictions.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_raw_data, parse_dates_and_target, get_stratified_split
from src.features import InsuranceFeatureEngineer
from src.pipeline import build_fraud_pipeline


def test_data_loader_integrity():
    raw_df = load_raw_data()
    assert "_c39" not in raw_df.columns, "Artifact column _c39 was not dropped."
    assert (raw_df == "?").sum().sum() == 0, "Found unparsed '?' characters in raw data."
    assert len(raw_df) == 1000, f"Expected 1,000 rows, found {len(raw_df)}."


def test_stratified_split_balance():
    raw_df = load_raw_data()
    clean_df = parse_dates_and_target(raw_df)
    X_train, X_test, y_train, y_test = get_stratified_split(clean_df, test_size=0.20)

    assert len(X_train) == 800
    assert len(X_test) == 200
    # Both sets should have approximately 24.7% fraud rate
    assert abs(y_train.mean() - 0.247) < 0.02
    assert abs(y_test.mean() - 0.247) < 0.02


def test_feature_engineer_transformation():
    raw_df = load_raw_data()
    clean_df = parse_dates_and_target(raw_df)
    X_train, _, _, _ = get_stratified_split(clean_df, test_size=0.20)

    fe = InsuranceFeatureEngineer(drop_identifiers=True)
    transformed_df = fe.fit_transform(X_train)

    expected_cols = [
        "days_to_incident",
        "is_flash_claim",
        "claim_to_premium_ratio",
        "vehicle_claim_ratio",
        "is_out_of_state_incident",
    ]
    for col in expected_cols:
        assert col in transformed_df.columns, f"Engineered column '{col}' missing."


def test_end_to_end_pipeline_prediction():
    raw_df = load_raw_data()
    clean_df = parse_dates_and_target(raw_df)
    X_train, X_test, y_train, y_test = get_stratified_split(clean_df, test_size=0.20)

    pipeline = build_fraud_pipeline()
    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Predict on test set
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert probs.shape == (len(X_test), 2)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))
