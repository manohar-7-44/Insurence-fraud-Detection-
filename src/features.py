"""
Domain-Specific Feature Engineering Module for Insurance Claim Fraud Detection.
Implements scikit-learn compatible TransformerMixin to guarantee zero data leakage
and seamless serialization for Phase 2 web application deployment.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from src.config import DROP_COLS
except ImportError:
    from config import DROP_COLS


class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that extracts insurance-domain fraud indicators:
    1. Flash Claim Dynamics (days between policy bind and accident)
    2. Financial Claim Ratios (claim vs annual premium, component percentages)
    3. Geographic Discrepancies (out-of-state incident flag)
    4. Suspicious Timing & Severity Anomalies (nighttime solo crashes)
    """

    def __init__(self, drop_identifiers: bool = True):
        self.drop_identifiers = drop_identifiers

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer; no parameters to learn from training distribution
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Temporal Dynamics (Flash Claim & Days to Incident)
        if "policy_bind_date" in df.columns and "incident_date" in df.columns:
            policy_date = pd.to_datetime(df["policy_bind_date"], errors="coerce")
            incident_date = pd.to_datetime(df["incident_date"], errors="coerce")
            
            # Days elapsed between policy purchase and incident
            df["days_to_incident"] = (incident_date - policy_date).dt.days
            # Flash Claim: claims filed within first 90 days of binding
            df["is_flash_claim"] = (df["days_to_incident"] <= 90).astype(int)
            
            # Calendar temporal features
            df["incident_month"] = incident_date.dt.month
            df["incident_day_of_week"] = incident_date.dt.dayofweek
            df["incident_is_weekend"] = (df["incident_day_of_week"] >= 5).astype(int)
            
            # Drop raw date strings so downstream models do not encounter dates
            df = df.drop(columns=["policy_bind_date", "incident_date"])

        # 2. Financial & Claim Component Ratios
        if "total_claim_amount" in df.columns:
            total_claim = df["total_claim_amount"].replace(0, np.nan)
            
            # Claim amount relative to annual premium paid
            if "policy_annual_premium" in df.columns:
                df["claim_to_premium_ratio"] = (
                    df["total_claim_amount"] / (df["policy_annual_premium"] + 1.0)
                ).round(3)

            # Component proportions
            if "vehicle_claim" in df.columns:
                df["vehicle_claim_ratio"] = (df["vehicle_claim"] / total_claim).fillna(0).round(3)
            if "injury_claim" in df.columns:
                df["injury_claim_ratio"] = (df["injury_claim"] / total_claim).fillna(0).round(3)
            if "property_claim" in df.columns:
                df["property_claim_ratio"] = (df["property_claim"] / total_claim).fillna(0).round(3)

        # 3. Capital Discrepancy (Net Wealth Metric)
        if "capital-gains" in df.columns and "capital-loss" in df.columns:
            df["net_capital"] = df["capital-gains"] - df["capital-loss"].abs()

        # 4. Geographic & Out-of-State Incident Discrepancy
        if "incident_state" in df.columns and "policy_state" in df.columns:
            df["is_out_of_state_incident"] = (
                df["incident_state"] != df["policy_state"]
            ).astype(int)

        # 5. Suspicious Severity & Circumstance Combinations
        if "incident_hour_of_the_day" in df.columns:
            # High-risk nighttime hours (11 PM - 5 AM)
            df["is_night_incident"] = df["incident_hour_of_the_day"].isin(
                [23, 0, 1, 2, 3, 4, 5]
            ).astype(int)

        if "incident_severity" in df.columns and "number_of_vehicles_involved" in df.columns:
            # Single-vehicle claim claiming Major Damage / Total Loss
            df["is_solo_severe_crash"] = (
                (df["incident_severity"].isin(["Major Damage", "Total Loss"]))
                & (df["number_of_vehicles_involved"] == 1)
            ).astype(int)

        # 6. Drop Arbitrary / High-Cardinality Identifiers if specified
        if self.drop_identifiers:
            cols_to_drop = [c for c in DROP_COLS if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        return df
