from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "insurance_claims.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Definition
TARGET_COL = "fraud_reported"
TARGET_MAPPING = {"Y": 1, "N": 0}

# Identifiers / Technical Artifacts to Drop
DROP_COLS = [
    "policy_number",       # Unique claim identifier (overfitting hazard)
    "insured_zip",         # Granular identifier (prevents generalizability)
    "incident_location",   # Street-level text address
    "_c39",                # Redundant empty trailing column in original CSV
]

# Date Columns for Feature Engineering
DATE_COLS = [
    "policy_bind_date",
    "incident_date",
]

# Numerical Features
NUMERICAL_COLS = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
]

# Categorical Features
CATEGORICAL_COLS = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "police_report_available",
    "auto_make",
    "auto_model",
]

# High-Risk Missing Fields that require special imputation
MISSING_VAL_INDICATORS = ["?"]
