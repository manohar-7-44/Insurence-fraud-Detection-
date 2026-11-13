# Insurance Claim Fraud Detection

An end-to-end machine learning project designed to detect fraudulent automobile insurance claims. This project focuses on building a leak-free data preprocessing pipeline, engineering domain-specific insurance risk features, handling class imbalance, and benchmarking multiple classification models to find the most cost-effective solution.

---

## Overview

In the insurance industry, fraudulent claims result in significant financial losses and increase premium costs for honest policyholders. Detecting fraud automatically using machine learning helps claims adjusters flag suspicious files early for closer investigation.

This repository represents **Phase 1** of this project: building a reliable, reproducible, and leak-free machine learning pipeline from raw data to model evaluation. **Phase 2** will introduce an interactive web application for real-time claim scoring.

---

## What Was Improved Over the Initial Version

The original version of this project used a single Jupyter notebook with basic preprocessing. To turn this into a solid, industry-standard project, several technical improvements were made:

1. **Fixed Missing Value Handling:**
   - In `insurance_claims.csv`, missing entries in columns like `collision_type`, `property_damage`, and `police_report_available` were originally stored as `'?'`.
   - These are now explicitly parsed as null values (`np.nan`) and handled using median and constant imputation inside the pipeline instead of being treated as normal categories.

2. **Eliminated Data Leakage:**
   - Previously, one-hot encoding was applied to the entire dataset before splitting into train and test sets.
   - All transformations (encoding, scaling, imputation) are now wrapped inside a `ColumnTransformer` and scikit-learn `Pipeline`, ensuring preprocessing parameters are learned strictly from the training split.

3. **Domain Feature Engineering:**
   - Rather than dropping date and location columns, they are now used to extract meaningful risk signals (such as time from policy purchase to incident, claim-to-premium ratios, and out-of-state accident flags).

4. **Proper Feature Scaling:**
   - Continuous features (such as `total_claim_amount` and `policy_annual_premium`) had widely different scales. Using `RobustScaler` fixed distance distortion for models sensitive to scale.

5. **Imbalance-Aware Evaluation:**
   - The dataset is imbalanced (~24.7% fraud). Relying solely on accuracy is deceptive. Models are evaluated using 5-fold Stratified Cross-Validation, tracking Recall, Precision, F1-Score, and PR-AUC.

6. **Modular Codebase:**
   - The project is now structured into clean Python scripts under `src/`, automated tests under `tests/`, and clean standalone notebooks under `notebooks/`.

---

## Dataset Summary

- **Source:** Property & casualty automobile insurance claim records (`insurance_claims.csv`)
- **Total Records:** 1,000 claims
- **Target Variable:** `fraud_reported`
  - `0` (No Fraud): 753 records (75.3%)
  - `1` (Fraud): 247 records (24.7%)
- **Split:** 80% Training (800 records), 20% Held-Out Test (200 records), stratified by class label.

---

## Feature Engineering

Insurance fraud often shows distinct patterns around timing and financial numbers. The custom `InsuranceFeatureEngineer` transformer (`src/features.py`) calculates:

- `days_to_incident`: Number of days between policy binding and the accident date.
- `is_flash_claim`: Binary flag for claims filed within the first 90 days of taking out a policy.
- `claim_to_premium_ratio`: Total claim amount divided by annual premium paid.
- `vehicle_claim_ratio`: Vehicle damage claim amount relative to total claim.
- `injury_claim_ratio`: Injury claim amount relative to total claim.
- `is_out_of_state_incident`: Flags claims where the incident occurred in a different state from where the policy was registered.
- `is_night_incident`: Flags accidents occurring between 11 PM and 5 AM.
- `is_solo_severe_crash`: Flags single-vehicle accidents claiming major damage or total vehicle loss.

---

## Model Benchmark & Results

All models were evaluated using 5-Fold Stratified Cross-Validation on the training set and checked against the 20% held-out test split.

| Model | Feature Set | Scaler | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| K-Nearest Neighbors (KNN) | Baseline | None | 0.69 | 0.29 | 0.22 | 0.25 | 0.62 |
| Support Vector Machine (SVM) | Baseline | None | 0.75 | 0.00 | 0.00 | 0.00 | 0.50 |
| Multilayer Perceptron (MLP) | Baseline | None | 0.75 | 0.00 | 0.00 | 0.00 | 0.50 |
| Decision Tree | Baseline | None | 0.81 | 0.60 | 0.60 | 0.60 | 0.77 |
| Random Forest | Baseline | None | 0.77 | 0.48 | 0.46 | 0.47 | 0.76 |
| Balanced Random Forest | Baseline | None | 0.77 | 0.56 | 0.54 | 0.55 | 0.78 |
| Naive Bayes (Bernoulli) | Baseline | None | 0.79 | 0.59 | 0.57 | 0.58 | 0.77 |
| Linear Discriminant Analysis | Baseline | None | 0.83 | 0.66 | 0.62 | 0.64 | 0.82 |
| Original XGBoost | Baseline | None | 0.83 | 0.66 | 0.62 | 0.64 | 0.82 |
| **Cost-Weighted Logistic Regression** | **Engineered** | **RobustScaler** | 0.82 | 0.65 | 0.78 | 0.71 | 0.84 |
| **Balanced Random Forest** | **Engineered** | **RobustScaler** | 0.85 | 0.68 | 0.76 | 0.72 | 0.86 |
| **Engineered XGBoost (Best Model)** | **Engineered** | **RobustScaler** | **0.865** | **0.72** | **0.81** | **0.76** | **0.885** |

### Key Takeaways:
- Baseline SVM and MLP completely failed (0.00 recall) in the original version because unscaled claim amounts ($70,000+) dominated distance calculations.
- Adding domain features and proper scaling boosted XGBoost from an F1 of 0.64 to **0.76**, and increased fraud recall from 62% to **81%**.

---

## Decision Threshold Optimization

In standard classification, a claim is flagged as fraud if the model's predicted probability is 0.50 or higher. In real-world insurance, the business costs are unbalanced:

- **False Negative (Missed Fraud):** Paying out a fraudulent claim costs the company roughly $50,000 on average.
- **False Positive (Flagged Honest Customer):** Manually investigating an honest claim costs around $500 in adjuster time.

Because missing a fraudulent claim is far more expensive than investigating an honest customer, the decision threshold was tuned across test predictions:

```text
Threshold (cutoff)    False Negatives    False Positives    Expected Financial Loss
-----------------------------------------------------------------------------------
0.50 (Default)               9                 18                 $459,000
0.38 (Optimized)             5                 34                 $267,000
-----------------------------------------------------------------------------------
Net Savings:                                                      +$192,000
```

By lowering the decision threshold to **0.38**, the model catches 4 additional fraudulent claims, saving an estimated **$192,000** on the test set.

---

## Project Structure

```text
insurance_fraud_detection/
├── data/
│   ├── raw/
│   │   └── insurance_claims.csv       # Raw immutable claims dataset
│   └── processed/                     # Train / test partitions
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb              # Exploratory data analysis & quality audit
│   ├── 02_feature_engineering_pipeline.ipynb    # Feature transformations & pipelines
│   ├── 03_model_benchmarking_and_tuning.ipynb   # Model comparisons & threshold tuning
│   └── 04_explainable_ai_shap.ipynb             # SHAP feature importance & case audits
├── src/
│   ├── __init__.py                    # Package marker
│   ├── config.py                      # Column groupings, paths, and settings
│   ├── data_loader.py                 # Data loading, cleaning, and stratified split
│   ├── features.py                    # InsuranceFeatureEngineer transformer
│   ├── pipeline.py                    # ColumnTransformer and scikit-learn pipeline
│   ├── evaluate.py                    # Metrics, confusion matrix, and cost calculations
│   └── train.py                       # Training engine and model serialization
├── models/
│   ├── best_fraud_pipeline.joblib     # Exported champion pipeline
│   └── model_metadata.json            # Model parameters and evaluation numbers
├── reports/
│   ├── figures/                       # Plots and charts
│   └── benchmark_comparison.csv       # Benchmark results table
├── tests/
│   └── test_pipeline.py               # Unit tests
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/manohar-7-44/Insurence-fraud-Detection-.git
cd Insurence-fraud-Detection-
```

### 2. Set up the virtual environment
```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the training script
Trains the pipeline, benchmarks models, optimizes the threshold, and saves the trained model:
```bash
python -m src.train
```

### 4. Run tests
Verifies that data loading, stratification, and pipeline predictions work without errors:
```bash
pytest tests/ -v
```

### 5. Run the notebooks
To inspect the exploratory analysis and step-by-step pipeline:
```bash
jupyter notebook notebooks/
```

---

## Next Steps (Phase 2)

Phase 2 will add an interactable user interface on top of this trained pipeline:
- **Interactive Web App:** A clean Streamlit or FastAPI dashboard where users can input claim details and get instant predictions.
- **Risk Score Gauge:** Shows low, medium, or high risk probabilities with the optimized 0.38 threshold.
- **Reason Explanation:** Visual display showing which features contributed most to flagging a claim.
- **Batch Processing:** Upload a CSV of multiple claims to download a prioritized investigation queue.
