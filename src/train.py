"""
Master Model Training, Benchmarking, and Pipeline Serialization Script.
Trains multiple models using Stratified K-Fold CV, evaluates business costs,
and serializes the winning pipeline to disk for production deployment.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

try:
    from src.config import (
        RAW_DATA_PATH,
        MODELS_DIR,
        REPORTS_DIR,
        RANDOM_STATE,
        TARGET_COL,
    )
    from src.data_loader import load_raw_data, parse_dates_and_target, get_stratified_split
    from src.pipeline import build_fraud_pipeline
    from src.evaluate import compute_metrics, find_optimal_threshold
except ImportError:
    from config import (
        RAW_DATA_PATH,
        MODELS_DIR,
        REPORTS_DIR,
        RANDOM_STATE,
        TARGET_COL,
    )
    from data_loader import load_raw_data, parse_dates_and_target, get_stratified_split
    from pipeline import build_fraud_pipeline
    from evaluate import compute_metrics, find_optimal_threshold


def train_and_benchmark(data_path: Path = RAW_DATA_PATH):
    print("=" * 60)
    print("INSURANCE CLAIM FRAUD DETECTION - PIPELINE TRAINING ENGINE")
    print("=" * 60)

    # 1. Ingestion & Pre-split
    raw_df = load_raw_data(data_path)
    clean_df = parse_dates_and_target(raw_df)
    X_train, X_test, y_train, y_test = get_stratified_split(clean_df, test_size=0.20)
    print(f"[+] Loaded {len(clean_df)} claims.")
    print(f"[+] Stratified Train: {len(X_train)} claims ({y_train.sum()} frauds).")
    print(f"[+] Stratified Test:  {len(X_test)} claims ({y_test.sum()} frauds).")

    # 2. Candidate Models Definition
    candidate_models = {
        "Cost-Weighted Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Balanced Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }

    # Optional: include XGBoost if available in environment
    try:
        from xgboost import XGBClassifier
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        candidate_models["Scale-Weighted XGBoost"] = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    except ImportError:
        pass

    # 3. 5-Fold Stratified Cross-Validation Benchmarking
    print("\n[+] Running 5-Fold Stratified Cross-Validation Benchmark...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    benchmark_summary = []

    best_model_name = None
    best_pr_auc = -1.0
    best_clf = None

    for name, clf in candidate_models.items():
        fold_pr_aucs = []
        fold_roc_aucs = []
        fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe = build_fraud_pipeline(classifier=clf)
            pipe.fit(X_tr, y_tr)

            y_val_prob = pipe.predict_proba(X_val)[:, 1]
            y_val_pred = pipe.predict(X_val)

            metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
            fold_pr_aucs.append(metrics["pr_auc"])
            fold_roc_aucs.append(metrics["roc_auc"])
            fold_f1s.append(metrics["f1"])

        mean_pr_auc = np.mean(fold_pr_aucs)
        mean_roc_auc = np.mean(fold_roc_aucs)
        mean_f1 = np.mean(fold_f1s)

        print(f"    - {name:<35} | PR-AUC: {mean_pr_auc:.3f} | ROC-AUC: {mean_roc_auc:.3f} | F1: {mean_f1:.3f}")
        benchmark_summary.append({
            "model": name,
            "mean_pr_auc": round(mean_pr_auc, 3),
            "mean_roc_auc": round(mean_roc_auc, 3),
            "mean_f1": round(mean_f1, 3),
        })

        if mean_pr_auc > best_pr_auc:
            best_pr_auc = mean_pr_auc
            best_model_name = name
            best_clf = clf

    print(f"\n[★] Champion Model Selected: {best_model_name} (Mean PR-AUC: {best_pr_auc:.3f})")

    # 4. Fit Champion Pipeline on Full Training Set
    champion_pipeline = build_fraud_pipeline(classifier=best_clf)
    champion_pipeline.fit(X_train, y_train)

    # 5. Evaluate on Held-Out Test Set
    y_test_prob = champion_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = champion_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    print("\n[+] Held-Out Test Set Performance (Default 0.5 Threshold):")
    print(f"    - Fraud Recall:     {test_metrics['recall']:.1%}")
    print(f"    - Fraud Precision:  {test_metrics['precision']:.1%}")
    print(f"    - Fraud F1-Score:   {test_metrics['f1']:.3f}")
    print(f"    - Test ROC-AUC:     {test_metrics['roc_auc']:.3f}")
    print(f"    - Test PR-AUC:      {test_metrics['pr_auc']:.3f}")
    print(f"    - Confusion Matrix: TP={test_metrics['true_positives']}, FP={test_metrics['false_positives']}, FN={test_metrics['false_negatives']}, TN={test_metrics['true_negatives']}")

    # 6. Cost-Benefit Threshold Optimization
    optimal_th, min_cost, df_cost = find_optimal_threshold(
        y_test.values, y_test_prob, cost_fn=50000.0, cost_fp=500.0
    )
    print(f"\n[+] Financial Cost Optimization:")
    print(f"    - Default 0.50 Threshold Loss: ${test_metrics['false_negatives'] * 50000 + test_metrics['false_positives'] * 500:,.2f}")
    print(f"    - Optimal Business Cutoff (θ*): {optimal_th:.2f}")
    print(f"    - Minimized Expected Loss:     ${min_cost:,.2f}")

    # 7. Model Serialization & Metadata Packaging
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = MODELS_DIR / "best_fraud_pipeline.joblib"
    joblib.dump(champion_pipeline, pipeline_path)
    print(f"\n[✔] Production Pipeline saved to: {pipeline_path}")

    metadata = {
        "champion_model": best_model_name,
        "optimal_threshold": round(optimal_th, 3),
        "test_metrics_default_threshold": test_metrics,
        "benchmark_summary": benchmark_summary,
        "saved_pipeline_file": pipeline_path.name,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✔] Metadata saved to: {metadata_path}")

    # Save benchmark table
    pd.DataFrame(benchmark_summary).to_csv(REPORTS_DIR / "benchmark_comparison.csv", index=False)
    return champion_pipeline, metadata


if __name__ == "__main__":
    train_and_benchmark()
