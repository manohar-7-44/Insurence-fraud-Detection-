"""
Model Evaluation and Business Impact Metrics Module.
Provides stratified cross-validation benchmarking, Precision-Recall AUC,
ROC-AUC, confusion matrices, and financial Cost-Benefit threshold optimization.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of fraud classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def compute_financial_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = 50000.0,  # Average loss when a fraudulent claim is paid
    cost_fp: float = 500.0,    # Administrative cost to investigate an honest customer
) -> Tuple[float, Dict[str, int]]:
    """
    Calculates expected business loss given a decision threshold and operational costs:
    Loss = (cost_fn * FN) + (cost_fp * FP)
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_loss = (fn * cost_fn) + (fp * cost_fp)
    counts = {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}

    return total_loss, counts


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 50000.0,
    cost_fp: float = 500.0,
    threshold_steps: int = 100,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Scans candidate probability thresholds to find the threshold that minimizes
    the total financial cost of misclassification.
    """
    thresholds = np.linspace(0.01, 0.99, threshold_steps)
    records = []

    best_threshold = 0.5
    min_loss = float("inf")

    for th in thresholds:
        loss, counts = compute_financial_loss(
            y_true, y_prob, threshold=th, cost_fn=cost_fn, cost_fp=cost_fp
        )
        f1 = f1_score(y_true, (y_prob >= th).astype(int), zero_division=0)
        rec = recall_score(y_true, (y_prob >= th).astype(int), zero_division=0)
        prec = precision_score(y_true, (y_prob >= th).astype(int), zero_division=0)

        records.append({
            "threshold": round(th, 3),
            "total_financial_loss": loss,
            "f1_score": round(f1, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            **counts,
        })

        if loss < min_loss:
            min_loss = loss
            best_threshold = th

    df_results = pd.DataFrame(records)
    return best_threshold, min_loss, df_results
