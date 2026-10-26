"""
evaluate.py - Comprehensive Evaluation Metrics for Binary & Multi-class ML-NIDS

Computes:
1. Standard Classification Metrics: Precision, Recall, F1-Score, Accuracy
2. Operational Diagnostic Metrics: False Positive Rate (FPR), False Negative Rate (FNR)
3. Threshold-Independent Curves: ROC-AUC, Precision-Recall AUC (Average Precision)
4. Multi-class per-class metrics, support, and confusion matrices
5. Model latency & inference throughput benchmarks
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def evaluate_binary_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "RandomForest_Binary"
) -> Dict[str, Any]:
    """
    Compute full diagnostic evaluation suite for a binary intrusion classifier.
    """
    t0 = time.time()
    y_pred = model.predict(X_val)
    inference_time = time.time() - t0
    latency_us_per_flow = (inference_time / len(X_val)) * 1_000_000

    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    # Confusion matrix breakdown
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Rate calculations
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    accuracy = float(accuracy_score(y_val, y_pred))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall = float(recall_score(y_val, y_pred, zero_division=0))
    f1 = float(f1_score(y_val, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_val, y_proba)) if y_proba is not None else None
    pr_auc = float(average_precision_score(y_val, y_proba)) if y_proba is not None else None

    return {
        "model_name": model_name,
        "sample_count": len(X_val),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "performance": {
            "total_inference_time_sec": round(inference_time, 4),
            "latency_us_per_flow": round(latency_us_per_flow, 2),
            "throughput_flows_per_sec": round(len(X_val) / max(1e-5, inference_time), 0)
        }
    }


def evaluate_multiclass_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "RandomForest_Multiclass"
) -> Dict[str, Any]:
    """
    Compute multi-class evaluation metrics broken down by attack category.
    """
    t0 = time.time()
    y_pred = model.predict(X_val)
    inference_time = time.time() - t0

    classes = list(model.classes_)
    cm = confusion_matrix(y_val, y_pred, labels=classes)
    report_dict = classification_report(y_val, y_pred, labels=classes, output_dict=True, zero_division=0)

    per_class_metrics = {}
    for cls_name in classes:
        if cls_name in report_dict:
            per_class_metrics[cls_name] = {
                "precision": round(report_dict[cls_name]["precision"], 6),
                "recall": round(report_dict[cls_name]["recall"], 6),
                "f1_score": round(report_dict[cls_name]["f1-score"], 6),
                "support": int(report_dict[cls_name]["support"])
            }

    return {
        "model_name": model_name,
        "sample_count": len(X_val),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 6),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 6),
        "classes": classes,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "performance": {
            "total_inference_time_sec": round(inference_time, 4),
            "throughput_flows_per_sec": round(len(X_val) / max(1e-5, inference_time), 0)
        }
    }
