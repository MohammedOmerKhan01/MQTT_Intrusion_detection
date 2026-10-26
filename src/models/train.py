"""
train.py - Leakage-Safe Random Forest Training, Tuning, and Evaluation Pipeline

Trains:
1. Baseline Binary Random Forest
2. Tuned Binary Random Forest (hyperparameters optimized via RandomizedSearchCV)
3. Multi-Class Attack Family Random Forest

Evaluates on strictly held-out Validation partition:
- Accuracy, Precision, Recall, F1-Score
- False Positive Rate (FPR) & False Negative Rate (FNR)
- ROC-AUC and Precision-Recall AUC (PR-AUC)
- Inference throughput (flows/sec) and per-flow latency (microseconds)
- Multi-class per-category metrics and confusion matrix

Persists models to artifacts/models/ and metadata to artifacts/metadata/.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.evaluate import evaluate_binary_model, evaluate_multiclass_model
from src.models.tune import tune_random_forest
from src.models.persistence import save_model_artifacts


def run_training_pipeline(base_dir: Path) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING STAGE 6: RANDOM FOREST MODEL DEVELOPMENT & EVALUATION")
    print("=" * 70)

    processed_dir = base_dir / "data" / "processed"
    train_path = processed_dir / "biflow_train.parquet"
    val_path = processed_dir / "biflow_val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed datasets not found. Run splitting.py first.")

    # 1. Load Parquet Data
    print("\n[1/5] Loading preprocessed Parquet datasets...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    feature_cols = [c for c in df_train.columns if not c.startswith("target_")]
    X_train = df_train[feature_cols]
    y_train = df_train["target_is_attack"]
    y_multi_train = df_train["target_attack_type"]

    X_val = df_val[feature_cols]
    y_val = df_val["target_is_attack"]
    y_multi_val = df_val["target_attack_type"]

    print(f"Loaded {len(X_train):,} training flows and {len(X_val):,} validation flows across {len(feature_cols)} features.")

    # 2. Train Baseline Binary Random Forest
    print("\n[2/5] Training Baseline Binary Random Forest (n_estimators=50, max_depth=15)...")
    t0 = time.time()
    rf_baseline = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    fit_time_base = time.time() - t0
    print(f"Baseline fitted in {fit_time_base:.2f}s.")

    metrics_baseline = evaluate_binary_model(rf_baseline, X_val, y_val, model_name="Baseline_RF")
    print(f"  - Validation Accuracy:  {metrics_baseline['accuracy']*100:.2f}%")
    print(f"  - Validation Precision: {metrics_baseline['precision']*100:.2f}%")
    print(f"  - Validation Recall:    {metrics_baseline['recall']*100:.2f}%")
    print(f"  - Validation F1-Score:  {metrics_baseline['f1_score']*100:.2f}%")
    print(f"  - False Positive Rate:  {metrics_baseline['false_positive_rate']*100:.4f}% ({metrics_baseline['confusion_matrix']['false_positive']} flows)")
    print(f"  - False Negative Rate:  {metrics_baseline['false_negative_rate']*100:.4f}% ({metrics_baseline['confusion_matrix']['false_negative']} flows)")
    print(f"  - ROC-AUC: {metrics_baseline['roc_auc']:.6f} | PR-AUC: {metrics_baseline['pr_auc']:.6f}")

    # 3. Hyperparameter Tuning
    print("\n[3/5] Tuning Hyperparameters via RandomizedSearchCV (Training partition only)...")
    tuning_res = tune_random_forest(X_train, y_train, n_iter=6, cv_folds=3, sample_limit=25000, random_state=42)
    best_params = tuning_res["best_params"]
    print(f"Tuning complete. Best parameters found:\n  {best_params}")

    # Fit tuned model on complete training set
    print("Fitting Tuned Random Forest on full 181,565 training flows...")
    t1 = time.time()
    rf_tuned = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    fit_time_tuned = time.time() - t1
    print(f"Tuned model fitted in {fit_time_tuned:.2f}s.")

    metrics_tuned = evaluate_binary_model(rf_tuned, X_val, y_val, model_name="Tuned_RF")
    print(f"  - Tuned Accuracy:       {metrics_tuned['accuracy']*100:.2f}%")
    print(f"  - Tuned F1-Score:       {metrics_tuned['f1_score']*100:.2f}%")
    print(f"  - False Positive Rate:  {metrics_tuned['false_positive_rate']*100:.4f}%")
    print(f"  - PR-AUC:               {metrics_tuned['pr_auc']:.6f}")

    # 4. Multi-Class Attack Classifier
    print("\n[4/5] Training Multi-Class Random Forest (5 Attack Classes)...")
    t2 = time.time()
    rf_multiclass = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_multiclass.fit(X_train, y_multi_train)
    fit_time_multi = time.time() - t2
    print(f"Multi-class model fitted in {fit_time_multi:.2f}s.")

    metrics_multiclass = evaluate_multiclass_model(rf_multiclass, X_val, y_multi_val)
    print("Multi-Class Per-Category Performance:")
    for cls_name, cls_stats in metrics_multiclass["per_class"].items():
        print(f"  - {cls_name:<16}: Precision={cls_stats['precision']*100:.2f}%, Recall={cls_stats['recall']*100:.2f}%, F1={cls_stats['f1_score']*100:.2f}% (Support: {cls_stats['support']:,})")

    # 5. Persist Models and Metadata
    print("\n[5/5] Persisting fitted models and metadata artifacts...")
    artifacts_dir = base_dir / "artifacts"
    
    # Save Tuned Binary Model
    save_model_artifacts(
        model=rf_tuned,
        metrics=metrics_tuned,
        feature_names=feature_cols,
        model_name="rf_binary_model",
        output_dir=artifacts_dir,
        hyperparameters=best_params,
        extra_metadata={"fit_time_seconds": round(fit_time_tuned, 2)}
    )

    # Save Multi-Class Model
    save_model_artifacts(
        model=rf_multiclass,
        metrics=metrics_multiclass,
        feature_names=feature_cols,
        model_name="rf_multiclass_model",
        output_dir=artifacts_dir,
        extra_metadata={"fit_time_seconds": round(fit_time_multi, 2)}
    )

    # Save comparison summary
    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "model_training_summary.json"

    training_summary = {
        "baseline_binary": metrics_baseline,
        "tuned_binary": metrics_tuned,
        "multiclass": metrics_multiclass,
        "best_hyperparameters": best_params,
        "feature_count": len(feature_cols)
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print(f"Training summary report saved to: {summary_path}")
    print("=" * 70)
    print("STAGE 6 MODEL DEVELOPMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return training_summary


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    run_training_pipeline(workspace_root)
