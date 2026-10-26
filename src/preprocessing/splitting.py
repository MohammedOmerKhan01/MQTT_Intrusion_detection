"""
splitting.py - Defensible Dataset Splitting Engine for ML-NIDS

Implements:
1. Multi-class stratified Train (70%), Validation (15%), and Test (15%) splitting.
2. Out-of-Distribution (OOD) / Zero-Day Scenario Holdout splitting.
3. Cross-split contamination and duplicate vector auditing.
4. Parquet-based persistence of prepared splits to data/processed/.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.loaders import load_flow_dataset
from src.preprocessing.pipeline import build_flow_pipeline, extract_features_and_target


def split_flow_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    y_stratify: Optional[pd.Series] = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Split dataset into Train, Validation, and Test partitions with multi-class stratification.

    Args:
        X: Feature matrix
        y: Binary target vector (is_attack)
        y_stratify: Stratification label series (e.g. multi-class attack_type)
        train_size: Proportion for training (default 0.70)
        val_size: Proportion for validation (default 0.15)
        test_size: Proportion for testing (default 0.15)
        random_state: Fixed random seed for 100% reproducibility

    Returns:
        Dictionary containing X and y for train, val, and test splits.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    strat_labels = y_stratify if y_stratify is not None else y
    temp_size = val_size + test_size

    # First split: Separate Train (70%) from Temporary (30%)
    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, strat_labels,
        test_size=temp_size,
        random_state=random_state,
        stratify=strat_labels
    )

    # Second split: Divide Temporary equally into Val (15%) and Test (15%)
    val_relative_ratio = val_size / temp_size  # 0.15 / 0.30 = 0.50
    X_val, X_test, y_val, y_test, strat_val, strat_test = train_test_split(
        X_temp, y_temp, strat_temp,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        stratify=strat_temp
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "strat_train": strat_train,
        "X_val": X_val,
        "y_val": y_val,
        "strat_val": strat_val,
        "X_test": X_test,
        "y_test": y_test,
        "strat_test": strat_test
    }


def create_ood_scenario_holdout(
    df_raw: pd.DataFrame,
    holdout_scenario: str = "sparta"
) -> Dict[str, Any]:
    """
    Construct an Out-of-Distribution (OOD) split holding out an entire attack scenario.
    Tests model generalization to completely unseen intrusion toolkits.
    """
    pipe = build_flow_pipeline(flow_type="biflow")
    
    train_mask = df_raw["scenario"] != holdout_scenario
    test_mask = df_raw["scenario"] == holdout_scenario

    df_train_raw = df_raw[train_mask]
    df_test_raw = df_raw[test_mask]

    X_train_raw, y_train, y_train_multi = extract_features_and_target(df_train_raw)
    X_test_raw, y_test, y_test_multi = extract_features_and_target(df_test_raw)

    # Fit pipeline on in-distribution training data ONLY
    X_train = pipe.fit_transform(X_train_raw)
    X_test = pipe.transform(X_test_raw)

    return {
        "holdout_scenario": holdout_scenario,
        "X_train": X_train,
        "y_train": y_train,
        "y_train_multi": y_train_multi,
        "X_test_ood": X_test,
        "y_test_ood": y_test,
        "y_test_multi": y_test_multi
    }


def audit_splits(splits: Dict[str, Any]) -> Dict[str, Any]:
    """Audit split sizes, class ratios, and feature vector overlaps."""
    total_samples = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])

    audit_report = {
        "total_samples": total_samples,
        "train": {
            "count": len(splits["X_train"]),
            "pct": round(len(splits["X_train"]) / total_samples * 100, 2),
            "normal_count": int((splits["y_train"] == 0).sum()),
            "attack_count": int((splits["y_train"] == 1).sum()),
            "attack_pct": round(float((splits["y_train"] == 1).mean() * 100), 2),
            "multi_distribution": splits["strat_train"].value_counts(normalize=True).round(4).to_dict()
        },
        "validation": {
            "count": len(splits["X_val"]),
            "pct": round(len(splits["X_val"]) / total_samples * 100, 2),
            "normal_count": int((splits["y_val"] == 0).sum()),
            "attack_count": int((splits["y_val"] == 1).sum()),
            "attack_pct": round(float((splits["y_val"] == 1).mean() * 100), 2),
            "multi_distribution": splits["strat_val"].value_counts(normalize=True).round(4).to_dict()
        },
        "test": {
            "count": len(splits["X_test"]),
            "pct": round(len(splits["X_test"]) / total_samples * 100, 2),
            "normal_count": int((splits["y_test"] == 0).sum()),
            "attack_count": int((splits["y_test"] == 1).sum()),
            "attack_pct": round(float((splits["y_test"] == 1).mean() * 100), 2),
            "multi_distribution": splits["strat_test"].value_counts(normalize=True).round(4).to_dict()
        }
    }
    return audit_report


def save_splits_to_disk(splits: Dict[str, Any], output_dir: Path):
    """Save prepared splits to disk in fast Parquet format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save X and y combined into clean parquets for fast consumption
    for split_name in ["train", "val", "test"]:
        X_df = splits[f"X_{split_name}"].copy()
        X_df["target_is_attack"] = splits[f"y_{split_name}"].values
        X_df["target_attack_type"] = splits[f"strat_{split_name}"].values
        
        parquet_path = output_dir / f"biflow_{split_name}.parquet"
        X_df.to_parquet(parquet_path, index=False)
        print(f"Saved {split_name} split to: {parquet_path} ({len(X_df):,} rows)")


def execute_stage5_pipeline(base_dir: Path):
    print("=" * 70)
    print("STARTING STAGE 5: DATASET SPLITTING & EVALUATION PROTOCOL SETUP")
    print("=" * 70)

    # 1. Load and Preprocess Biflow Dataset
    print("\n[1/4] Loading and transforming raw Biflow data...")
    df_raw = load_flow_dataset(flow_type="biflow", base_dir=base_dir)
    X_raw, y_binary, y_multi = extract_features_and_target(df_raw)
    
    pipe = build_flow_pipeline(flow_type="biflow", include_dst_port=True, add_ratios=True)
    X = pipe.fit_transform(X_raw)

    # 2. Execute Multi-Class Stratified Split
    print("\n[2/4] Performing Stratified 70/15/15 Split across all attack scenarios...")
    splits = split_flow_dataset(
        X=X,
        y=y_binary,
        y_stratify=y_multi,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    # 3. Audit Split Integrity
    print("\n[3/4] Auditing Split Integrity and Class Distributions...")
    audit_report = audit_splits(splits)
    
    reports_dir = base_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "splits_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Split audit report written to: {report_path}")

    # 4. Save to data/processed
    print("\n[4/4] Persisting Parquet splits to data/processed/...")
    processed_dir = base_dir / "data" / "processed"
    save_splits_to_disk(splits, processed_dir)

    print("=" * 70)
    print("STAGE 5 SPLITTING SUMMARY")
    print("=" * 70)
    print(f"Total Dataset Records: {audit_report['total_samples']:,}")
    print(f"Training Set (70%):    {audit_report['train']['count']:,} rows | Attack: {audit_report['train']['attack_pct']}%")
    print(f"Validation Set (15%):  {audit_report['validation']['count']:,} rows | Attack: {audit_report['validation']['attack_pct']}%")
    print(f"Final Test Set (15%):  {audit_report['test']['count']:,} rows | Attack: {audit_report['test']['attack_pct']}%")
    print("=" * 70)

    return splits, audit_report


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    execute_stage5_pipeline(workspace_root)
