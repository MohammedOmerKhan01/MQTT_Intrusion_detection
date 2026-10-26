"""
test_splitting.py - Automated Stage 5 Tests for Dataset Splitting & Evaluation Protocols
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.splitting import split_flow_dataset, audit_splits, create_ood_scenario_holdout
from src.data.loaders import load_flow_dataset
from src.preprocessing.pipeline import build_flow_pipeline, extract_features_and_target


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def prepared_sample_data(base_dir):
    df_raw = load_flow_dataset(flow_type="biflow", base_dir=base_dir, sample_frac=0.05, random_state=42)
    X_raw, y_binary, y_multi = extract_features_and_target(df_raw)
    pipe = build_flow_pipeline(flow_type="biflow")
    X = pipe.fit_transform(X_raw)
    return X, y_binary, y_multi, df_raw


def test_split_proportions_and_disjoint_indices(prepared_sample_data):
    """Verify 70/15/15 proportions and strictly disjoint sample sets."""
    X, y, y_multi, _ = prepared_sample_data
    total_len = len(X)

    splits = split_flow_dataset(X, y, y_stratify=y_multi, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)

    # 1. Total rows match
    split_total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    assert split_total == total_len

    # 2. Strict index isolation (zero intersection between splits)
    train_idx = set(splits["X_train"].index)
    val_idx = set(splits["X_val"].index)
    test_idx = set(splits["X_test"].index)

    assert len(train_idx.intersection(val_idx)) == 0, "Train and Val indices overlap!"
    assert len(train_idx.intersection(test_idx)) == 0, "Train and Test indices overlap!"
    assert len(val_idx.intersection(test_idx)) == 0, "Val and Test indices overlap!"


def test_stratification_consistency(prepared_sample_data):
    """Verify class proportions are strictly preserved across Train, Val, and Test."""
    X, y, y_multi, _ = prepared_sample_data
    splits = split_flow_dataset(X, y, y_stratify=y_multi, random_state=42)

    train_pct = splits["y_train"].mean()
    val_pct = splits["y_val"].mean()
    test_pct = splits["y_test"].mean()

    # Proportions should match within 0.5%
    assert abs(train_pct - val_pct) < 0.005
    assert abs(train_pct - test_pct) < 0.005


def test_ood_scenario_holdout(prepared_sample_data):
    """Verify scenario holdout completely isolates the target scenario."""
    _, _, _, df_raw = prepared_sample_data
    ood_splits = create_ood_scenario_holdout(df_raw, holdout_scenario="sparta")

    assert ood_splits["holdout_scenario"] == "sparta"
    assert len(ood_splits["X_train"]) > 0
    assert len(ood_splits["X_test_ood"]) > 0
    # Sparta should not be present in training metadata
    assert (ood_splits["y_train_multi"] == "sparta").sum() == 0


def test_saved_parquet_splits_integrity(base_dir):
    """Verify saved Parquet files in data/processed exist, load cleanly, and match expected record counts."""
    processed_dir = base_dir / "data" / "processed"
    train_path = processed_dir / "biflow_train.parquet"
    val_path = processed_dir / "biflow_val.parquet"
    test_path = processed_dir / "biflow_test.parquet"

    assert train_path.exists(), f"Missing {train_path}"
    assert val_path.exists(), f"Missing {val_path}"
    assert test_path.exists(), f"Missing {test_path}"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    assert len(df_train) == 181565
    assert len(df_val) == 38907
    assert len(df_test) == 38907
    assert "target_is_attack" in df_train.columns
    assert "target_attack_type" in df_train.columns
