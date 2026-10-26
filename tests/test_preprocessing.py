"""
test_preprocessing.py - Automated Stage 4 Tests for Leakage-Safe Feature Preprocessing
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.loaders import load_flow_dataset
from src.preprocessing.pipeline import (
    build_flow_pipeline,
    extract_features_and_target,
    IDENTIFIER_COLUMNS,
    METADATA_COLUMNS
)


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sample_biflow(base_dir):
    # Load 5% sample for rapid test validation
    return load_flow_dataset(flow_type="biflow", base_dir=base_dir, sample_frac=0.05, random_state=42)


def test_separation_of_features_and_targets(sample_biflow):
    """Verify clean separation of X features, y_binary target, and metadata."""
    X_raw, y_binary, y_multi = extract_features_and_target(sample_biflow)

    assert len(X_raw) == len(y_binary) == len(sample_biflow)
    assert "is_attack" not in X_raw.columns
    assert "scenario" not in X_raw.columns
    assert "attack_type" not in X_raw.columns
    assert set(y_binary.unique()).issubset({0, 1})


def test_complete_exclusion_of_identifiers(sample_biflow):
    """Verify that network identifiers (ip_src, ip_dst, prt_src) are 100% purged."""
    X_raw, _, _ = extract_features_and_target(sample_biflow)
    pipeline = build_flow_pipeline(flow_type="biflow", include_dst_port=True, add_ratios=True)
    X_trans = pipeline.fit_transform(X_raw)

    for identifier in IDENTIFIER_COLUMNS:
        assert identifier not in X_trans.columns, f"Identifier {identifier} found in features!"


def test_engineered_ratios_validity(sample_biflow):
    """Verify engineered flow ratios are finite, non-null, and non-negative."""
    X_raw, _, _ = extract_features_and_target(sample_biflow)
    pipeline = build_flow_pipeline(flow_type="biflow", include_dst_port=True, add_ratios=True)
    X_trans = pipeline.fit_transform(X_raw)

    engineered_cols = ["pkt_ratio", "total_pkts", "byte_ratio", "total_bytes", "bytes_per_pkt", "pkt_len_ratio"]
    for col in engineered_cols:
        assert col in X_trans.columns, f"Missing engineered column {col}"
        assert X_trans[col].isnull().sum() == 0, f"Found NaN in {col}"
        assert not np.isinf(X_trans[col]).any(), f"Found Inf in {col}"
        assert (X_trans[col] >= 0).all(), f"Found negative values in {col}"


def test_schema_json_persistence(base_dir):
    """Verify that the stable feature schema JSON artifact was saved and has 34 features."""
    schema_path = base_dir / "artifacts" / "metadata" / "biflow_feature_schema.json"
    assert schema_path.exists(), f"Schema file not found at {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["flow_type"] == "biflow"
    assert data["feature_count"] == 34
    assert len(data["feature_names"]) == 34
    assert "pkt_ratio" in data["feature_names"]
