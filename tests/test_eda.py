"""
test_eda.py - Automated Stage 2 Data Quality & Quality Audit Tests
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.loaders import load_flow_dataset
from src.data.schema import get_biflow_columns


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def biflow_sample(base_dir):
    # Load 10% stratified sample for fast test execution
    return load_flow_dataset(flow_type="biflow", base_dir=base_dir, sample_frac=0.1, random_state=42)


def test_data_loader_structure(biflow_sample):
    """Verify loaded dataframe contains expected columns and derived metadata."""
    expected_cols = get_biflow_columns()
    for col in expected_cols:
        assert col in biflow_sample.columns, f"Missing feature column: {col}"
    assert "scenario" in biflow_sample.columns
    assert "attack_type" in biflow_sample.columns
    assert len(biflow_sample) > 20000


def test_no_missing_or_infinite_values(biflow_sample):
    """Assert zero missing, NaN, or infinite values in flow features."""
    num_cols = biflow_sample.select_dtypes(include=[np.number])
    assert num_cols.isnull().sum().sum() == 0, "Found null values in dataset"
    assert np.isinf(num_cols).sum().sum() == 0, "Found infinite values in dataset"


def test_port_and_protocol_validity(biflow_sample):
    """Verify TCP/UDP port ranges (0-65535) and protocol numbers."""
    assert biflow_sample['prt_src'].between(0, 65535).all(), "Source port out of valid range"
    assert biflow_sample['prt_dst'].between(0, 65535).all(), "Destination port out of valid range"
    assert biflow_sample['proto'].isin([6, 17]).all(), "Unexpected protocol number found"


def test_non_negative_network_measurements(biflow_sample):
    """Verify that packet counts, byte counts, and IATs are strictly non-negative."""
    measurement_cols = [
        'fwd_num_pkts', 'bwd_num_pkts', 'fwd_num_bytes', 'bwd_num_bytes',
        'fwd_mean_iat', 'bwd_mean_iat', 'fwd_min_iat', 'fwd_max_iat'
    ]
    for col in measurement_cols:
        assert (biflow_sample[col] >= 0).all(), f"Found negative values in {col}"


def test_zero_5tuple_label_conflicts(biflow_sample):
    """Verify that no communication 5-tuple has contradictory benign and attack labels."""
    tuple_cols = ['ip_src', 'ip_dst', 'prt_src', 'prt_dst', 'proto']
    conflicts = biflow_sample.groupby(tuple_cols)['is_attack'].nunique()
    assert (conflicts > 1).sum() == 0, "Found 5-tuple collisions with conflicting labels"


def test_eda_artifacts_exist(base_dir):
    """Verify that all figures and summary report were produced."""
    figures_dir = base_dir / "artifacts" / "figures"
    expected_figures = [
        "class_balance.png",
        "feature_distributions.png",
        "tcp_flags_comparison.png",
        "correlation_heatmap.png",
        "leakage_analysis.png"
    ]
    for fig in expected_figures:
        fig_path = figures_dir / fig
        assert fig_path.exists(), f"Expected figure {fig} not found at {fig_path}"
        assert fig_path.stat().st_size > 0, f"Figure {fig} is empty"

    summary_path = base_dir / "artifacts" / "reports" / "eda_summary.json"
    assert summary_path.exists(), "EDA summary JSON was not produced"
