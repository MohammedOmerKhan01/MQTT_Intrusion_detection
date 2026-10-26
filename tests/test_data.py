"""
test_data.py - Automated Stage 1 Tests for Dataset Inventory & Schema Integrity
"""

import os
import json
import zipfile
import pytest
import pandas as pd
from pathlib import Path

from src.data.schema import UNIFLOW_SCHEMA, BIFLOW_SCHEMA, PACKET_SCHEMA, get_uniflow_columns, get_biflow_columns


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_archives_exist(base_dir):
    """Verify that all four required dataset archives exist in workspace root."""
    archives = [
        "uniflow_features.zip",
        "biflow_features.zip",
        "packet_features.zip",
        "pcap_files.zip"
    ]
    for arch in archives:
        path = base_dir / arch
        assert path.exists(), f"Expected archive {arch} does not exist at {path}"
        assert path.stat().st_size > 0, f"Archive {arch} is empty (0 bytes)"


def test_manifest_file_validity(base_dir):
    """Ensure data/manifest.json exists, is valid JSON, and has required sections."""
    manifest_path = base_dir / "data" / "manifest.json"
    assert manifest_path.exists(), "Manifest data/manifest.json was not generated."
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "archives" in data
    assert "aggregates" in data
    assert "uniflow_features" in data["archives"]
    assert "biflow_features" in data["archives"]
    assert data["aggregates"]["uniflow_total_records"] == 495525
    assert data["aggregates"]["biflow_total_records"] == 259379


def test_uniflow_schema_and_labels(base_dir):
    """Verify schema match and label validity for uniflow files."""
    uniflow_zip = base_dir / "uniflow_features.zip"
    expected_cols = get_uniflow_columns()
    assert len(expected_cols) == 19

    with zipfile.ZipFile(uniflow_zip, 'r') as zf:
        for fname in zf.namelist():
            if not fname.endswith('.csv'):
                continue
            with zf.open(fname) as f:
                # Read 50 rows sample for fast schema test
                df = pd.read_csv(f, nrows=50)
                assert list(df.columns) == expected_cols, f"Schema mismatch in {fname}"
                assert set(df['is_attack'].unique()).issubset({0, 1}), f"Invalid labels in {fname}"


def test_biflow_schema_and_labels(base_dir):
    """Verify schema match and label validity for biflow files."""
    biflow_zip = base_dir / "biflow_features.zip"
    expected_cols = get_biflow_columns()
    assert len(expected_cols) == 32

    with zipfile.ZipFile(biflow_zip, 'r') as zf:
        for fname in zf.namelist():
            if not fname.endswith('.csv'):
                continue
            with zf.open(fname) as f:
                df = pd.read_csv(f, nrows=50)
                assert list(df.columns) == expected_cols, f"Schema mismatch in {fname}"
                assert set(df['is_attack'].unique()).issubset({0, 1}), f"Invalid labels in {fname}"
