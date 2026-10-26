"""
test_persistence.py - Comprehensive Unit Tests for Stage 10 Model Persistence & CLI

Validates:
1. Cryptographic SHA-256 hash generation and integrity verification.
2. Tampering detection and protection against untrusted model deserialization.
3. Model manifest schema compliance and complete metadata bundle fields.
4. Composite deployment bundle (nids_bundle.joblib) serialization.
5. Standalone headless CLI (src/models/predict.py) execution on tabular and syslog outputs.
6. Strict schema rejection of incompatible (uniflow) inputs.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.persistence import (
    compute_file_sha256,
    verify_artifact_integrity,
    load_model_artifact_safe,
    generate_system_manifest,
    save_system_manifest,
    save_composite_deployment_bundle,
    SecurityIntegrityError,
    SchemaMismatchError
)


@pytest.fixture(scope="module")
def manifest():
    manifest_path = ROOT_DIR / "artifacts" / "metadata" / "model_manifest.json"
    if not manifest_path.exists():
        manifest_path = save_system_manifest()
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sha256_checksum_calculation():
    """Verify cryptographic SHA-256 computation."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"DeepMind-Antigravity-NIDS-Test-Bytes")
        tf_path = Path(tf.name)

    try:
        sha = compute_file_sha256(tf_path)
        assert isinstance(sha, str)
        assert len(sha) == 64
        # Verify deterministic hash
        sha_again = compute_file_sha256(tf_path)
        assert sha == sha_again
    finally:
        os.unlink(tf_path)


def test_integrity_verification_pass_and_fail():
    """Verify artifact integrity checking and tampering detection."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"GenuineModelWeightsData")
        tf_path = Path(tf.name)

    try:
        true_sha = compute_file_sha256(tf_path)
        # Should pass with true hash
        assert verify_artifact_integrity(tf_path, true_sha) is True

        # Should raise SecurityIntegrityError with tampered hash
        fake_sha = "0000000000000000000000000000000000000000000000000000000000000000"
        with pytest.raises(SecurityIntegrityError, match="Cryptographic hash mismatch"):
            verify_artifact_integrity(tf_path, fake_sha)
    finally:
        os.unlink(tf_path)


def test_manifest_schema_completeness(manifest):
    """Verify all mandatory metadata fields exist in model_manifest.json."""
    required_keys = [
        "manifest_version", "model_id", "model_version", "created_at",
        "artifacts", "class_label_mapping", "feature_schema",
        "training_configuration", "selected_hyperparameters",
        "evaluation_metrics", "operational_thresholds",
        "shap_configuration", "system_dependencies"
    ]
    for key in required_keys:
        assert key in manifest, f"Missing required manifest key: {key}"

    # Verify 34 features
    assert manifest["feature_schema"]["feature_count"] == 34
    assert len(manifest["feature_schema"]["feature_names"]) == 34

    # Verify class mappings
    assert manifest["class_label_mapping"]["binary_classification"]["1"] == "ATTACK"
    assert "sparta" in manifest["class_label_mapping"]["multiclass_classification"].values()

    # Verify evaluation metrics
    metrics = manifest["evaluation_metrics"]
    assert metrics["holdout_test_accuracy"] >= 0.99
    assert metrics["false_positive_rate"] < 0.001
    assert "confusion_matrix" in metrics


def test_safe_model_loading_with_checksum():
    """Verify that load_model_artifact_safe validates checksums before loading."""
    binary_path = ROOT_DIR / "artifacts" / "models" / "rf_binary_model.joblib"
    model = load_model_artifact_safe(binary_path, enforce_checksum=True)
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_composite_deployment_bundle():
    """Verify that composite bundle persists and loads correctly."""
    bundle_path = ROOT_DIR / "artifacts" / "models" / "nids_bundle.joblib"
    if not bundle_path.exists():
        bundle_path = save_composite_deployment_bundle()

    assert bundle_path.exists()
    import joblib
    bundle = joblib.load(bundle_path)
    assert "binary_model" in bundle
    assert "multiclass_model" in bundle
    assert "manifest" in bundle
    assert bundle["manifest"]["model_version"] == "1.0.0"


def test_cli_execution_json_and_cef():
    """Verify standalone CLI predict.py executes and formats JSON and Syslog CEF."""
    test_data_path = ROOT_DIR / "data" / "processed" / "biflow_test.parquet"

    # Test 1: JSON format with checksum verification
    cmd_json = [
        sys.executable,
        str(ROOT_DIR / "src" / "models" / "predict.py"),
        "--input", str(test_data_path),
        "--limit", "3",
        "--format", "json",
        "--verify-checksum"
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True)
    assert res_json.returncode == 0, f"CLI JSON failed: {res_json.stderr}"
    output_obj = json.loads(res_json.stdout)
    assert output_obj["model_version"] == "1.0.0"
    assert output_obj["telemetry_summary"]["total_flows_evaluated"] == 3

    # Test 2: Syslog CEF format
    cmd_cef = [
        sys.executable,
        str(ROOT_DIR / "src" / "models" / "predict.py"),
        "--input", str(test_data_path),
        "--limit", "2",
        "--format", "syslog"
    ]
    res_cef = subprocess.run(cmd_cef, capture_output=True, text=True)
    assert res_cef.returncode == 0, f"CLI CEF failed: {res_cef.stderr}"
    assert "CEF:0|DeepMind-Cyber|ML-NIDS" in res_cef.stdout


def test_cli_rejects_uniflow_incompatible_schema():
    """Verify that standalone CLI explicitly rejects uniflow data instead of silent substitution."""
    # Create mock uniflow DataFrame
    uniflow_df = pd.DataFrame([{
        "proto": 6,
        "num_pkts": 5,
        "num_bytes": 350,
        "mean_pkt_len": 70.0,
        "mean_iat": 0.001
    }])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
        uniflow_df.to_csv(tf.name, index=False)
        tf_path = Path(tf.name)

    try:
        cmd_uniflow = [
            sys.executable,
            str(ROOT_DIR / "src" / "models" / "predict.py"),
            "--input", str(tf_path),
            "--format", "json"
        ]
        res = subprocess.run(cmd_uniflow, capture_output=True, text=True)
        # Should reject with exit code 3 (SchemaMismatchError)
        assert res.returncode == 3
        assert "Uniflow representation detected" in res.stderr
    finally:
        os.unlink(tf_path)
