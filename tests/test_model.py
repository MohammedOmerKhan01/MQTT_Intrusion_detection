"""
test_model.py - Automated Stage 6 Tests for Random Forest Models & Evaluation
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.models.persistence import load_model_artifact


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def val_data(base_dir):
    val_path = base_dir / "data" / "processed" / "biflow_val.parquet"
    assert val_path.exists(), f"Missing {val_path}"
    df_val = pd.read_parquet(val_path)
    feature_cols = [c for c in df_val.columns if not c.startswith("target_")]
    X_val = df_val[feature_cols].iloc[:200]
    y_val = df_val["target_is_attack"].iloc[:200]
    return X_val, y_val


def test_saved_binary_model_integrity(base_dir, val_data):
    """Verify saved binary Random Forest loads cleanly and computes valid probabilities."""
    model_path = base_dir / "artifacts" / "models" / "rf_binary_model.joblib"
    assert model_path.exists(), f"Binary model not found at {model_path}"

    model = load_model_artifact(model_path)
    X_val, _ = val_data

    preds = model.predict(X_val)
    probas = model.predict_proba(X_val)

    assert len(preds) == len(X_val)
    assert probas.shape == (len(X_val), 2)
    # Probabilities sum to 1.0
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert set(preds).issubset({0, 1})


def test_saved_multiclass_model_integrity(base_dir, val_data):
    """Verify saved multiclass model outputs valid 5-class predictions."""
    model_path = base_dir / "artifacts" / "models" / "rf_multiclass_model.joblib"
    assert model_path.exists(), f"Multiclass model not found at {model_path}"

    model = load_model_artifact(model_path)
    X_val, _ = val_data

    preds = model.predict(X_val)
    probas = model.predict_proba(X_val)

    assert len(preds) == len(X_val)
    assert probas.shape == (len(X_val), 5)
    expected_classes = {"mqtt_bruteforce", "normal", "scan_A", "scan_sU", "sparta"}
    assert set(model.classes_) == expected_classes
    assert set(preds).issubset(expected_classes)


def test_training_summary_metrics(base_dir):
    """Verify model training summary JSON report contains all required metrics."""
    summary_path = base_dir / "artifacts" / "reports" / "model_training_summary.json"
    assert summary_path.exists(), f"Summary report not found at {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "baseline_binary" in data
    assert "tuned_binary" in data
    assert "multiclass" in data

    base_m = data["baseline_binary"]
    assert "false_positive_rate" in base_m
    assert "false_negative_rate" in base_m
    assert "roc_auc" in base_m
    assert "pr_auc" in base_m
    assert base_m["roc_auc"] > 0.99
