"""
test_explainability.py - Automated Stage 8 Tests for SHAP Global & Local XAI
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.models.persistence import load_model_artifact
from src.explainability.local_explanations import LocalFlowExplainer


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def loaded_model_and_val_data(base_dir):
    model_path = base_dir / "artifacts" / "models" / "rf_binary_model.joblib"
    val_path = base_dir / "data" / "processed" / "biflow_val.parquet"
    assert model_path.exists(), f"Missing {model_path}"
    assert val_path.exists(), f"Missing {val_path}"

    model = load_model_artifact(model_path)
    df_val = pd.read_parquet(val_path)
    feat_cols = [c for c in df_val.columns if not c.startswith("target_")]
    return model, df_val, feat_cols


def test_local_flow_explainer_additivity(loaded_model_and_val_data):
    """Verify local SHAP values strictly satisfy the Cooperative Game Theory Additivity Axiom."""
    model, df_val, feat_cols = loaded_model_and_val_data
    explainer = LocalFlowExplainer(model, feat_cols)

    sample = df_val.iloc[0]
    exp = explainer.explain_flow(sample)

    prob_attack = exp["confidence_scores"]["attack_probability"]
    base_val = exp["base_value_attack"]
    total_shap = sum(c["shap_value"] for c in exp["all_feature_contributions"])

    # Additivity Axiom: base_value + sum(shap_values) == model_output
    assert abs((base_val + total_shap) - prob_attack) < 1e-4, "SHAP additivity violated!"


def test_local_explanation_structure(loaded_model_and_val_data):
    """Verify local explanation dictionary structure and narrative briefing points."""
    model, df_val, feat_cols = loaded_model_and_val_data
    explainer = LocalFlowExplainer(model, feat_cols)

    # Test attack sample
    attack_sample = df_val[df_val["target_is_attack"] == 1].iloc[0]
    exp_attack = explainer.explain_flow(attack_sample)

    assert exp_attack["predicted_class"] == 1
    assert "MALICIOUS" in exp_attack["verdict"]
    assert len(exp_attack["top_attack_drivers"]) > 0
    assert len(exp_attack["analyst_briefing_points"]) > 0
    assert "disclaimer" in exp_attack


def test_shap_artifacts_exist(base_dir):
    """Verify that all global and local SHAP reports and figures were produced."""
    reports_dir = base_dir / "artifacts" / "reports"
    assert (reports_dir / "shap_global_importance.json").exists()
    assert (reports_dir / "local_shap_demonstration.json").exists()

    figures_dir = base_dir / "artifacts" / "figures"
    expected_figures = [
        "shap_global_bar.png",
        "shap_summary_beeswarm.png",
        "feature_importance_vs_shap.png",
        "local_shap_alert.png",
        "local_shap_normal.png"
    ]
    for fig in expected_figures:
        p = figures_dir / fig
        assert p.exists(), f"Missing figure: {fig}"
        assert p.stat().st_size > 0, f"Figure {fig} is empty"
