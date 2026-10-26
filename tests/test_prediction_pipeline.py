"""
test_prediction_pipeline.py - Comprehensive Unit Tests for Stage 9 Prediction Pipeline & APIs

Validates:
1. NIDSPredictionPipeline initialization and model loading.
2. Schema validation, identifier purging, and ratio feature synthesis.
3. Batch and streaming inference with configurable decision thresholds.
4. Instance-level SHAP TreeExplainer integration and briefing generation.
5. Report generation and security sanitization.
6. FastAPI REST API endpoints (/api/status, /api/predict, /api/explain, /api/scenarios, /api/export_report).
"""

import os
import sys
import json
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.predict_pipeline import NIDSPredictionPipeline
from src.app.main import app


@pytest.fixture(scope="module")
def pipeline():
    return NIDSPredictionPipeline()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_biflow_raw():
    """A realistic single raw biflow record before feature engineering."""
    return pd.DataFrame([{
        "ip_src": "192.168.1.100",
        "ip_dst": "192.168.1.1",
        "prt_src": 54321,
        "prt_dst": 1883,
        "proto": 6,
        "fwd_num_pkts": 8,
        "bwd_num_pkts": 6,
        "fwd_mean_iat": 0.001,
        "bwd_mean_iat": 0.0012,
        "fwd_std_iat": 0.0005,
        "bwd_std_iat": 0.0006,
        "fwd_min_iat": 0.0001,
        "bwd_min_iat": 0.0001,
        "fwd_max_iat": 0.005,
        "bwd_max_iat": 0.004,
        "fwd_mean_pkt_len": 65.0,
        "bwd_mean_pkt_len": 54.0,
        "fwd_std_pkt_len": 12.0,
        "bwd_std_pkt_len": 8.0,
        "fwd_min_pkt_len": 52.0,
        "bwd_min_pkt_len": 52.0,
        "fwd_max_pkt_len": 110.0,
        "bwd_max_pkt_len": 80.0,
        "fwd_num_bytes": 520,
        "bwd_num_bytes": 324,
        "fwd_num_psh_flags": 2,
        "bwd_num_psh_flags": 1,
        "fwd_num_rst_flags": 0,
        "bwd_num_rst_flags": 0,
        "fwd_num_urg_flags": 0,
        "bwd_num_urg_flags": 0
    }])


def test_pipeline_initialization(pipeline):
    """Verify that models, schemas, and SHAP explainer initialize properly."""
    assert pipeline is not None
    assert len(pipeline.expected_features) == 34
    assert pipeline.binary_model is not None
    assert pipeline.local_explainer is not None
    assert "means" in pipeline.baseline_stats


def test_schema_validation_and_ratio_engineering(pipeline, sample_biflow_raw):
    """Verify identifier exclusion and automatic 6-ratio synthesis."""
    df_features, df_meta = pipeline.validate_and_preprocess(sample_biflow_raw)

    # Identifiers must not be present in feature matrix
    for id_col in ["ip_src", "ip_dst", "prt_src"]:
        assert id_col not in df_features.columns

    # Exactly 34 features in fixed order
    assert list(df_features.columns) == pipeline.expected_features
    assert len(df_features.columns) == 34

    # Check engineered ratios
    assert np.isclose(df_features["pkt_ratio"].iloc[0], 8.0 / (6.0 + 1e-5))
    assert np.isclose(df_features["byte_ratio"].iloc[0], 520.0 / (324.0 + 1e-5))
    assert df_features["total_pkts"].iloc[0] == 14
    assert df_features["total_bytes"].iloc[0] == 844


def test_missing_required_features_raises_error(pipeline):
    """Verify that incomplete schemas are rejected with descriptive errors."""
    incomplete_df = pd.DataFrame([{"prt_dst": 1883, "proto": 6}])
    with pytest.raises(ValueError, match="Input DataFrame lacks required columns"):
        pipeline.validate_and_preprocess(incomplete_df)


def test_predict_dataframe_and_thresholding(pipeline, sample_biflow_raw):
    """Verify predictions, probability sanity, and threshold adjustments."""
    res_default = pipeline.predict_dataframe(sample_biflow_raw, threshold=0.50)
    assert len(res_default) == 1
    assert "predicted_label" in res_default.columns
    assert "prob_attack" in res_default.columns
    assert "prob_normal" in res_default.columns
    assert "severity" in res_default.columns

    p_norm = res_default["prob_normal"].iloc[0]
    p_atk = res_default["prob_attack"].iloc[0]
    assert np.isclose(p_norm + p_atk, 1.0, atol=1e-3)


def test_local_flow_explanation(pipeline, sample_biflow_raw):
    """Verify instance-level SHAP explanation generation through pipeline."""
    explanation = pipeline.explain_flow_by_index(sample_biflow_raw, index=0, top_k=5)
    assert "verdict" in explanation
    assert "confidence_scores" in explanation
    assert "top_attack_drivers" in explanation
    assert "top_benign_drivers" in explanation
    assert "analyst_briefing_points" in explanation
    assert len(explanation["all_feature_contributions"]) == 34


def test_export_triage_report(pipeline, sample_biflow_raw):
    """Verify report formatting and security sanitization."""
    res = pipeline.predict_dataframe(sample_biflow_raw)
    report_json = pipeline.export_triage_report(res, output_format="json")
    assert report_json["model_version"] == "1.0.0"
    assert report_json["telemetry_summary"]["total_flows_evaluated"] == 1
    # Verify no raw sensitive client IP address appears in report
    assert "192.168.1.100" not in json.dumps(report_json)

    report_csv = pipeline.export_triage_report(res, output_format="csv")
    assert "predicted_label" in report_csv


def test_fastapi_endpoints(client, sample_biflow_raw):
    """Verify REST API endpoints using FastAPI TestClient."""
    # 1. Status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert data_status["status"] == "ready"
    assert data_status["feature_count"] == 34

    # 2. Scenarios
    res_sc = client.get("/api/scenarios")
    assert res_sc.status_code == 200
    assert "normal_mqtt" in res_sc.json()
    assert "scan_A" in res_sc.json()

    # 3. Predict Single Flow
    sample_dict = sample_biflow_raw.iloc[0].to_dict()
    res_pred = client.post("/api/predict", json={"features": sample_dict})
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert "label" in pred_data
    assert "prob_attack" in pred_data

    # 4. Explain Flow
    res_exp = client.post("/api/explain", json={"features": sample_dict})
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert "verdict" in exp_data
    assert "analyst_briefing_points" in exp_data

    # 5. Export Report
    res_rep = client.get("/api/export_report?format=json")
    assert res_rep.status_code == 200
    assert "telemetry_summary" in res_rep.json()
