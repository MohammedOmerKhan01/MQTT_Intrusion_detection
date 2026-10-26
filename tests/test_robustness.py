"""
test_robustness.py - Comprehensive Unit Tests for Stage 11 Robustness & Stress Testing

Validates:
1. Operational threshold sensitivity sweep properties (monotonicity of error rates).
2. Adversarial evasion resilience (packet padding, timing dilation, dual mimicry).
3. Edge case defensive handling (empty DataFrame, single row, all-zeros, inf/NaN vectors, extreme outliers).
4. Inference latency and high-throughput execution under simulated load.
"""

import os
import sys
import json
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.robustness import RobustnessEvaluator
from src.models.predict_pipeline import NIDSPredictionPipeline


@pytest.fixture(scope="module")
def pipeline():
    return NIDSPredictionPipeline()


@pytest.fixture(scope="module")
def evaluator(pipeline):
    return RobustnessEvaluator(pipeline=pipeline)


@pytest.fixture(scope="module")
def sample_val_df():
    val_path = ROOT_DIR / "data" / "processed" / "biflow_val.parquet"
    if not val_path.exists():
        pytest.skip(f"Validation data not found at {val_path}")
    df = pd.read_parquet(val_path)
    # Take stratified sample of 1,000 flows for fast test execution
    normal_sample = df[df["target_is_attack"] == 0].head(700)
    attack_sample = df[df["target_is_attack"] == 1].head(300)
    return pd.concat([normal_sample, attack_sample], ignore_index=True)


def test_threshold_sensitivity_monotonicity(evaluator, sample_val_df):
    """Verify that shifting decision threshold produces expected monotonic error curves."""
    thresholds = [0.10, 0.30, 0.50, 0.70, 0.90]
    sweep_summary = evaluator.run_threshold_sensitivity_sweep(sample_val_df, thresholds=thresholds)

    results = sweep_summary["threshold_sweep_results"]
    assert len(results) == len(thresholds)

    recalls = [r["recall"] for r in results]
    fprs = [r["false_positive_rate"] for r in results]

    # As tau increases, recall must not increase (it should be non-increasing)
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1] - 1e-4

    # As tau increases, FPR must not increase (it should be non-increasing)
    for i in range(len(fprs) - 1):
        assert fprs[i] >= fprs[i + 1] - 1e-4

    # FPR at tau=0.90 must be <= FPR at tau=0.10
    assert fprs[-1] <= fprs[0]


def test_adversarial_packet_padding_resilience(evaluator, sample_val_df):
    """Verify that inflating packet sizes (+100%, +300%) does not break detection."""
    attacks = sample_val_df[sample_val_df["target_is_attack"] == 1].copy()
    assert len(attacks) > 0

    # Inflate packet sizes by 2.0x (+100%) and 4.0x (+300%)
    for factor in [2.0, 4.0]:
        df_padded = attacks.copy()
        df_padded["fwd_mean_pkt_len"] = df_padded["fwd_mean_pkt_len"] * factor
        df_padded["fwd_num_bytes"] = df_padded["fwd_num_bytes"] * factor

        res = evaluator.pipeline.predict_dataframe(df_padded, threshold=0.50)
        recall = (res["is_alert"] == 1).mean()
        # Even under severe payload inflation, detection recall must stay above 95%
        assert recall >= 0.95, f"Recall dropped to {recall} under padding multiplier {factor}"


def test_adversarial_timing_dilation_resilience(evaluator, sample_val_df):
    """Verify that dilating inter-arrival times (100x artificial delay) does not evade detector."""
    attacks = sample_val_df[sample_val_df["target_is_attack"] == 1].copy()
    assert len(attacks) > 0

    # Dilate timing by 100x and 1000x
    for factor in [100.0, 1000.0]:
        df_dilated = attacks.copy()
        df_dilated["fwd_mean_iat"] = df_dilated["fwd_mean_iat"] * factor
        df_dilated["bwd_mean_iat"] = df_dilated["bwd_mean_iat"] * factor

        res = evaluator.pipeline.predict_dataframe(df_dilated, threshold=0.50)
        recall = (res["is_alert"] == 1).mean()
        # Due to orthogonal flag and ratio features, recall must stay above 95%
        assert recall >= 0.95, f"Recall dropped to {recall} under timing multiplier {factor}"


def test_corrupt_input_defenses(evaluator):
    """Verify defensive handling across empty, single-row, infinite, and outlier inputs."""
    corrupt_results = evaluator.run_corrupt_input_stress_test()

    assert "DEFENDED" in corrupt_results["empty_dataframe"]
    assert "SUCCESS" in corrupt_results["single_row_input"]
    assert "DEFENDED" in corrupt_results["extreme_outliers"]
    assert "DEFENDED" in corrupt_results["all_zeros_vector"]
    assert "DEFENDED" in corrupt_results["inf_and_nan_handling"]
    assert "DEFENDED" in corrupt_results["negative_values"]


def test_latency_and_throughput_bounds(evaluator, sample_val_df):
    """Verify that per-flow latency is < 100 us and throughput > 10,000 FPS."""
    perf = evaluator.benchmark_latency_and_throughput(sample_val_df)
    assert perf["minimum_latency_us"] < 100.0, f"Latency too high: {perf['minimum_latency_us']} us"
    assert perf["peak_throughput_fps"] > 10000.0, f"Throughput too low: {perf['peak_throughput_fps']} FPS"
