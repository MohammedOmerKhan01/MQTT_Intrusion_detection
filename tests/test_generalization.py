"""
test_generalization.py - Automated Stage 7 Tests for Overfitting Prevention & Generalization
"""

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_generalization_report_integrity(base_dir):
    """Verify generalization audit report was produced and passed criteria."""
    report_file = base_dir / "artifacts" / "reports" / "generalization_audit.json"
    assert report_file.exists(), f"Missing audit report: {report_file}"

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Train-Val Generalization Gap <= 1.0% (0.01)
    gap_data = data["train_val_gap"]
    assert gap_data["accuracy_gap"] < 0.01, f"Generalization gap too wide: {gap_data['accuracy_gap']}"
    assert gap_data["verdict"] == "ACCEPTABLE_GENERALIZATION"

    # 2. OOD Zero-Day Recall >= 99%
    ood_data = data["ood_zero_day_generalization"]
    assert ood_data["recall"] >= 0.99, f"OOD Recall dropped below 99%: {ood_data['recall']}"
    assert ood_data["confusion_matrix"]["fn"] <= 10

    # 3. Port Ablation (Pure Behavioral) F1 >= 99%
    ablation_data = data["port_ablation_study"]
    assert ablation_data["f1_score"] >= 0.99, f"Pure behavioral F1 dropped: {ablation_data['f1_score']}"


def test_generalization_figures_exist(base_dir):
    """Verify diagnostic visualization figures exist and are non-empty."""
    fig_dir = base_dir / "artifacts" / "figures"
    expected = [
        "train_val_gap.png",
        "ood_scenario_evaluation.png",
        "port_ablation_comparison.png"
    ]
    for fig in expected:
        p = fig_dir / fig
        assert p.exists(), f"Missing figure {fig}"
        assert p.stat().st_size > 0, f"Figure {fig} is empty"
