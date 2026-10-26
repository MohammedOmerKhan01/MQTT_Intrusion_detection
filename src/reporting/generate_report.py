"""
generate_report.py - Automated Academic Report & Telemetry Aggregator

Collects all verified experimental metrics, figures, and model metadata
to generate executive summaries and export markdown/JSON reports.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def compile_full_experiment_telemetry(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    root = base_dir or ROOT_DIR
    meta_dir = root / "artifacts" / "metadata"
    rep_dir = root / "artifacts" / "reports"

    telemetry = {
        "project_name": "ML-NIDS with Random Forest and Explainable AI (SHAP)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": "Antigravity Senior Cybersecurity & ML Research Team",
        "evaluation_summary": {}
    }

    # Load Model Manifest
    manifest_path = meta_dir / "model_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            telemetry["manifest"] = json.load(f)

    # Load Robustness Audit
    robust_path = rep_dir / "robustness_audit.json"
    if robust_path.exists():
        with open(robust_path, "r", encoding="utf-8") as f:
            telemetry["robustness"] = json.load(f)

    # Load Generalization Audit
    gen_path = rep_dir / "generalization_audit.json"
    if gen_path.exists():
        with open(gen_path, "r", encoding="utf-8") as f:
            telemetry["generalization"] = json.load(f)

    # Load SHAP Global Importance
    shap_path = rep_dir / "shap_global_importance.json"
    if shap_path.exists():
        with open(shap_path, "r", encoding="utf-8") as f:
            telemetry["shap_global"] = json.load(f)

    return telemetry


def save_compiled_summary(output_path: Optional[Path] = None, base_dir: Optional[Path] = None) -> Path:
    root = base_dir or ROOT_DIR
    out = output_path or (root / "artifacts" / "reports" / "full_experiment_telemetry.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = compile_full_experiment_telemetry(base_dir=root)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out


if __name__ == "__main__":
    out_path = save_compiled_summary()
    print(f"Compiled full experiment telemetry to: {out_path}")
