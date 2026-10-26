"""
local_explanations.py - Local Instance-Level XAI Engine using SHAP TreeExplainer

Generates local explanations for individual network flows:
1. Exact Shapley values phi_i(x) for each feature
2. Base value E[f(x)] and model predicted probabilities P(Attack|x)
3. Positive force contributors (increasing attack risk) vs. Negative force contributors (supporting benign)
4. Plain-English narrative analyst briefing for SOC incident investigation
5. Local SHAP force / waterfall visualization generation
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.persistence import load_model_artifact


class LocalFlowExplainer:
    """Pre-warmed local SHAP explainer for single-flow inference and alert triage."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        
        # Determine base value for Class 1 (Attack)
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) == 2:
            self.base_value_attack = float(ev[1])
        else:
            self.base_value_attack = float(ev)

    def explain_flow(
        self,
        flow_vector: Union[pd.Series, pd.DataFrame, np.ndarray],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute local SHAP attribution and generate a structured SOC analyst briefing.

        Args:
            flow_vector: 1D Series or 1-row DataFrame containing the 34 features
            top_k: Number of top driving features to highlight in briefing

        Returns:
            Dict containing predicted probabilities, base value, feature attributions, and narrative.
        """
        if isinstance(flow_vector, pd.Series):
            df_in = pd.DataFrame([flow_vector])[self.feature_names]
        elif isinstance(flow_vector, pd.DataFrame):
            df_in = flow_vector[self.feature_names].iloc[[0]]
        else:
            df_in = pd.DataFrame([flow_vector], columns=self.feature_names)

        # 1. Model Predictions
        probas = self.model.predict_proba(df_in)[0]
        prob_normal = float(probas[0])
        prob_attack = float(probas[1])
        predicted_class = 1 if prob_attack >= 0.5 else 0

        # 2. Compute exact local SHAP values
        shap_res = self.explainer(df_in)
        if len(shap_res.values.shape) == 3 and shap_res.values.shape[2] == 2:
            phi_attack = shap_res.values[0, :, 1]
        else:
            phi_attack = shap_res.values[0]

        # 3. Format individual feature contributions
        contributions = []
        for idx, feat_name in enumerate(self.feature_names):
            val = float(df_in.iloc[0, idx])
            phi = float(phi_attack[idx])
            contributions.append({
                "feature": feat_name,
                "value": round(val, 4),
                "shap_value": round(phi, 5),
                "abs_shap": abs(round(phi, 5)),
                "direction": "INCREASES_ATTACK_RISK" if phi > 0.001 else ("SUPPORTS_NORMAL" if phi < -0.001 else "NEUTRAL")
            })

        # Sort descending by absolute SHAP contribution
        contributions.sort(key=lambda item: item["abs_shap"], reverse=True)

        top_attack_push = [c for c in contributions if c["shap_value"] > 0][:top_k]
        top_normal_push = [c for c in contributions if c["shap_value"] < 0][:top_k]

        # 4. Generate Plain-English Analyst Briefing
        narrative_points = []
        if predicted_class == 1:
            verdict = "MALICIOUS INTRUSION ALERT"
            executive_summary = (
                f"Flow flagged as MALICIOUS with {prob_attack*100:.1f}% estimated probability. "
                f"The baseline model prior was {self.base_value_attack*100:.1f}%, and observed features "
                f"pushed the posterior by +{(prob_attack - self.base_value_attack)*100:.1f}% toward an alert."
            )
            for item in top_attack_push:
                narrative_points.append(
                    f"Feature '{item['feature']}' observed at {item['value']} pushed model output toward Attack by +{item['shap_value']:.4f}."
                )
        else:
            verdict = "BENIGN TRAFFIC (NORMAL)"
            executive_summary = (
                f"Flow classified as BENIGN with {prob_normal*100:.1f}% confidence. "
                f"Model estimated attack probability is only {prob_attack*100:.1f}%."
            )
            for item in top_normal_push:
                narrative_points.append(
                    f"Feature '{item['feature']}' observed at {item['value']} reduced attack risk by {item['shap_value']:.4f} (supports Normal)."
                )

        return {
            "verdict": verdict,
            "predicted_class": predicted_class,
            "confidence_scores": {
                "normal_probability": round(prob_normal, 4),
                "attack_probability": round(prob_attack, 4)
            },
            "base_value_attack": round(self.base_value_attack, 4),
            "executive_summary": executive_summary,
            "top_attack_drivers": top_attack_push,
            "top_benign_drivers": top_normal_push,
            "analyst_briefing_points": narrative_points,
            "all_feature_contributions": contributions,
            "disclaimer": (
                "NOTE: SHAP contributions describe internal model feature attribution, "
                "not causal proof of malicious intent on the physical network."
            )
        }

    def plot_local_explanation(
        self,
        explanation_dict: Dict[str, Any],
        output_path: Path,
        top_display: int = 10
    ):
        """Render and save a local horizontal force bar plot."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        top_features = explanation_dict["all_feature_contributions"][:top_display]
        top_features = sorted(top_features, key=lambda x: x["shap_value"])

        names = [f"{f['feature']} = {f['value']}" for f in top_features]
        values = [f["shap_value"] for f in top_features]
        colors = ["#d9534f" if v > 0 else "#2b5c8f" for v in values]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="black", linestyle="--", alpha=0.6)
        ax.set_xlabel("Local SHAP Attribution (Contribution to Attack Probability)", fontsize=10)
        
        prob_atk = explanation_dict["confidence_scores"]["attack_probability"]
        verdict = explanation_dict["verdict"]
        ax.set_title(f"Local XAI Explanation: {verdict} (P(Attack) = {prob_atk*100:.1f}%)", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def run_local_explainability_demo(base_dir: Path) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING STAGE 8: LOCAL EXPLAINABLE AI DEMONSTRATION")
    print("=" * 70)

    model_path = base_dir / "artifacts" / "models" / "rf_binary_model.joblib"
    val_path = base_dir / "data" / "processed" / "biflow_val.parquet"

    model = load_model_artifact(model_path)
    df_val = pd.read_parquet(val_path)
    feat_cols = [c for c in df_val.columns if not c.startswith("target_")]

    local_explainer = LocalFlowExplainer(model, feat_cols)

    # 1. Explain an Attack Flow
    attack_sample = df_val[df_val["target_is_attack"] == 1].iloc[0]
    attack_exp = local_explainer.explain_flow(attack_sample)

    fig_dir = base_dir / "artifacts" / "figures"
    local_explainer.plot_local_explanation(attack_exp, fig_dir / "local_shap_alert.png")

    # 2. Explain a Normal Flow
    normal_sample = df_val[df_val["target_is_attack"] == 0].iloc[0]
    normal_exp = local_explainer.explain_flow(normal_sample)
    local_explainer.plot_local_explanation(normal_exp, fig_dir / "local_shap_normal.png")

    print("\n--- SAMPLE 1: MALICIOUS ATTACK FLOW ---")
    print(f"Verdict: {attack_exp['verdict']}")
    print(f"P(Attack): {attack_exp['confidence_scores']['attack_probability']*100:.1f}%")
    print("Executive Summary:\n ", attack_exp['executive_summary'])
    print("Top Attack Drivers:")
    for d in attack_exp["top_attack_drivers"][:3]:
        print(f"  * {d['feature']} = {d['value']} (SHAP: +{d['shap_value']:.4f})")

    print("\n--- SAMPLE 2: BENIGN NORMAL FLOW ---")
    print(f"Verdict: {normal_exp['verdict']}")
    print(f"P(Normal): {normal_exp['confidence_scores']['normal_probability']*100:.1f}%")
    print("Executive Summary:\n ", normal_exp['executive_summary'])
    print("Top Normal Drivers:")
    for d in normal_exp["top_benign_drivers"][:3]:
        print(f"  * {d['feature']} = {d['value']} (SHAP: {d['shap_value']:.4f})")

    demo_report = {
        "malicious_sample_explanation": attack_exp,
        "benign_sample_explanation": normal_exp,
        "figures_generated": ["local_shap_alert.png", "local_shap_normal.png"]
    }

    rep_path = base_dir / "artifacts" / "reports" / "local_shap_demonstration.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(demo_report, f, indent=2)

    print(f"\nLocal XAI report saved to: {rep_path}")
    print("=" * 70)
    print("STAGE 8 LOCAL EXPLAINABILITY DEMONSTRATION COMPLETE")
    print("=" * 70)

    return demo_report


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    run_local_explainability_demo(workspace_root)
