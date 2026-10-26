"""
global_explanations.py - Global Model Explainability Engine using SHAP

Computes:
1. Exact TreeExplainer SHAP values across a representative stratified sample
2. Global feature importance via Mean Absolute SHAP value E[|phi_j|]
3. SHAP Summary Beeswarm Plot (visualizing feature value impact on attack risk)
4. SHAP Feature Importance Bar Plot
5. Gini MDI vs. SHAP Importance Comparison (diagnosing tree split bias)
6. Export of global explanation metadata to artifacts/reports/shap_global_importance.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Ensure workspace root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.persistence import load_model_artifact


def compute_global_shap_explanations(
    model: Any,
    X_val: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """
    Compute TreeExplainer SHAP values on a representative stratified validation sample.

    Returns:
        shap_values_attack: np.ndarray of shape (sample_size, n_features) for Class 1
        base_value_attack: float baseline attack probability
        X_sample: pd.DataFrame of the explained samples
    """
    if len(X_val) > sample_size:
        X_sample = X_val.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X_val

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)

    # In shap >= 0.40, explanation.values has shape (N, M, 2) for binary classification
    if len(explanation.values.shape) == 3 and explanation.values.shape[2] == 2:
        shap_values_attack = explanation.values[:, :, 1]
        base_value_attack = float(explanation.base_values[0, 1])
    else:
        # Fallback if binary 1D output
        shap_values_attack = explanation.values
        base_value_attack = float(explanation.base_values[0])

    return shap_values_attack, base_value_attack, X_sample


def generate_global_shap_visualizations(
    shap_values_attack: np.ndarray,
    X_sample: pd.DataFrame,
    model: Any,
    output_dir: Path
) -> Dict[str, Any]:
    """Generate high-contrast publication-ready global SHAP plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    feature_names = list(X_sample.columns)

    # 1. Compute Mean Absolute SHAP Importance
    mean_abs_shap = np.mean(np.abs(shap_values_attack), axis=0)
    shap_ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # 2. Global SHAP Bar Plot (Top 15 Features)
    plt.figure(figsize=(10, 6))
    top15 = shap_ranking.head(15).sort_values(by="mean_abs_shap", ascending=True)
    plt.barh(top15["feature"], top15["mean_abs_shap"], color="#2b5c8f")
    plt.xlabel("Mean |SHAP Value| (Average Impact on Attack Probability)", fontsize=11)
    plt.title("Global Feature Importance (SHAP TreeExplainer - Attack Class)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_bar.png", dpi=300)
    plt.close()

    # 3. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(11, 7))
    # Create shap Explanation object for beeswarm plotting
    shap_exp = shap.Explanation(
        values=shap_values_attack,
        data=X_sample.values,
        feature_names=feature_names
    )
    shap.plots.beeswarm(shap_exp, max_display=15, show=False)
    plt.title("SHAP Beeswarm Summary: Feature Values vs. Impact on Attack Probability", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300)
    plt.close()

    # 4. Gini MDI vs. SHAP Comparison Plot (Diagnosing Feature Importance Bias)
    gini_importances = model.feature_importances_
    comp_df = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": gini_importances,
        "mean_abs_shap": mean_abs_shap
    })
    # Normalize both to percentage share
    comp_df["gini_pct"] = (comp_df["gini_importance"] / comp_df["gini_importance"].sum()) * 100
    comp_df["shap_pct"] = (comp_df["mean_abs_shap"] / comp_df["mean_abs_shap"].sum()) * 100
    top_comp = comp_df.sort_values(by="shap_pct", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(top_comp))
    width = 0.35
    ax.bar(x - width/2, top_comp["shap_pct"], width, label="SHAP Attribution (%)", color="#27ae60")
    ax.bar(x + width/2, top_comp["gini_pct"], width, label="Gini MDI Importance (%)", color="#2980b9")
    ax.set_ylabel("Relative Importance Share (%)", fontsize=11)
    ax.set_title("Feature Importance Diagnostic: SHAP Attribution vs. Random Forest Gini MDI", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(top_comp["feature"], rotation=30, ha="right", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_vs_shap.png", dpi=300)
    plt.close()

    return shap_ranking.to_dict(orient="records")


def run_global_explainability(base_dir: Path, sample_size: int = 1000) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING STAGE 8: GLOBAL EXPLAINABLE AI PIPELINE (SHAP TREEEXPLAINER)")
    print("=" * 70)

    model_path = base_dir / "artifacts" / "models" / "rf_binary_model.joblib"
    val_path = base_dir / "data" / "processed" / "biflow_val.parquet"

    print("\n[1/3] Loading fitted Random Forest model and validation dataset...")
    model = load_model_artifact(model_path)
    df_val = pd.read_parquet(val_path)
    feat_cols = [c for c in df_val.columns if not c.startswith("target_")]
    X_val = df_val[feat_cols]

    # 2. Compute SHAP values
    print(f"\n[2/3] Computing exact TreeExplainer SHAP values over {sample_size:,} validation flows...")
    t0 = time.time()
    shap_vals, base_val, X_sample = compute_global_shap_explanations(model, X_val, sample_size=sample_size)
    print(f"SHAP computation finished in {time.time()-t0:.2f}s.")
    print(f"  - SHAP Matrix Dimensions: {shap_vals.shape} ({shap_vals.shape[0]} samples x {shap_vals.shape[1]} features)")
    print(f"  - Expected Baseline Probability E[f(x)]: {base_val:.4f}")

    # 3. Generate Visualizations & Metadata
    print("\n[3/3] Generating Global SHAP Visualizations & Ranking Report...")
    fig_dir = base_dir / "artifacts" / "figures"
    rankings = generate_global_shap_visualizations(shap_vals, X_sample, model, fig_dir)

    rep_dir = base_dir / "artifacts" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_file = rep_dir / "shap_global_importance.json"

    report_data = {
        "explainer_type": "TreeExplainer",
        "shap_version": shap.__version__,
        "explained_samples": sample_size,
        "base_value_attack": round(base_val, 6),
        "top_features": rankings[:10],
        "all_rankings": rankings,
        "figures_generated": [
            "shap_global_bar.png",
            "shap_summary_beeswarm.png",
            "feature_importance_vs_shap.png"
        ]
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Global SHAP report saved to: {report_file}")
    print("=" * 70)
    print("TOP 5 GLOBAL INTRUSION PREDICTORS (SHAP ATTRIBUTION):")
    for i, item in enumerate(rankings[:5], 1):
        print(f"  {i}. {item['feature']:<22} (Mean |SHAP|: {item['mean_abs_shap']:.5f})")
    print("=" * 70)

    return report_data


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    run_global_explainability(workspace_root)
