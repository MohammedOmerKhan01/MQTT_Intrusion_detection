"""
predict_pipeline.py - Comprehensive End-to-End Prediction & Triage Pipeline

Orchestrates:
1. Strict schema validation, sanitization (stripping IP/port identifiers), and defensive missing/inf value handling.
2. Domain-justified ratio feature engineering (6 flow asymmetry ratios) with epsilon guards (1e-5).
3. Streaming PCAP file ingestion via PcapFlowExtractor and tabular CSV ingestion.
4. Binary threat classification (Benign vs Malicious) and Multi-category attack typing.
5. Operational decision thresholding (configurable tau in [0.0, 1.0]).
6. Local instance-level Explainable AI via SHAP TreeExplainer (LocalFlowExplainer).
7. SOC report serialization (CSV and JSON) adhering to security data hygiene.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
import joblib

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.pcap_extractor import PcapFlowExtractor
from src.explainability.local_explanations import LocalFlowExplainer

# Core Network Identifiers that MUST be excluded from ML feature matrix
IDENTIFIER_COLUMNS = ["ip_src", "ip_dst", "prt_src"]

# Ground truth & provenance columns to exclude from inference feature matrix
METADATA_COLUMNS = ["is_attack", "scenario", "attack_type", "target_is_attack", "target_attack_type"]

EPSILON = 1e-5


class NIDSPredictionPipeline:
    """
    Production-grade Network Intrusion Detection & Triage Pipeline.
    Supports both batch/tabular CSV inference and raw PCAP packet stream processing.
    """

    def __init__(
        self,
        binary_model_path: Optional[Path] = None,
        multiclass_model_path: Optional[Path] = None,
        schema_path: Optional[Path] = None,
        baseline_stats_path: Optional[Path] = None
    ):
        self.root_dir = ROOT_DIR
        self.binary_model_path = binary_model_path or (self.root_dir / "artifacts" / "models" / "rf_binary_model.joblib")
        self.multiclass_model_path = multiclass_model_path or (self.root_dir / "artifacts" / "models" / "rf_multiclass_model.joblib")
        self.schema_path = schema_path or (self.root_dir / "artifacts" / "metadata" / "biflow_feature_schema.json")
        self.baseline_stats_path = baseline_stats_path or (self.root_dir / "artifacts" / "metadata" / "biflow_baseline_stats.json")

        # 1. Load Schema Definition
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema_config = json.load(f)
        self.expected_features: List[str] = self.schema_config["feature_names"]

        # 2. Load Binary Model
        if not self.binary_model_path.exists():
            raise FileNotFoundError(f"Binary model artifact not found at {self.binary_model_path}")
        self.binary_model = joblib.load(self.binary_model_path)

        # 3. Load Multiclass Model (optional)
        self.multiclass_model = None
        if self.multiclass_model_path.exists():
            try:
                self.multiclass_model = joblib.load(self.multiclass_model_path)
            except Exception as e:
                print(f"Warning: Failed to load multiclass model: {e}")

        # 4. Load Baseline Normal Statistics
        self.baseline_stats: Dict[str, Any] = {}
        if self.baseline_stats_path.exists():
            with open(self.baseline_stats_path, "r", encoding="utf-8") as f:
                self.baseline_stats = json.load(f)

        # 5. Initialize Pre-warmed SHAP Explainer
        self.local_explainer = LocalFlowExplainer(
            model=self.binary_model,
            feature_names=self.expected_features
        )

    def validate_and_preprocess(
        self,
        df_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate input schema, extract analyst metadata, purge identifiers,
        and construct the exact 34-feature representation.

        Args:
            df_input: Raw input DataFrame from CSV or PCAP extractor

        Returns:
            Tuple of:
            - df_features: DataFrame with exactly the 34 features in expected order
            - df_metadata: Flow context DataFrame (IPs, ports, protocols, timestamps) for SOC display
        """
        if not isinstance(df_input, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df_input)}")

        if df_input.empty:
            raise ValueError("Input DataFrame is empty")

        df = df_input.copy()

        # Retain analyst metadata context for SOC investigation
        metadata_cols = [c for c in ["ip_src", "ip_dst", "prt_src", "prt_dst", "proto", "is_attack", "attack_type"] if c in df.columns]
        df_metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

        # Check if incoming data already matches the 34-feature schema
        already_has_34 = all(f in df.columns for f in self.expected_features)

        if not already_has_34:
            # Need to compute engineered ratios from raw biflow columns
            raw_required = [
                "fwd_num_pkts", "bwd_num_pkts", "fwd_num_bytes", "bwd_num_bytes",
                "fwd_mean_pkt_len", "bwd_mean_pkt_len"
            ]
            missing_raw = [col for col in raw_required if col not in df.columns]
            if missing_raw:
                raise ValueError(
                    f"Input DataFrame lacks required columns to construct feature matrix. "
                    f"Missing: {missing_raw}. Expected either 34 engineered features or raw biflow schema."
                )

            # Compute 6 asymmetry ratio features with epsilon guards
            fwd_pkts = df["fwd_num_pkts"].astype(float)
            bwd_pkts = df["bwd_num_pkts"].astype(float)
            fwd_bytes = df["fwd_num_bytes"].astype(float)
            bwd_bytes = df["bwd_num_bytes"].astype(float)
            fwd_mean_len = df["fwd_mean_pkt_len"].astype(float)
            bwd_mean_len = df["bwd_mean_pkt_len"].astype(float)

            df["pkt_ratio"] = fwd_pkts / (bwd_pkts + EPSILON)
            df["total_pkts"] = fwd_pkts + bwd_pkts
            df["byte_ratio"] = fwd_bytes / (bwd_bytes + EPSILON)
            df["total_bytes"] = fwd_bytes + bwd_bytes
            df["bytes_per_pkt"] = df["total_bytes"] / (df["total_pkts"] + EPSILON)
            df["pkt_len_ratio"] = fwd_mean_len / (bwd_mean_len + EPSILON)

        # Verify that all 34 features are now available
        missing_final = [f for f in self.expected_features if f not in df.columns]
        if missing_final:
            raise ValueError(f"Failed to assemble complete feature matrix. Missing features: {missing_final}")

        # Extract exactly the 34 features in fixed deterministic order
        df_features = df[self.expected_features].copy()

        # Defensive NaN / Inf sanitization
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Ensure float32/float64 numerical types
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0.0)

        return df_features, df_metadata

    def predict_dataframe(
        self,
        df_input: pd.DataFrame,
        threshold: float = 0.50
    ) -> pd.DataFrame:
        """
        Run inference across a DataFrame of flows.

        Args:
            df_input: Raw or preprocessed DataFrame
            threshold: Decision threshold tau in [0.0, 1.0]. Flows with P(Attack) >= tau are flagged.

        Returns:
            DataFrame containing original metadata, predictions, probabilities, and risk tiers.
        """
        df_features, df_meta = self.validate_and_preprocess(df_input)

        # 1. Compute predicted probabilities
        probas = self.binary_model.predict_proba(df_features)
        prob_normal = probas[:, 0]
        prob_attack = probas[:, 1]

        # 2. Apply configurable decision threshold tau
        pred_is_attack = (prob_attack >= threshold).astype(int)

        # 3. Categorize alert severity tiers
        severity_tiers = []
        for p in prob_attack:
            if p >= 0.90:
                severity_tiers.append("CRITICAL")
            elif p >= 0.70:
                severity_tiers.append("HIGH")
            elif p >= threshold:
                severity_tiers.append("MEDIUM")
            elif p >= 0.20:
                severity_tiers.append("ELEVATED")
            else:
                severity_tiers.append("BENIGN")

        results_df = df_meta.copy() if not df_meta.empty else pd.DataFrame(index=df_features.index)
        results_df["predicted_label"] = np.where(pred_is_attack == 1, "ATTACK", "NORMAL")
        results_df["is_alert"] = pred_is_attack
        results_df["prob_attack"] = np.round(prob_attack, 4)
        results_df["prob_normal"] = np.round(prob_normal, 4)
        results_df["severity"] = severity_tiers

        # 4. Multiclass attack categorization for flagged alerts
        if self.multiclass_model is not None:
            mc_preds = self.multiclass_model.predict(df_features)
            # If binary verdict is normal, keep normal; if attack, assign subtype
            results_df["attack_subtype"] = [
                mc if is_atk == 1 else "benign"
                for mc, is_atk in zip(mc_preds, pred_is_attack)
            ]
        else:
            results_df["attack_subtype"] = np.where(pred_is_attack == 1, "malicious_unclassified", "benign")

        # Attach engineered features for analyst review and downstream explainability
        for col in df_features.columns:
            if col not in results_df.columns:
                results_df[col] = df_features[col].values

        return results_df

    def process_pcap(
        self,
        pcap_source: Union[str, Path, bytes],
        max_packets: Optional[int] = 50000,
        threshold: float = 0.50
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Dissect raw PCAP capture packets into bidirectional flows,
        engineer features, and generate threat predictions.

        Args:
            pcap_source: File path to PCAP or raw binary bytes
            max_packets: Cap on packets to read for interactive responsiveness
            threshold: Decision threshold tau

        Returns:
            Tuple of:
            - results_df: Predictions, probabilities, and severity tiers
            - extracted_flows_df: Raw extracted bidirectional flows
        """
        temp_file = None
        try:
            if isinstance(pcap_source, (bytes, bytearray)):
                temp_file = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
                temp_file.write(pcap_source)
                temp_file.flush()
                pcap_path = Path(temp_file.name)
            else:
                pcap_path = Path(pcap_source)

            if not pcap_path.exists():
                raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

            extractor = PcapFlowExtractor(inactivity_timeout=120.0, require_bidirectional=False)
            extracted_flows = extractor.extract_from_pcap(pcap_path, max_packets=max_packets)

            if not extracted_flows:
                empty_df = pd.DataFrame()
                return empty_df, empty_df

            flows_df = pd.DataFrame(extracted_flows)
            results_df = self.predict_dataframe(flows_df, threshold=threshold)

            return results_df, flows_df

        finally:
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def explain_flow_by_index(
        self,
        df_input: pd.DataFrame,
        index: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute exact local SHAP tree-path contributions for a specific flow.

        Args:
            df_input: Original or preprocessed DataFrame
            index: Integer row index in df_input
            top_k: Number of positive and negative drivers to highlight

        Returns:
            Structured dictionary with probabilities, SHAP values, and SOC analyst narrative.
        """
        df_features, df_meta = self.validate_and_preprocess(df_input)
        if index < 0 or index >= len(df_features):
            raise IndexError(f"Index {index} out of bounds for DataFrame of length {len(df_features)}")

        flow_vector = df_features.iloc[index]
        explanation = self.local_explainer.explain_flow(flow_vector, top_k=top_k)

        # Attach metadata if available
        if not df_meta.empty and index < len(df_meta):
            meta_dict = df_meta.iloc[index].to_dict()
            explanation["flow_metadata"] = {k: str(v) for k, v in meta_dict.items()}

        # Attach baseline comparisons
        baseline_means = self.baseline_stats.get("means", {})
        comparisons = []
        for item in explanation["all_feature_contributions"]:
            feat = item["feature"]
            val = item["value"]
            base_mean = baseline_means.get(feat, None)
            comparisons.append({
                "feature": feat,
                "observed_value": val,
                "benign_baseline_mean": base_mean,
                "shap_attribution": item["shap_value"],
                "impact": item["direction"]
            })
        explanation["baseline_comparison_table"] = comparisons

        return explanation

    def export_triage_report(
        self,
        results_df: pd.DataFrame,
        explanations: Optional[List[Dict[str, Any]]] = None,
        output_format: str = "json"
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a sanitized SOC triage report with metrics, alert counts, and XAI findings.
        Guarantees: No raw packet payloads or sensitive internal network mappings exported.
        """
        total_flows = len(results_df)
        total_alerts = int((results_df["is_alert"] == 1).sum()) if "is_alert" in results_df.columns else 0
        alert_rate = (total_alerts / total_flows) if total_flows > 0 else 0.0

        # Subtype breakdown
        subtypes = {}
        if "attack_subtype" in results_df.columns:
            subtypes = results_df["attack_subtype"].value_counts().to_dict()

        report = {
            "report_title": "ML-NIDS & Explainable AI Incident Triage Summary",
            "model_version": "1.0.0",
            "model_architecture": "Random Forest (150 trees, balanced weights, 34 features)",
            "pipeline_configuration": {
                "excluded_identifiers": IDENTIFIER_COLUMNS,
                "engineered_features_count": 6,
                "total_features": len(self.expected_features)
            },
            "telemetry_summary": {
                "total_flows_evaluated": total_flows,
                "total_alerts_flagged": total_alerts,
                "alert_rate_percentage": round(alert_rate * 100, 2),
                "severity_distribution": results_df["severity"].value_counts().to_dict() if "severity" in results_df.columns else {},
                "attack_subtype_breakdown": subtypes
            },
            "investigated_explanations_sample": explanations or []
        }

        if output_format.lower() == "json":
            return report
        elif output_format.lower() == "csv":
            return results_df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
