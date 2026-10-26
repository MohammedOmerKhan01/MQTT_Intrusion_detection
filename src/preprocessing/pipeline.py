"""
pipeline.py - Leakage-Safe Feature Engineering & Preprocessing Pipeline

Constructs scikit-learn compatible ColumnTransformer and Pipeline components
for network flow tabular datasets (Uniflow and Biflow).

Guarantees:
1. Strict exclusion of network identifiers (ip_src, ip_dst, prt_src) to eliminate testbed memorization.
2. Separation of ground truth target (is_attack) and provenance tags (scenario, attack_type).
3. Addition of domain-justified traffic dynamics (asymmetry ratios, bytes-per-packet).
4. Full monotonicity preservation for tree-based algorithms (no redundant scaling).
5. Deterministic feature column ordering saved to artifacts/metadata/feature_schema.json.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.schema import get_biflow_columns, get_uniflow_columns


# Excluded identifier columns (testbed topology & client ephemeral state)
IDENTIFIER_COLUMNS = ["ip_src", "ip_dst", "prt_src"]

# Target and provenance metadata columns to exclude from feature matrix
METADATA_COLUMNS = ["is_attack", "scenario", "attack_type"]


class FlowFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects valid predictive features, rejects identifiers, and validates input schema.
    """

    def __init__(
        self,
        flow_type: str = "biflow",
        include_dst_port: bool = True,
        port_service_categorization: bool = True
    ):
        self.flow_type = flow_type
        self.include_dst_port = include_dst_port
        self.port_service_categorization = port_service_categorization
        self.selected_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        # Base candidate columns from input
        candidates = [c for c in X.columns if c not in IDENTIFIER_COLUMNS and c not in METADATA_COLUMNS]

        if not self.include_dst_port and "prt_dst" in candidates:
            candidates.remove("prt_dst")

        self.selected_columns_ = sorted(candidates)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        # Verify all selected columns exist in incoming dataframe
        missing = [c for c in self.selected_columns_ if c not in X.columns]
        if missing:
            raise KeyError(f"Input DataFrame is missing required feature columns: {missing}")

        X_out = X[self.selected_columns_].copy()

        # Handle infinite or NaN values defensively
        X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return X_out


class FlowRatioFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes domain-justified traffic symmetry and behavioral flow ratios.
    All division operations use an epsilon guard (1e-5) to avoid zero-division.
    """

    def __init__(self, flow_type: str = "biflow", add_ratios: bool = True):
        self.flow_type = flow_type
        self.add_ratios = add_ratios
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        cols = list(X.columns)
        if self.add_ratios and self.flow_type == "biflow":
            # Add engineered ratio names
            if "fwd_num_pkts" in cols and "bwd_num_pkts" in cols:
                cols.append("pkt_ratio")
                cols.append("total_pkts")
            if "fwd_num_bytes" in cols and "bwd_num_bytes" in cols:
                cols.append("byte_ratio")
                cols.append("total_bytes")
                cols.append("bytes_per_pkt")
            if "fwd_mean_pkt_len" in cols and "bwd_mean_pkt_len" in cols:
                cols.append("pkt_len_ratio")
        elif self.add_ratios and self.flow_type == "uniflow":
            if "num_bytes" in cols and "num_pkts" in cols:
                cols.append("bytes_per_pkt")

        self.feature_names_out_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.add_ratios:
            return X.copy()

        X_out = X.copy()
        eps = 1e-5

        if self.flow_type == "biflow":
            if "fwd_num_pkts" in X_out.columns and "bwd_num_pkts" in X_out.columns:
                X_out["pkt_ratio"] = (X_out["fwd_num_pkts"] + eps) / (X_out["bwd_num_pkts"] + eps)
                X_out["total_pkts"] = X_out["fwd_num_pkts"] + X_out["bwd_num_pkts"]

            if "fwd_num_bytes" in X_out.columns and "bwd_num_bytes" in X_out.columns:
                X_out["byte_ratio"] = (X_out["fwd_num_bytes"] + eps) / (X_out["bwd_num_bytes"] + eps)
                X_out["total_bytes"] = X_out["fwd_num_bytes"] + X_out["bwd_num_bytes"]
                if "total_pkts" in X_out.columns:
                    X_out["bytes_per_pkt"] = X_out["total_bytes"] / (X_out["total_pkts"] + eps)

            if "fwd_mean_pkt_len" in X_out.columns and "bwd_mean_pkt_len" in X_out.columns:
                X_out["pkt_len_ratio"] = (X_out["fwd_mean_pkt_len"] + eps) / (X_out["bwd_mean_pkt_len"] + eps)

        elif self.flow_type == "uniflow":
            if "num_bytes" in X_out.columns and "num_pkts" in X_out.columns:
                X_out["bytes_per_pkt"] = X_out["num_bytes"] / (X_out["num_pkts"] + eps)

        # Replace any residual non-finite values
        X_out = X_out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_out_


def build_flow_pipeline(
    flow_type: str = "biflow",
    include_dst_port: bool = True,
    add_ratios: bool = True
) -> Pipeline:
    """
    Construct a complete, leak-free preprocessing Pipeline.

    Args:
        flow_type: 'biflow' or 'uniflow'
        include_dst_port: Whether to retain destination port as a predictive feature
        add_ratios: Whether to compute domain flow ratios

    Returns:
        sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline(steps=[
        ("selector", FlowFeatureSelector(flow_type=flow_type, include_dst_port=include_dst_port)),
        ("engineer", FlowRatioFeatureEngineer(flow_type=flow_type, add_ratios=add_ratios))
    ])
    return pipeline


def extract_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Cleanly separate input features X from binary target y and metadata tags.

    Returns:
        X: pd.DataFrame containing feature columns
        y_binary: pd.Series containing is_attack (0 or 1)
        y_multi: Optional pd.Series containing attack_type string
    """
    if "is_attack" not in df.columns:
        raise KeyError("Dataframe must contain 'is_attack' column")

    y_binary = df["is_attack"].astype(int)
    y_multi = df["attack_type"] if "attack_type" in df.columns else None

    # Exclude metadata and target columns from raw feature pool
    cols_to_drop = [c for c in METADATA_COLUMNS if c in df.columns]
    X_raw = df.drop(columns=cols_to_drop)

    return X_raw, y_binary, y_multi


def save_feature_schema(
    feature_names: List[str],
    flow_type: str,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save stable, versioned feature schema to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_info = {
        "flow_type": flow_type,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "excluded_identifiers": IDENTIFIER_COLUMNS,
        "excluded_metadata": METADATA_COLUMNS,
        "metadata": metadata or {}
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)
