"""
prepare.py - End-to-End Feature Preprocessing & Schema Generation Runner

Executes the leakage-safe pipeline across both Uniflow and Biflow datasets,
validates that identifiers are fully purged, calculates engineered ratios,
and persists stable feature schemas to artifacts/metadata/.
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.loaders import load_flow_dataset
from src.preprocessing.pipeline import (
    build_flow_pipeline,
    extract_features_and_target,
    save_feature_schema,
    IDENTIFIER_COLUMNS,
    METADATA_COLUMNS
)


def run_pipeline_preparation(base_dir: Path):
    print("=" * 70)
    print("STARTING STAGE 4: LEAKAGE-SAFE PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Process Biflow Data
    print("\n[1/3] Loading and Preprocessing Biflow Dataset...")
    biflow_raw = load_flow_dataset(flow_type="biflow", base_dir=base_dir)
    print(f"Loaded {len(biflow_raw):,} raw biflow rows with {len(biflow_raw.columns)} columns.")

    X_bi_raw, y_bi, y_bi_multi = extract_features_and_target(biflow_raw)
    biflow_pipe = build_flow_pipeline(flow_type="biflow", include_dst_port=True, add_ratios=True)
    
    # Fit-transform
    X_bi_preprocessed = biflow_pipe.fit_transform(X_bi_raw)
    biflow_features = list(X_bi_preprocessed.columns)

    print(f"Biflow Preprocessing Complete:")
    print(f"  - Input shape:  {X_bi_raw.shape}")
    print(f"  - Output shape: {X_bi_preprocessed.shape} ({len(biflow_features)} engineered features)")
    print(f"  - Target distribution: Normal={int((y_bi == 0).sum())}, Attack={int((y_bi == 1).sum())}")

    # Verify no identifiers leaked
    leaked_identifiers = [col for col in IDENTIFIER_COLUMNS if col in X_bi_preprocessed.columns]
    assert len(leaked_identifiers) == 0, f"Critical Leakage: Identifiers found in features: {leaked_identifiers}"
    print(f"  - Leakage Check: 100% of raw identifiers ({IDENTIFIER_COLUMNS}) successfully excluded.")

    # 2. Process Uniflow Data
    print("\n[2/3] Loading and Preprocessing Uniflow Dataset...")
    uniflow_raw = load_flow_dataset(flow_type="uniflow", base_dir=base_dir)
    X_uni_raw, y_uni, y_uni_multi = extract_features_and_target(uniflow_raw)
    uniflow_pipe = build_flow_pipeline(flow_type="uniflow", include_dst_port=True, add_ratios=True)
    
    X_uni_preprocessed = uniflow_pipe.fit_transform(X_uni_raw)
    uniflow_features = list(X_uni_preprocessed.columns)

    print(f"Uniflow Preprocessing Complete:")
    print(f"  - Input shape:  {X_uni_raw.shape}")
    print(f"  - Output shape: {X_uni_preprocessed.shape} ({len(uniflow_features)} engineered features)")

    # 3. Persist Feature Schemas
    metadata_dir = base_dir / "artifacts" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    biflow_schema_path = metadata_dir / "biflow_feature_schema.json"
    save_feature_schema(
        feature_names=biflow_features,
        flow_type="biflow",
        output_path=biflow_schema_path,
        metadata={
            "description": "Leakage-safe engineered features for Biflow classification",
            "base_numerical_features": 27,
            "engineered_ratio_features": ["pkt_ratio", "total_pkts", "byte_ratio", "total_bytes", "bytes_per_pkt", "pkt_len_ratio"]
        }
    )
    print(f"\n[3/3] Feature schemas saved:")
    print(f"  - Biflow:  {biflow_schema_path}")

    uniflow_schema_path = metadata_dir / "uniflow_feature_schema.json"
    save_feature_schema(
        feature_names=uniflow_features,
        flow_type="uniflow",
        output_path=uniflow_schema_path,
        metadata={
            "description": "Leakage-safe engineered features for Uniflow classification"
        }
    )
    print(f"  - Uniflow: {uniflow_schema_path}")

    print("=" * 70)
    print("STAGE 4 PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Biflow Predictor Columns ({len(biflow_features)}):")
    for i, col in enumerate(biflow_features, 1):
        print(f"  {i:02d}. {col}")
    print("=" * 70)

    return X_bi_preprocessed, y_bi, biflow_pipe


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline_preparation(workspace_root)
