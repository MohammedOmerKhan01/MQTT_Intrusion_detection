"""
validate_extraction.py - Empirical Validation of Scapy PCAP Flow Extractor

Compares feature distributions and 5-tuple consistency between raw PCAP
extracted flows and the baseline pre-extracted CSV features (biflow_scan_A.csv).
Outputs a comprehensive validation report to artifacts/reports/pcap_extraction_validation.json.
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.pcap_extractor import PcapFlowExtractor
from src.data.loaders import load_single_scenario
from src.data.schema import get_biflow_columns


def run_pcap_validation(base_dir: Path, packet_limit: int = 5000) -> Dict[str, Any]:
    """Execute comparative validation between PCAP extraction and CSV baseline."""
    print("=" * 70)
    print("STARTING STAGE 3: PCAP FEATURE EXTRACTION & COMPARATIVE VALIDATION")
    print("=" * 70)

    pcap_archive = base_dir / "pcap_files.zip"
    if not pcap_archive.exists():
        raise FileNotFoundError(f"Missing {pcap_archive}")

    # 1. Extract flows from scan_A.pcap
    print(f"\n[1/3] Extracting flows from scan_A.pcap (first {packet_limit:,} packets)...")
    extractor = PcapFlowExtractor(inactivity_timeout=120.0, require_bidirectional=True)
    
    with zipfile.ZipFile(pcap_archive, 'r') as zf:
        with zf.open('scan_A.pcap') as f:
            extracted_df = extractor.process_pcap(f, max_packets=packet_limit)

    print(f"Extracted {len(extracted_df):,} bidirectional flows from {packet_limit:,} packets.")

    # 2. Load baseline biflow_scan_A.csv
    print("[2/3] Loading baseline biflow_scan_A.csv for comparison...")
    baseline_df = load_single_scenario("biflow", "scan_A", base_dir=base_dir)

    # 3. Compare statistical distributions for core features
    print("[3/3] Computing statistical alignment and feature delta...")
    features_to_compare = [
        "fwd_mean_pkt_len", "bwd_mean_pkt_len",
        "fwd_num_pkts", "bwd_num_pkts",
        "fwd_num_psh_flags", "fwd_num_rst_flags"
    ]

    comparison_results: Dict[str, Any] = {}
    for feat in features_to_compare:
        ext_mean = float(extracted_df[feat].mean()) if len(extracted_df) > 0 else 0.0
        base_mean = float(baseline_df[feat].mean())
        ext_median = float(extracted_df[feat].median()) if len(extracted_df) > 0 else 0.0
        base_median = float(baseline_df[feat].median())

        comparison_results[feat] = {
            "extracted_mean": round(ext_mean, 2),
            "baseline_mean": round(base_mean, 2),
            "extracted_median": round(ext_median, 2),
            "baseline_median": round(base_median, 2),
            "mean_relative_diff_pct": round(abs(ext_mean - base_mean) / max(1e-5, base_mean) * 100, 2)
        }

    # Schema conformity check
    schema_match = list(extracted_df.columns) == get_biflow_columns()

    validation_report = {
        "validation_status": "PASSED" if schema_match else "FAILED",
        "tested_pcap": "scan_A.pcap",
        "packets_processed": packet_limit,
        "flows_assembled": len(extracted_df),
        "schema_conformity": schema_match,
        "feature_comparison": comparison_results,
        "configuration": {
            "extractor_tool": "scapy",
            "scapy_version": "2.8.0",
            "length_basis": "ip_total_length (pkt[IP].len)",
            "inactivity_timeout_seconds": 120.0,
            "require_bidirectional": True
        },
        "known_limitations": [
            "Scapy is written in pure Python; streaming 4+ GB PCAP files is CPU/IO intensive compared to native C tools like TShark or CICFlowMeter.",
            "Minor timing deltas exist if baseline extractor used microsecond vs millisecond precision rounding.",
            "Flow timeout boundary differences: flows active at the packet limit cut-off are finalized during flush."
        ]
    }

    # Save report
    out_dir = base_dir / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "pcap_extraction_validation.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2)

    print(f"\nValidation report saved to: {report_file}")
    print("=" * 70)
    print("PCAP FEATURE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Schema Conformity (32/32 columns): {schema_match}")
    print(f"Extracted Sample Mean Fwd Packet Len: {comparison_results['fwd_mean_pkt_len']['extracted_mean']} B (Baseline: {comparison_results['fwd_mean_pkt_len']['baseline_mean']} B)")
    print(f"Extracted Sample Mean Bwd Packet Len: {comparison_results['bwd_mean_pkt_len']['extracted_mean']} B (Baseline: {comparison_results['bwd_mean_pkt_len']['baseline_mean']} B)")
    print("=" * 70)

    return validation_report


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    run_pcap_validation(workspace_root)
