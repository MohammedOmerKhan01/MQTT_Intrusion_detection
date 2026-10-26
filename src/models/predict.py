"""
predict.py - Standalone Headless Command-Line Interface (CLI) for ML-NIDS Inference

Supports:
1. Batch tabular flow inference (.csv, .parquet) and streaming packet captures (.pcap, .pcapng).
2. Strict schema verification and uniflow/biflow mismatch rejection.
3. Cryptographic SHA-256 model checksum verification (--verify-checksum).
4. Configurable decision thresholding (--threshold tau).
5. Output formatting in JSON, CSV, and ArcSight Common Event Format (CEF / Syslog) for SIEM ingestion.
6. Optional local SHAP explainability for flagged alerts (--explain).

Usage Examples:
    python src/models/predict.py --input data/processed/biflow_test.parquet --format json --output alerts.json
    python src/models/predict.py --input traffic.csv --threshold 0.35 --format syslog
    python src/models/predict.py --input capture.pcap --explain --output pcap_triage.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.persistence import load_model_artifact_safe, SecurityIntegrityError, SchemaMismatchError
from src.models.predict_pipeline import NIDSPredictionPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML-NIDS Standalone Inference CLI - High-throughput threat detection and SIEM forwarding."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input network traffic file (.csv, .parquet, .pcap, .pcapng)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to write output results. If omitted, summary is printed to stdout."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.50,
        help="Operational decision threshold tau in [0.0, 1.0]. Flows with P(Attack) >= tau are flagged (default: 0.50)."
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "csv", "syslog"],
        default="json",
        help="Output serialization format: json, csv, or syslog (ArcSight CEF strings for SIEM) (default: json)."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Compute local SHAP TreeExplainer attributions and briefings for flagged alerts."
    )
    parser.add_argument(
        "--verify-checksum",
        action="store_true",
        help="Validate model SHA-256 against model_manifest.json before executing inference."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records or packets to process."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed diagnostic stderr logging."
    )
    return parser.parse_args()


def format_cef_event(row: Dict[str, Any], top_drivers: Optional[List[str]] = None) -> str:
    """
    Format a flow prediction into an ArcSight Common Event Format (CEF) syslog string.
    Format:
    CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
    """
    is_atk = int(row.get("is_alert", 0)) == 1
    sig_id = "THREAT_ALERT" if is_atk else "BENIGN_FLOW"
    name = "Malicious Network Intrusion Detected" if is_atk else "Legitimate Network Flow Passed"

    # Map severity to 0-10 scale
    sev_map = {"CRITICAL": 10, "HIGH": 8, "MEDIUM": 5, "ELEVATED": 3, "BENIGN": 1}
    sev_num = sev_map.get(str(row.get("severity", "BENIGN")), 1)

    src_ip = row.get("ip_src", "unknown")
    dst_ip = row.get("ip_dst", "unknown")
    src_prt = row.get("prt_src", 0)
    dst_prt = row.get("prt_dst", 0)
    proto_num = row.get("proto", 6)
    proto_str = "TCP" if proto_num == 6 else ("UDP" if proto_num == 17 else str(proto_num))
    prob_atk = float(row.get("prob_attack", 0.0))
    subtype = row.get("attack_subtype", "unknown")

    ext_parts = [
        f"src={src_ip}",
        f"dst={dst_ip}",
        f"spt={src_prt}",
        f"dpt={dst_prt}",
        f"proto={proto_str}",
        f"act={'ALERT' if is_atk else 'ALLOW'}",
        f"prob={prob_atk:.4f}",
        f"cat={subtype}"
    ]

    if top_drivers:
        ext_parts.append(f"shaptop={'|'.join(top_drivers[:3])}")

    ext_str = " ".join(ext_parts)
    return f"CEF:0|DeepMind-Cyber|ML-NIDS|1.0.0|{sig_id}|{name}|{sev_num}|{ext_str}"


def validate_input_data_format(file_path: Path) -> str:
    """Inspect input file extension and detect representation."""
    suffix = file_path.suffix.lower()
    if suffix in [".pcap", ".pcapng"]:
        return "pcap"
    elif suffix in [".parquet"]:
        return "parquet"
    elif suffix in [".csv"]:
        return "csv"
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .csv, .parquet, .pcap, .pcapng")


def run_cli_inference(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input file does not exist: {input_path}\n")
        return 1

    if args.verbose:
        sys.stderr.write(f"[*] Initializing ML-NIDS Inference Engine...\n")
        sys.stderr.write(f"[*] Target Input: {input_path}\n")
        sys.stderr.write(f"[*] Decision Threshold (tau): {args.threshold}\n")

    # 1. Initialize Pipeline with optional Checksum Verification
    try:
        pipeline = NIDSPredictionPipeline()
        if args.verify_checksum:
            if args.verbose:
                sys.stderr.write("[*] Verifying cryptographic SHA-256 model checksums...\n")
            load_model_artifact_safe(pipeline.binary_model_path, enforce_checksum=True)
            if pipeline.multiclass_model_path.exists():
                load_model_artifact_safe(pipeline.multiclass_model_path, enforce_checksum=True)
            if args.verbose:
                sys.stderr.write("[+] Cryptographic integrity verified: SHA-256 signatures match manifest.\n")
    except SecurityIntegrityError as sie:
        sys.stderr.write(f"[FATAL SECURITY ERROR] {sie}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"Error initializing pipeline: {e}\n")
        return 1

    # 2. Ingest Data according to format
    data_format = validate_input_data_format(input_path)

    try:
        if data_format == "pcap":
            if args.verbose:
                sys.stderr.write(f"[*] Streaming PCAP capture through PcapFlowExtractor...\n")
            results_df, _ = pipeline.process_pcap(
                input_path,
                max_packets=args.limit or 50000,
                threshold=args.threshold
            )
            df_in = results_df # already scored
        else:
            if args.verbose:
                sys.stderr.write(f"[*] Reading tabular {data_format} file...\n")
            if data_format == "parquet":
                df_raw = pd.read_parquet(input_path)
            else:
                df_raw = pd.read_csv(input_path)

            if args.limit:
                df_raw = df_raw.head(args.limit)

            # Defensive Check: Uniflow Rejection
            if "proto" in df_raw.columns and "fwd_num_pkts" not in df_raw.columns and len(df_raw.columns) <= 20:
                raise SchemaMismatchError(
                    f"Uniflow representation detected ({len(df_raw.columns)} columns). "
                    f"The biflow detection engine requires bidirectional conversational metrics (forward and backward packet lengths, IATs, RST/PSH flags). "
                    f"Cannot silently substitute uniflow into a bidirectional model."
                )

            results_df = pipeline.predict_dataframe(df_raw, threshold=args.threshold)
            df_in = df_raw

    except SchemaMismatchError as sme:
        sys.stderr.write(f"[SCHEMA REJECTION ERROR] {sme}\n")
        return 3
    except Exception as e:
        sys.stderr.write(f"Inference execution failed: {e}\n")
        return 1

    total_flows = len(results_df)
    total_alerts = int((results_df["is_alert"] == 1).sum()) if "is_alert" in results_df.columns else 0
    alert_rate = (total_alerts / total_flows * 100) if total_flows > 0 else 0.0

    if args.verbose:
        sys.stderr.write(f"[+] Evaluation Complete: {total_flows:,} flows evaluated, {total_alerts:,} alerts flagged ({alert_rate:.2f}%).\n")

    # 3. Optional Explainability computation for flagged alerts
    explanations_map = {}
    if args.explain and total_alerts > 0:
        if args.verbose:
            sys.stderr.write(f"[*] Computing local SHAP TreeExplainer attributions for alerts (capped at 50)...\n")
        alert_indices = results_df[results_df["is_alert"] == 1].index.tolist()[:50]
        for idx in alert_indices:
            try:
                exp = pipeline.explain_flow_by_index(df_in, index=idx, top_k=3)
                explanations_map[idx] = exp
            except Exception as e:
                pass

    # 4. Serialize Output
    if args.format == "syslog":
        cef_lines = []
        for idx, row in results_df.iterrows():
            top_drivers = None
            if idx in explanations_map:
                top_drivers = [f"{d['feature']}:{d['shap_value']}" for d in explanations_map[idx]["top_attack_drivers"]]
            cef_lines.append(format_cef_event(row.to_dict(), top_drivers=top_drivers))

        output_str = "\n".join(cef_lines)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            if args.verbose:
                sys.stderr.write(f"[+] Wrote {len(cef_lines)} CEF syslog events to {args.output}\n")
        else:
            print(output_str[:3000])

    elif args.format == "csv":
        if args.output:
            results_df.to_csv(args.output, index=False)
            if args.verbose:
                sys.stderr.write(f"[+] Wrote CSV predictions to {args.output}\n")
        else:
            print(results_df.head(20).to_csv(index=False))

    else: # json
        report = pipeline.export_triage_report(
            results_df,
            explanations=list(explanations_map.values()),
            output_format="json"
        )
        report["predictions_sample"] = results_df.head(100).to_dict(orient="records")
        output_str = json.dumps(report, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            if args.verbose:
                sys.stderr.write(f"[+] Wrote JSON triage report to {args.output}\n")
        else:
            print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(run_cli_inference(parse_arguments()))
