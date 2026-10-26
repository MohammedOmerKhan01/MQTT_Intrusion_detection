"""
inventory.py - Complete Dataset Inventory & Manifest Generation Script

Performs deep, safe, memory-conscious inspection of all network intrusion archives:
- Unidirectional flow features (uniflow)
- Bidirectional flow features (biflow)
- Packet-level features
- Raw PCAP captures

Outputs structured summary to console and writes a machine-readable manifest to data/manifest.json.
"""

import os
import sys
import json
import struct
import zipfile
import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.schema import UNIFLOW_SCHEMA, BIFLOW_SCHEMA, PACKET_SCHEMA


def inspect_pcap_header(raw_bytes: bytes) -> Dict[str, Any]:
    """Inspect the initial 24 bytes of a PCAP file to detect file format and link-layer type."""
    if len(raw_bytes) < 24:
        return {"format": "Invalid/Too short", "linktype": None, "snaplen": None}
    
    magic = raw_bytes[:4]
    if magic in [b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1']:
        endian = "big" if magic == b'\xa1\xb2\xc3\xd4' else "little"
        prefix = ">" if endian == "big" else "<"
        v_major, v_minor = struct.unpack(f"{prefix}HH", raw_bytes[4:8])
        snaplen, linktype = struct.unpack(f"{prefix}II", raw_bytes[16:24])
        return {
            "format": "libpcap_microsecond",
            "endian": endian,
            "version": f"{v_major}.{v_minor}",
            "snaplen": snaplen,
            "linktype": linktype,
            "linktype_name": "Ethernet (1)" if linktype == 1 else f"LinkType({linktype})"
        }
    elif magic in [b'\xa1\xb2\x3c\x4d', b'\x4d\x3c\xb2\xa1']:
        return {"format": "libpcap_nanosecond", "endian": "unknown", "version": "unknown"}
    elif magic == b'\n\r\r\n':
        return {"format": "pcapng", "endian": "unknown", "version": "unknown"}
    else:
        return {"format": f"Unknown_magic({magic.hex()})", "endian": None, "version": None}


def count_file_lines_in_zip(zf: zipfile.ZipFile, member_name: str, chunk_size: int = 1024 * 1024) -> int:
    """Safely count lines in a large compressed file using stream chunking."""
    line_count = 0
    with zf.open(member_name) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            line_count += chunk.count(b'\n')
    return line_count


def inspect_flow_archive(zip_path: str, expected_schema: Dict[str, Any], flow_type: str) -> Dict[str, Any]:
    """Inspect all CSV files within a flow feature zip archive."""
    results = {}
    if not os.path.exists(zip_path):
        return {"error": f"Archive {zip_path} not found"}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            if not member.filename.endswith('.csv'):
                continue
            with zf.open(member.filename) as f:
                df = pd.read_csv(f)
                
                # Check column schema match
                actual_cols = list(df.columns)
                expected_cols = list(expected_schema.keys())
                schema_match = actual_cols == expected_cols
                missing_cols = [c for c in expected_cols if c not in actual_cols]
                extra_cols = [c for c in actual_cols if c not in expected_cols]

                # Class distribution
                if 'is_attack' in df.columns:
                    dist = df['is_attack'].value_counts(dropna=False).to_dict()
                    total = len(df)
                    attack_count = dist.get(1, 0)
                    normal_count = dist.get(0, 0)
                    attack_pct = round((attack_count / total) * 100, 2) if total > 0 else 0.0
                    normal_pct = round((normal_count / total) * 100, 2) if total > 0 else 0.0
                else:
                    dist = {}
                    attack_count = normal_count = 0
                    attack_pct = normal_pct = 0.0

                # Missing values
                null_dict = df.isnull().sum().to_dict()
                cols_with_nulls = {k: int(v) for k, v in null_dict.items() if v > 0}

                # Network identifiers
                ip_src_unique = int(df['ip_src'].nunique()) if 'ip_src' in df.columns else None
                ip_dst_unique = int(df['ip_dst'].nunique()) if 'ip_dst' in df.columns else None
                proto_counts = {str(k): int(v) for k, v in df['proto'].value_counts().to_dict().items()} if 'proto' in df.columns else {}

                results[member.filename] = {
                    "uncompressed_bytes": member.file_size,
                    "uncompressed_mb": round(member.file_size / (1024 * 1024), 2),
                    "compressed_bytes": member.compress_size,
                    "compressed_mb": round(member.compress_size / (1024 * 1024), 2),
                    "total_records": len(df),
                    "column_count": len(actual_cols),
                    "columns": actual_cols,
                    "schema_match": schema_match,
                    "missing_cols": missing_cols,
                    "extra_cols": extra_cols,
                    "class_distribution": {
                        "normal_0": int(normal_count),
                        "attack_1": int(attack_count),
                        "normal_pct": normal_pct,
                        "attack_pct": attack_pct
                    },
                    "null_counts": cols_with_nulls,
                    "network_summary": {
                        "unique_src_ips": ip_src_unique,
                        "unique_dst_ips": ip_dst_unique,
                        "protocol_distribution": proto_counts
                    }
                }
    return results


def inspect_packet_archive(zip_path: str) -> Dict[str, Any]:
    """Inspect packet feature archive safely by streaming lines and inspecting sample headers."""
    results = {}
    if not os.path.exists(zip_path):
        return {"error": f"Archive {zip_path} not found"}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            if not member.filename.endswith('.csv'):
                continue
            with zf.open(member.filename) as f:
                df_sample = pd.read_csv(f, nrows=5)
                actual_cols = list(df_sample.columns)
            
            line_count = count_file_lines_in_zip(zf, member.filename)
            records = max(0, line_count - 1)  # Subtract header

            results[member.filename] = {
                "uncompressed_bytes": member.file_size,
                "uncompressed_mb": round(member.file_size / (1024 * 1024), 2),
                "compressed_bytes": member.compress_size,
                "compressed_mb": round(member.compress_size / (1024 * 1024), 2),
                "total_records": records,
                "column_count": len(actual_cols),
                "columns": actual_cols,
                "sample_row": df_sample.iloc[0].to_dict() if len(df_sample) > 0 else {}
            }
    return results


def inspect_pcap_archive(zip_path: str) -> Dict[str, Any]:
    """Inspect PCAP files within the pcap archive."""
    results = {}
    if not os.path.exists(zip_path):
        return {"error": f"Archive {zip_path} not found"}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            with zf.open(member.filename) as f:
                header_bytes = f.read(24)
                metadata = inspect_pcap_header(header_bytes)
            
            results[member.filename] = {
                "uncompressed_bytes": member.file_size,
                "uncompressed_mb": round(member.file_size / (1024 * 1024), 2),
                "compressed_bytes": member.compress_size,
                "compressed_mb": round(member.compress_size / (1024 * 1024), 2),
                "pcap_metadata": metadata
            }
    return results


def generate_manifest(base_dir: Path) -> Dict[str, Any]:
    """Build the complete, verified dataset inventory manifest."""
    manifest = {
        "manifest_version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "base_directory": str(base_dir),
        "dataset_summary": {
            "total_archives": 4,
            "scenarios": ["mqtt_bruteforce", "normal", "scan_A", "scan_sU", "sparta"]
        },
        "archives": {}
    }

    print("=" * 70)
    print("STARTING DATASET INVENTORY INSPECTION")
    print("=" * 70)

    # 1. Uniflow
    uniflow_zip = base_dir / "uniflow_features.zip"
    print(f"\n[1/4] Inspecting Unidirectional Flows: {uniflow_zip.name} ...")
    uniflow_info = inspect_flow_archive(str(uniflow_zip), UNIFLOW_SCHEMA, "uniflow")
    manifest["archives"]["uniflow_features"] = {
        "archive_file": uniflow_zip.name,
        "archive_size_mb": round(os.path.getsize(uniflow_zip) / (1024 * 1024), 2) if uniflow_zip.exists() else 0,
        "files": uniflow_info
    }

    # 2. Biflow
    biflow_zip = base_dir / "biflow_features.zip"
    print(f"[2/4] Inspecting Bidirectional Flows: {biflow_zip.name} ...")
    biflow_info = inspect_flow_archive(str(biflow_zip), BIFLOW_SCHEMA, "biflow")
    manifest["archives"]["biflow_features"] = {
        "archive_file": biflow_zip.name,
        "archive_size_mb": round(os.path.getsize(biflow_zip) / (1024 * 1024), 2) if biflow_zip.exists() else 0,
        "files": biflow_info
    }

    # 3. Packet features
    packet_zip = base_dir / "packet_features.zip"
    print(f"[3/4] Inspecting Packet-level Features: {packet_zip.name} ...")
    packet_info = inspect_packet_archive(str(packet_zip))
    manifest["archives"]["packet_features"] = {
        "archive_file": packet_zip.name,
        "archive_size_mb": round(os.path.getsize(packet_zip) / (1024 * 1024), 2) if packet_zip.exists() else 0,
        "files": packet_info
    }

    # 4. PCAP files
    pcap_zip = base_dir / "pcap_files.zip"
    print(f"[4/4] Inspecting Raw PCAP Captures: {pcap_zip.name} ...")
    pcap_info = inspect_pcap_archive(str(pcap_zip))
    manifest["archives"]["pcap_files"] = {
        "archive_file": pcap_zip.name,
        "archive_size_mb": round(os.path.getsize(pcap_zip) / (1024 * 1024), 2) if pcap_zip.exists() else 0,
        "files": pcap_info
    }

    # Compute overall aggregates
    total_uniflow_records = sum(f.get("total_records", 0) for f in uniflow_info.values() if isinstance(f, dict))
    total_biflow_records = sum(f.get("total_records", 0) for f in biflow_info.values() if isinstance(f, dict))
    total_packet_records = sum(f.get("total_records", 0) for f in packet_info.values() if isinstance(f, dict))

    manifest["aggregates"] = {
        "uniflow_total_records": total_uniflow_records,
        "biflow_total_records": total_biflow_records,
        "packet_total_records": total_packet_records
    }

    # Save to data/manifest.json
    manifest_path = base_dir / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as out_f:
        json.dump(manifest, out_f, indent=2)

    print(f"\nManifest successfully saved to: {manifest_path}")
    print("=" * 70)
    print("DATASET INVENTORY SUMMARY")
    print("=" * 70)
    print(f"Total Uniflow Records: {total_uniflow_records:,}")
    print(f"Total Biflow Records:  {total_biflow_records:,}")
    print(f"Total Packet Records:  {total_packet_records:,}")
    print("-" * 70)
    print(f"{'Archive / File':<32} | {'Records':>10} | {'Normal (0)':>10} | {'Attack (1)':>10} | {'Atk %':>7}")
    print("-" * 70)

    for group_name, group_data in [("UNIFLOW", uniflow_info), ("BIFLOW", biflow_info)]:
        print(f"[{group_name}]")
        for fname, finfo in group_data.items():
            if not isinstance(finfo, dict):
                continue
            rec = finfo.get("total_records", 0)
            cd = finfo.get("class_distribution", {})
            norm = cd.get("normal_0", 0)
            atk = cd.get("attack_1", 0)
            pct = cd.get("attack_pct", 0.0)
            print(f"  {fname:<30} | {rec:>10,} | {norm:>10,} | {atk:>10,} | {pct:>6.2f}%")
    print("=" * 70)

    return manifest


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    generate_manifest(workspace_root)
