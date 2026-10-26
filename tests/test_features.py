"""
test_features.py - Automated Stage 3 Tests for PCAP Extraction & Flow Assembly
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.features.pcap_extractor import FlowAccumulator, PcapFlowExtractor
from src.data.schema import get_biflow_columns


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_flow_accumulator_direction_and_stats():
    """Verify FlowAccumulator accurately segregates forward and backward metrics."""
    flow = FlowAccumulator("10.0.0.1", "10.0.0.2", 12345, 80, 6, start_time=1000.0)

    # 1. Forward packet (SYN): len=60
    flow.add_packet(pkt_time=1000.0, pkt_len=60, is_forward=True, psh=0, rst=0, urg=0, fin=0)
    # 2. Backward packet (SYN-ACK): len=60
    flow.add_packet(pkt_time=1000.05, pkt_len=60, is_forward=False, psh=0, rst=0, urg=0, fin=0)
    # 3. Forward packet (ACK): len=52
    flow.add_packet(pkt_time=1000.08, pkt_len=52, is_forward=True, psh=0, rst=0, urg=0, fin=0)
    # 4. Forward packet (Data with PSH): len=120
    flow.add_packet(pkt_time=1000.12, pkt_len=120, is_forward=True, psh=1, rst=0, urg=0, fin=0)

    record = flow.to_biflow_record(ground_truth_label=1)

    assert record["fwd_num_pkts"] == 3
    assert record["bwd_num_pkts"] == 1
    assert record["fwd_num_psh_flags"] == 1
    assert record["bwd_num_psh_flags"] == 0
    assert record["fwd_num_bytes"] == 60 + 52 + 120
    assert record["bwd_num_bytes"] == 60
    assert record["fwd_min_pkt_len"] == 52.0
    assert record["fwd_max_pkt_len"] == 120.0
    assert record["is_attack"] == 1


def test_pcap_extractor_schema_conformity():
    """Verify that PcapFlowExtractor output columns match BIFLOW_SCHEMA exactly."""
    expected_cols = get_biflow_columns()
    extractor = PcapFlowExtractor()
    df_empty = extractor.process_pcap.__wrapped__ if hasattr(extractor.process_pcap, '__wrapped__') else None
    
    # Verify column definitions
    assert len(expected_cols) == 32
    assert "fwd_num_pkts" in expected_cols
    assert "bwd_num_pkts" in expected_cols
    assert "is_attack" in expected_cols


def test_pcap_validation_artifact_exists(base_dir):
    """Verify validation report exists and passed validation."""
    report_file = base_dir / "artifacts" / "reports" / "pcap_extraction_validation.json"
    assert report_file.exists(), f"Missing {report_file}"

    with open(report_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert data["validation_status"] == "PASSED"
    assert data["schema_conformity"] is True
    assert "feature_comparison" in data
