"""
pcap_extractor.py - Reproducible PCAP to Flow Feature Extraction Engine

Extracts bidirectional flow statistics (matching BIFLOW_SCHEMA) from raw
PCAP captures using Scapy streaming.

Architecture & Conventions:
1. Flow Keying: 5-tuple (ip_src, ip_dst, prt_src, prt_dst, proto)
2. Direction Determination: The host transmitting the initial packet of a session
   defines the Forward (fwd) direction; return traffic defines the Backward (bwd) direction.
3. Timeout Model: Dual inactivity timeout (default 120s) and TCP teardown (FIN/RST).
4. Packet Length Basis: IP total length (pkt[IP].len) to ensure compatibility with
   the pre-extracted benchmark flow datasets.
"""

import os
import sys
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator

import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.schema import get_biflow_columns


class FlowAccumulator:
    """Accumulates packets and maintains streaming statistics for a single bidirectional session."""

    def __init__(self, initiator_src: str, responder_dst: str, src_port: int, dst_port: int, proto: int, start_time: float):
        self.ip_src = initiator_src
        self.ip_dst = responder_dst
        self.prt_src = src_port
        self.prt_dst = dst_port
        self.proto = proto
        
        self.start_time = start_time
        self.last_time = start_time
        
        # Forward stream metrics
        self.fwd_last_time: Optional[float] = None
        self.fwd_pkt_lens: List[int] = []
        self.fwd_iats: List[float] = []
        self.fwd_psh = 0
        self.fwd_rst = 0
        self.fwd_urg = 0
        
        # Backward stream metrics
        self.bwd_last_time: Optional[float] = None
        self.bwd_pkt_lens: List[int] = []
        self.bwd_iats: List[float] = []
        self.bwd_psh = 0
        self.bwd_rst = 0
        self.bwd_urg = 0
        
        self.is_closed = False

    def add_packet(self, pkt_time: float, pkt_len: int, is_forward: bool, psh: int, rst: int, urg: int, fin: int):
        """Update directional statistics with newly dissected packet."""
        self.last_time = pkt_time

        if is_forward:
            if self.fwd_last_time is not None:
                iat = max(0.0, pkt_time - self.fwd_last_time)
                self.fwd_iats.append(iat)
            self.fwd_last_time = pkt_time
            self.fwd_pkt_lens.append(pkt_len)
            self.fwd_psh += psh
            self.fwd_rst += rst
            self.fwd_urg += urg
        else:
            if self.bwd_last_time is not None:
                iat = max(0.0, pkt_time - self.bwd_last_time)
                self.bwd_iats.append(iat)
            self.bwd_last_time = pkt_time
            self.bwd_pkt_lens.append(pkt_len)
            self.bwd_psh += psh
            self.bwd_rst += rst
            self.bwd_urg += urg

        # Check for TCP connection teardown
        if rst > 0 or fin > 0:
            self.is_closed = True

    def to_biflow_record(self, ground_truth_label: int = -1) -> Dict[str, Any]:
        """Convert accumulated statistics into a dictionary strictly matching BIFLOW_SCHEMA."""
        f_pkts = len(self.fwd_pkt_lens)
        b_pkts = len(self.bwd_pkt_lens)

        # Forward stats
        f_mean_pkt = float(np.mean(self.fwd_pkt_lens)) if f_pkts > 0 else 0.0
        f_std_pkt = float(np.std(self.fwd_pkt_lens)) if f_pkts > 1 else 0.0
        f_min_pkt = float(min(self.fwd_pkt_lens)) if f_pkts > 0 else 0.0
        f_max_pkt = float(max(self.fwd_pkt_lens)) if f_pkts > 0 else 0.0
        f_bytes = int(sum(self.fwd_pkt_lens))

        f_mean_iat = float(np.mean(self.fwd_iats)) if len(self.fwd_iats) > 0 else 0.0
        f_std_iat = float(np.std(self.fwd_iats)) if len(self.fwd_iats) > 1 else 0.0
        f_min_iat = float(min(self.fwd_iats)) if len(self.fwd_iats) > 0 else 0.0
        f_max_iat = float(max(self.fwd_iats)) if len(self.fwd_iats) > 0 else 0.0

        # Backward stats
        b_mean_pkt = float(np.mean(self.bwd_pkt_lens)) if b_pkts > 0 else 0.0
        b_std_pkt = float(np.std(self.bwd_pkt_lens)) if b_pkts > 1 else 0.0
        b_min_pkt = float(min(self.bwd_pkt_lens)) if b_pkts > 0 else 0.0
        b_max_pkt = float(max(self.bwd_pkt_lens)) if b_pkts > 0 else 0.0
        b_bytes = int(sum(self.bwd_pkt_lens))

        b_mean_iat = float(np.mean(self.bwd_iats)) if len(self.bwd_iats) > 0 else 0.0
        b_std_iat = float(np.std(self.bwd_iats)) if len(self.bwd_iats) > 1 else 0.0
        b_min_iat = float(min(self.bwd_iats)) if len(self.bwd_iats) > 0 else 0.0
        b_max_iat = float(max(self.bwd_iats)) if len(self.bwd_iats) > 0 else 0.0

        return {
            "ip_src": self.ip_src,
            "ip_dst": self.ip_dst,
            "prt_src": self.prt_src,
            "prt_dst": self.prt_dst,
            "proto": self.proto,
            "fwd_num_pkts": f_pkts,
            "bwd_num_pkts": b_pkts,
            "fwd_mean_iat": f_mean_iat,
            "bwd_mean_iat": b_mean_iat,
            "fwd_std_iat": f_std_iat,
            "bwd_std_iat": b_std_iat,
            "fwd_min_iat": f_min_iat,
            "bwd_min_iat": b_min_iat,
            "fwd_max_iat": f_max_iat,
            "bwd_max_iat": b_max_iat,
            "fwd_mean_pkt_len": f_mean_pkt,
            "bwd_mean_pkt_len": b_mean_pkt,
            "fwd_std_pkt_len": f_std_pkt,
            "bwd_std_pkt_len": b_std_pkt,
            "fwd_min_pkt_len": f_min_pkt,
            "bwd_min_pkt_len": b_min_pkt,
            "fwd_max_pkt_len": f_max_pkt,
            "bwd_max_pkt_len": b_max_pkt,
            "fwd_num_bytes": f_bytes,
            "bwd_num_bytes": b_bytes,
            "fwd_num_psh_flags": self.fwd_psh,
            "bwd_num_psh_flags": self.bwd_psh,
            "fwd_num_rst_flags": self.fwd_rst,
            "bwd_num_rst_flags": self.bwd_rst,
            "fwd_num_urg_flags": self.fwd_urg,
            "bwd_num_urg_flags": self.bwd_urg,
            "is_attack": ground_truth_label
        }


class PcapFlowExtractor:
    """Streaming PCAP packet reader and bidirectional flow assembler."""

    def __init__(self, inactivity_timeout: float = 120.0, require_bidirectional: bool = True):
        self.inactivity_timeout = inactivity_timeout
        self.require_bidirectional = require_bidirectional
        self.active_flows: Dict[Tuple, FlowAccumulator] = {}
        self.completed_flows: List[Dict[str, Any]] = []

    def _get_session_key(self, ip_a: str, ip_b: str, port_a: int, port_b: int, proto: int) -> Tuple:
        """Create bidirectional session key independent of directional perspective."""
        endpoint_a = (ip_a, port_a)
        endpoint_b = (ip_b, port_b)
        if endpoint_a <= endpoint_b:
            return (endpoint_a, endpoint_b, proto)
        return (endpoint_b, endpoint_a, proto)

    def process_pcap(
        self,
        pcap_source: Any,
        max_packets: Optional[int] = None,
        label_policy_func: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Process a PCAP file path or file-like stream and assemble bidirectional flows.

        Args:
            pcap_source: File path (str/Path) or open binary file-like stream
            max_packets: Optional packet limit for testing/sampling
            label_policy_func: Optional callback (ip_src, ip_dst, prt_src, prt_dst, proto) -> int

        Returns:
            pd.DataFrame matching BIFLOW_SCHEMA exactly.
        """
        self.active_flows.clear()
        self.completed_flows.clear()

        reader = PcapReader(pcap_source)
        packet_count = 0

        try:
            for pkt in reader:
                if IP not in pkt:
                    # Non-IPv4 packets skipped in this flow model
                    continue

                packet_count += 1
                pkt_time = float(pkt.time)
                ip_src = pkt[IP].src
                ip_dst = pkt[IP].dst
                proto = int(pkt[IP].proto)
                pkt_len = int(pkt[IP].len)

                # Extract transport ports and TCP flags
                prt_src, prt_dst = 0, 0
                psh, rst, urg, fin = 0, 0, 0, 0

                if TCP in pkt:
                    prt_src = int(pkt[TCP].sport)
                    prt_dst = int(pkt[TCP].dport)
                    flags = int(pkt[TCP].flags)
                    psh = 1 if flags & 0x08 else 0
                    rst = 1 if flags & 0x04 else 0
                    urg = 1 if flags & 0x20 else 0
                    fin = 1 if flags & 0x01 else 0
                elif UDP in pkt:
                    prt_src = int(pkt[UDP].sport)
                    prt_dst = int(pkt[UDP].dport)
                else:
                    # Generic IP protocol
                    prt_src, prt_dst = 0, 0

                session_key = self._get_session_key(ip_src, ip_dst, prt_src, prt_dst, proto)

                # Check for active flow timeout
                if session_key in self.active_flows:
                    flow = self.active_flows[session_key]
                    if (pkt_time - flow.last_time) > self.inactivity_timeout:
                        # Flow expired; finalize it
                        if not self.require_bidirectional or len(flow.bwd_pkt_lens) > 0:
                            lbl = label_policy_func(flow.ip_src, flow.ip_dst, flow.prt_src, flow.prt_dst, flow.proto) if label_policy_func else -1
                            self.completed_flows.append(flow.to_biflow_record(lbl))
                        del self.active_flows[session_key]

                # Update or initialize flow
                if session_key not in self.active_flows:
                    self.active_flows[session_key] = FlowAccumulator(
                        initiator_src=ip_src,
                        responder_dst=ip_dst,
                        src_port=prt_src,
                        dst_port=prt_dst,
                        proto=proto,
                        start_time=pkt_time
                    )
                    is_forward = True
                else:
                    flow = self.active_flows[session_key]
                    is_forward = (ip_src == flow.ip_src and prt_src == flow.prt_src)

                flow = self.active_flows[session_key]
                flow.add_packet(pkt_time, pkt_len, is_forward, psh, rst, urg, fin)

                if flow.is_closed:
                    if not self.require_bidirectional or len(flow.bwd_pkt_lens) > 0:
                        lbl = label_policy_func(flow.ip_src, flow.ip_dst, flow.prt_src, flow.prt_dst, flow.proto) if label_policy_func else -1
                        self.completed_flows.append(flow.to_biflow_record(lbl))
                    del self.active_flows[session_key]

                if max_packets is not None and packet_count >= max_packets:
                    break

        finally:
            reader.close()

        # Flush any remaining active flows
        for flow in self.active_flows.values():
            if not self.require_bidirectional or len(flow.bwd_pkt_lens) > 0:
                lbl = label_policy_func(flow.ip_src, flow.ip_dst, flow.prt_src, flow.prt_dst, flow.proto) if label_policy_func else -1
                self.completed_flows.append(flow.to_biflow_record(lbl))
        self.active_flows.clear()

        if not self.completed_flows:
            return pd.DataFrame(columns=get_biflow_columns())

        df_out = pd.DataFrame(self.completed_flows)
        # Ensure exact column ordering matching BIFLOW_SCHEMA
        ordered_cols = get_biflow_columns()
        return df_out[ordered_cols]
