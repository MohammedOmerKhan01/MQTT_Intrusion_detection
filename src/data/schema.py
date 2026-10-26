"""
schema.py - Formal feature schema and data dictionary definitions
for Uniflow, Biflow, and Packet-level network intrusion datasets.
"""

from typing import Dict, List, Any

# Unidirectional Flow Feature Schema (19 Columns)
UNIFLOW_SCHEMA: Dict[str, Dict[str, Any]] = {
    "ip_src": {"type": "str", "category": "identifier", "description": "Source IP address"},
    "ip_dst": {"type": "str", "category": "identifier", "description": "Destination IP address"},
    "prt_src": {"type": "int", "category": "identifier", "description": "Source port number (0-65535)"},
    "prt_dst": {"type": "int", "category": "identifier", "description": "Destination port number (0-65535)"},
    "proto": {"type": "int", "category": "protocol", "description": "IP protocol number (e.g., 6=TCP, 17=UDP)"},
    "num_pkts": {"type": "int", "category": "traffic_volume", "description": "Total packets transmitted in the uniflow"},
    "mean_iat": {"type": "float", "category": "temporal", "description": "Mean packet inter-arrival time in seconds"},
    "std_iat": {"type": "float", "category": "temporal", "description": "Standard deviation of packet inter-arrival times"},
    "min_iat": {"type": "float", "category": "temporal", "description": "Minimum packet inter-arrival time in seconds"},
    "max_iat": {"type": "float", "category": "temporal", "description": "Maximum packet inter-arrival time in seconds"},
    "mean_pkt_len": {"type": "float", "category": "packet_size", "description": "Mean packet length in bytes"},
    "num_bytes": {"type": "int", "category": "traffic_volume", "description": "Total bytes transferred in the uniflow"},
    "num_psh_flags": {"type": "int", "category": "tcp_flags", "description": "Count of packets with TCP PSH flag set"},
    "num_rst_flags": {"type": "int", "category": "tcp_flags", "description": "Count of packets with TCP RST flag set"},
    "num_urg_flags": {"type": "int", "category": "tcp_flags", "description": "Count of packets with TCP URG flag set"},
    "std_pkt_len": {"type": "float", "category": "packet_size", "description": "Standard deviation of packet lengths in bytes"},
    "min_pkt_len": {"type": "float", "category": "packet_size", "description": "Minimum packet length in bytes"},
    "max_pkt_len": {"type": "float", "category": "packet_size", "description": "Maximum packet length in bytes"},
    "is_attack": {"type": "int", "category": "target", "description": "Ground truth label (0 = Benign/Normal, 1 = Attack)"}
}

# Bidirectional Flow Feature Schema (32 Columns)
BIFLOW_SCHEMA: Dict[str, Dict[str, Any]] = {
    "ip_src": {"type": "str", "category": "identifier", "description": "Source IP address"},
    "ip_dst": {"type": "str", "category": "identifier", "description": "Destination IP address"},
    "prt_src": {"type": "int", "category": "identifier", "description": "Source port number (0-65535)"},
    "prt_dst": {"type": "int", "category": "identifier", "description": "Destination port number (0-65535)"},
    "proto": {"type": "int", "category": "protocol", "description": "IP protocol number (e.g., 6=TCP, 17=UDP)"},
    "fwd_num_pkts": {"type": "int", "category": "traffic_volume", "description": "Forward direction packet count"},
    "bwd_num_pkts": {"type": "int", "category": "traffic_volume", "description": "Backward direction packet count"},
    "fwd_mean_iat": {"type": "float", "category": "temporal", "description": "Forward direction mean inter-arrival time (sec)"},
    "bwd_mean_iat": {"type": "float", "category": "temporal", "description": "Backward direction mean inter-arrival time (sec)"},
    "fwd_std_iat": {"type": "float", "category": "temporal", "description": "Forward direction inter-arrival time standard deviation"},
    "bwd_std_iat": {"type": "float", "category": "temporal", "description": "Backward direction inter-arrival time standard deviation"},
    "fwd_min_iat": {"type": "float", "category": "temporal", "description": "Forward direction minimum inter-arrival time (sec)"},
    "bwd_min_iat": {"type": "float", "category": "temporal", "description": "Backward direction minimum inter-arrival time (sec)"},
    "fwd_max_iat": {"type": "float", "category": "temporal", "description": "Forward direction maximum inter-arrival time (sec)"},
    "bwd_max_iat": {"type": "float", "category": "temporal", "description": "Backward direction maximum inter-arrival time (sec)"},
    "fwd_mean_pkt_len": {"type": "float", "category": "packet_size", "description": "Forward direction mean packet length (bytes)"},
    "bwd_mean_pkt_len": {"type": "float", "category": "packet_size", "description": "Backward direction mean packet length (bytes)"},
    "fwd_std_pkt_len": {"type": "float", "category": "packet_size", "description": "Forward direction packet length standard deviation"},
    "bwd_std_pkt_len": {"type": "float", "category": "packet_size", "description": "Backward direction packet length standard deviation"},
    "fwd_min_pkt_len": {"type": "float", "category": "packet_size", "description": "Forward direction minimum packet length (bytes)"},
    "bwd_min_pkt_len": {"type": "float", "category": "packet_size", "description": "Backward direction minimum packet length (bytes)"},
    "fwd_max_pkt_len": {"type": "float", "category": "packet_size", "description": "Forward direction maximum packet length (bytes)"},
    "bwd_max_pkt_len": {"type": "float", "category": "packet_size", "description": "Backward direction maximum packet length (bytes)"},
    "fwd_num_bytes": {"type": "int", "category": "traffic_volume", "description": "Forward direction byte count"},
    "bwd_num_bytes": {"type": "int", "category": "traffic_volume", "description": "Backward direction byte count"},
    "fwd_num_psh_flags": {"type": "int", "category": "tcp_flags", "description": "Forward direction TCP PSH flag count"},
    "bwd_num_psh_flags": {"type": "int", "category": "tcp_flags", "description": "Backward direction TCP PSH flag count"},
    "fwd_num_rst_flags": {"type": "int", "category": "tcp_flags", "description": "Forward direction TCP RST flag count"},
    "bwd_num_rst_flags": {"type": "int", "category": "tcp_flags", "description": "Backward direction TCP RST flag count"},
    "fwd_num_urg_flags": {"type": "int", "category": "tcp_flags", "description": "Forward direction TCP URG flag count"},
    "bwd_num_urg_flags": {"type": "int", "category": "tcp_flags", "description": "Backward direction TCP URG flag count"},
    "is_attack": {"type": "int", "category": "target", "description": "Ground truth label (0 = Benign/Normal, 1 = Attack)"}
}

# Packet-level Feature Schema (31 Columns)
PACKET_SCHEMA: Dict[str, Dict[str, Any]] = {
    "timestamp": {"type": "str", "category": "temporal", "description": "Absolute packet capture timestamp string"},
    "src_ip": {"type": "str", "category": "identifier", "description": "Source IP address"},
    "dst_ip": {"type": "str", "category": "identifier", "description": "Destination IP address"},
    "protocol": {"type": "str", "category": "protocol", "description": "Dissected protocol name (TCP, UDP, DNS, DATA, etc.)"},
    "ttl": {"type": "int", "category": "ip_header", "description": "IP Time to Live"},
    "ip_len": {"type": "int", "category": "ip_header", "description": "IP total packet length in bytes"},
    "ip_flag_df": {"type": "int", "category": "ip_header", "description": "IP Don't Fragment flag (0 or 1)"},
    "ip_flag_mf": {"type": "int", "category": "ip_header", "description": "IP More Fragments flag (0 or 1)"},
    "ip_flag_rb": {"type": "int", "category": "ip_header", "description": "IP Reserved Bit flag (0 or 1)"},
    "src_port": {"type": "int", "category": "identifier", "description": "Transport layer source port"},
    "dst_port": {"type": "int", "category": "identifier", "description": "Transport layer destination port"},
    "tcp_flag_res": {"type": "float", "category": "tcp_flags", "description": "TCP Reserved flag"},
    "tcp_flag_ns": {"type": "float", "category": "tcp_flags", "description": "TCP ECN-nonce concealment protection flag"},
    "tcp_flag_cwr": {"type": "float", "category": "tcp_flags", "description": "TCP Congestion Window Reduced flag"},
    "tcp_flag_ecn": {"type": "float", "category": "tcp_flags", "description": "TCP ECN Echo flag"},
    "tcp_flag_urg": {"type": "float", "category": "tcp_flags", "description": "TCP Urgent flag"},
    "tcp_flag_ack": {"type": "float", "category": "tcp_flags", "description": "TCP Acknowledgment flag"},
    "tcp_flag_push": {"type": "float", "category": "tcp_flags", "description": "TCP Push flag"},
    "tcp_flag_reset": {"type": "float", "category": "tcp_flags", "description": "TCP Reset flag"},
    "tcp_flag_syn": {"type": "float", "category": "tcp_flags", "description": "TCP Synchronize flag"},
    "tcp_flag_fin": {"type": "float", "category": "tcp_flags", "description": "TCP Finish flag"},
    "mqtt_messagetype": {"type": "float", "category": "mqtt_app", "description": "MQTT control packet type (e.g., CONNECT=1, PUBLISH=3)"},
    "mqtt_messagelength": {"type": "float", "category": "mqtt_app", "description": "MQTT remaining message length"},
    "mqtt_flag_uname": {"type": "float", "category": "mqtt_app", "description": "MQTT Connect Username flag"},
    "mqtt_flag_passwd": {"type": "float", "category": "mqtt_app", "description": "MQTT Connect Password flag"},
    "mqtt_flag_retain": {"type": "float", "category": "mqtt_app", "description": "MQTT Retain message flag"},
    "mqtt_flag_qos": {"type": "float", "category": "mqtt_app", "description": "MQTT Quality of Service level (0, 1, 2)"},
    "mqtt_flag_willflag": {"type": "float", "category": "mqtt_app", "description": "MQTT Will Flag"},
    "mqtt_flag_clean": {"type": "float", "category": "mqtt_app", "description": "MQTT Clean Session flag"},
    "mqtt_flag_reserved": {"type": "float", "category": "mqtt_app", "description": "MQTT reserved header flag"},
    "is_attack": {"type": "int", "category": "target", "description": "Ground truth label (0 = Benign/Normal, 1 = Attack)"}
}

def get_uniflow_columns() -> List[str]:
    """Return the ordered list of column names for uniflow features."""
    return list(UNIFLOW_SCHEMA.keys())

def get_biflow_columns() -> List[str]:
    """Return the ordered list of column names for biflow features."""
    return list(BIFLOW_SCHEMA.keys())

def get_packet_columns() -> List[str]:
    """Return the ordered list of column names for packet features."""
    return list(PACKET_SCHEMA.keys())
