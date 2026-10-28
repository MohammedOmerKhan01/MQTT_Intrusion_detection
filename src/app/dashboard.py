"""
dashboard.py - Interactive SOC Analyst Dashboard & Explainable AI Console

Built with Streamlit for cybersecurity analysts and academic evaluation.
Includes 6 dedicated modules:
A. Overview & System Telemetry
B. Dataset Analysis & Quality Auditing
C. Threat Detection & Ingestion (CSV and PCAP)
D. Alert Investigation & Local SHAP Explainability
E. SOC Incident Reports & Serialization
F. Model Provenance, Feature Governance & Limitations
"""

import os
import sys
import json
import io
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure clean Matplotlib config directory before importing matplotlib
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
temp_mpl_dir = ROOT_DIR / ".temp_mpl"
temp_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(temp_mpl_dir))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.predict_pipeline import NIDSPredictionPipeline

# Page Configuration
st.set_page_config(
    page_title="ML-NIDS & Explainable AI SOC Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1626;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: monospace;
    }
    .metric-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
    }
    .alert-badge-critical {
        background-color: #450A0A;
        border: 1px solid #DC2626;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: 600;
    }
    .alert-badge-benign {
        background-color: #064E3B;
        border: 1px solid #059669;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Network Port to Service Name Mapping
PORT_NAME_MAP = {
    1883: "1883 (MQTT)",
    53: "53 (DNS)",
    22: "22 (SSH)",
    80: "80 (HTTP)",
    443: "443 (HTTPS)",
    21: "21 (FTP)",
    25: "25 (SMTP)",
    8080: "8080 (HTTP-Alt)",
    8883: "8883 (MQTTS)",
    3389: "3389 (RDP)",
    3306: "3306 (MySQL)"
}


def style_dark_axis(fig, ax, title="", xlabel="", ylabel=""):
    """Apply high-contrast cybersecurity dark theme to Matplotlib axis."""
    fig.patch.set_facecolor('#0E1626')
    ax.set_facecolor('#0B1120')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color='#E2E8F0', fontsize=10, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, color='#E2E8F0', fontsize=10, fontweight='bold')
    if title:
        ax.set_title(title, color='#FFFFFF', fontsize=11, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_color('#1E293B')
    ax.grid(True, color='#1E293B', linestyle='--', alpha=0.6)


def plot_verdict_donut(normal_count: int, attack_count: int, title="Threat Classification Breakdown"):
    """Render high-contrast donut chart for Normal vs Malicious flow verdicts."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.patch.set_facecolor('#0E1626')
    ax.set_facecolor('#0E1626')
    labels = ["Normal (0)", "Attack Alert (1)"]
    sizes = [normal_count, attack_count]
    colors = ["#10B981", "#EF4444"]
    if sum(sizes) == 0:
        sizes = [1, 0]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, textprops={'color': '#E2E8F0', 'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'width': 0.45, 'edgecolor': '#0E1626', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_color('#FFFFFF')
        autotext.set_fontsize(9)
    ax.set_title(title, color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_severity_bar(severity_counts: Dict[str, int], title="Alert Severity Distribution"):
    """Render color-coded bar chart of alert severity tiers."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="Severity Tier", ylabel="Flow Count")
    order = ["BENIGN", "ELEVATED", "MEDIUM", "HIGH", "CRITICAL"]
    palette = {"BENIGN": "#10B981", "ELEVATED": "#FBBF24", "MEDIUM": "#F59E0B", "HIGH": "#EF4444", "CRITICAL": "#991B1B"}
    vals = [severity_counts.get(k, 0) for k in order]
    cols = [palette[k] for k in order]
    bars = ax.bar(order, vals, color=cols, edgecolor='#1E293B', linewidth=1)
    max_val = max(vals) if vals and max(vals) > 0 else 1
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + (max_val * 0.02), f"{val:,}", ha='center', va='bottom', color='#E2E8F0', fontsize=8, fontweight='bold')
    ax.set_ylim(0, max_val * 1.20)
    plt.tight_layout()
    return fig


def plot_subtypes_bar(subtype_counts: Dict[str, int], title="Identified Threat Subtypes"):
    """Render horizontal bar chart for attack taxonomy classification."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="Flow Count", ylabel="")
    items = sorted(subtype_counts.items(), key=lambda x: x[1])
    labels = [k.replace('_', ' ').title() for k, v in items]
    vals = [v for k, v in items]
    total = sum(vals) if sum(vals) > 0 else 1
    colors = ["#10B981" if "Benign" in lbl else "#EF4444" for lbl in labels]
    bars = ax.barh(labels, vals, color=colors, edgecolor='#1E293B', linewidth=1)
    max_val = max(vals) if vals and max(vals) > 0 else 1
    for bar, val in zip(bars, vals):
        pct = (val / total) * 100
        ax.text(val + (max_val * 0.02), bar.get_y() + bar.get_height()/2, f"{val:,} ({pct:.1f}%)", ha='left', va='center', color='#E2E8F0', fontsize=8, fontweight='bold')
    ax.set_xlim(0, max_val * 1.35)
    plt.tight_layout()
    return fig


def plot_prob_distribution(prob_series: pd.Series, threshold=0.50, title="Threat Probability Density P(Attack)"):
    """Render probability distribution histogram with configurable decision threshold line."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="P(Attack)", ylabel="Flow Count")
    n, bins, patches = ax.hist(prob_series, bins=np.linspace(0, 1, 21), edgecolor='#0E1626', linewidth=1)
    for bin_left, patch in zip(bins[:-1], patches):
        if bin_left >= threshold:
            patch.set_facecolor('#EF4444')
        elif bin_left >= 0.20:
            patch.set_facecolor('#F59E0B')
        else:
            patch.set_facecolor('#10B981')
    ax.axvline(threshold, color='#38BDF8', linestyle='--', linewidth=2, label=f"Decision Threshold (\u03c4 = {threshold:.2f})")
    ax.legend(facecolor='#0E1626', edgecolor='#1E293B', labelcolor='#E2E8F0', fontsize=8)
    plt.tight_layout()
    return fig


def plot_protocol_donut(proto_series: pd.Series, title="Transport Protocol Composition"):
    """Render protocol breakdown donut chart."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.patch.set_facecolor('#0E1626')
    ax.set_facecolor('#0E1626')
    counts = proto_series.value_counts()
    proto_names = {6: "TCP (6)", 17: "UDP (17)", 1: "ICMP (1)"}
    labels = [proto_names.get(int(p), f"Proto {p}") for p in counts.index]
    palette = ["#06B6D4", "#8B5CF6", "#F59E0B", "#10B981", "#EC4899"]
    colors = [palette[i % len(palette)] for i in range(len(counts))]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, textprops={'color': '#E2E8F0', 'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'width': 0.45, 'edgecolor': '#0E1626', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_color('#FFFFFF')
        at.set_fontsize(9)
    ax.set_title(title, color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_top_ports_bar(port_series: pd.Series, title="Top Destination Ports"):
    """Render bar chart of destination ports annotated with network service names."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="Flow Count", ylabel="Destination Port")
    counts = port_series.value_counts().head(8).sort_values(ascending=True)
    labels = [PORT_NAME_MAP.get(int(p), str(int(p))) for p in counts.index]
    vals = counts.values
    colors = ["#06B6D4" if "1883" in lbl else ("#8B5CF6" if "53" in lbl else "#38BDF8") for lbl in labels]
    bars = ax.barh(labels, vals, color=colors, edgecolor='#1E293B', linewidth=1)
    max_val = max(vals) if len(vals) > 0 and max(vals) > 0 else 1
    for bar, val in zip(bars, vals):
        ax.text(val + (max_val * 0.02), bar.get_y() + bar.get_height()/2, f"{val:,}", ha='left', va='center', color='#E2E8F0', fontsize=8, fontweight='bold')
    ax.set_xlim(0, max_val * 1.25)
    plt.tight_layout()
    return fig


def plot_packet_lengths(df: pd.DataFrame, title="Packet Length Dynamics"):
    """Render forward vs backward packet length distributions."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="Packet Length (bytes)", ylabel="Flow Count")
    if "fwd_mean_pkt_len" in df.columns:
        ax.hist(df["fwd_mean_pkt_len"].dropna(), bins=15, alpha=0.6, color="#06B6D4", label="Forward Mean Len", edgecolor='#0E1626')
    if "bwd_mean_pkt_len" in df.columns:
        ax.hist(df["bwd_mean_pkt_len"].dropna(), bins=15, alpha=0.6, color="#A855F7", label="Backward Mean Len", edgecolor='#0E1626')
    ax.legend(facecolor='#0E1626', edgecolor='#1E293B', labelcolor='#E2E8F0', fontsize=8)
    plt.tight_layout()
    return fig


def plot_ratios_distribution(df: pd.DataFrame, title="Flow Dynamics & Asymmetry Ratios (Log Scale)"):
    """Render boxplots of flow asymmetry ratios capturing bidirectional behavior."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="Ratio Feature", ylabel="Log10 Value")
    ratio_cols = [c for c in ["pkt_ratio", "byte_ratio", "bytes_per_pkt", "pkt_len_ratio"] if c in df.columns]
    if ratio_cols:
        plot_data = []
        labels = []
        for col in ratio_cols:
            vals = np.log10(df[col].replace(0, np.nan).dropna() + 1e-6)
            if not vals.empty:
                plot_data.append(vals)
                labels.append(col)
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=False)
            colors = ["#06B6D4", "#38BDF8", "#10B981", "#A855F7"]
            for patch, col in zip(bp['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
                patch.set_edgecolor('#FFFFFF')
            for median in bp['medians']:
                median.set_color('#FFFFFF')
                median.set_linewidth(1.5)
            for whisker in bp['whiskers']:
                whisker.set_color('#94A3B8')
            for cap in bp['caps']:
                cap.set_color('#94A3B8')
    plt.tight_layout()
    return fig


def plot_tcp_flags(df: pd.DataFrame, title="TCP Control Flag Activity"):
    """Render forward vs backward TCP control flag rates."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_dark_axis(fig, ax, title=title, xlabel="Control Flag", ylabel="Mean Flags per Flow")
    flag_cols = [c for c in ["fwd_num_psh_flags", "bwd_num_psh_flags", "fwd_num_rst_flags", "bwd_num_rst_flags"] if c in df.columns]
    if flag_cols:
        means = [float(df[c].mean()) for c in flag_cols]
        labels = ["Fwd PSH", "Bwd PSH", "Fwd RST", "Bwd RST"]
        colors = ["#38BDF8", "#06B6D4", "#EF4444", "#DC2626"]
        bars = ax.bar(labels, means, color=colors, edgecolor='#1E293B', linewidth=1)
        max_val = max(means) if means and max(means) > 0 else 1
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, val + (max_val * 0.02), f"{val:.2f}", ha='center', va='bottom', color='#E2E8F0', fontsize=8, fontweight='bold')
        ax.set_ylim(0, max_val * 1.25)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title="Feature Correlation Matrix (Spearman)"):
    """Render Spearman rank correlation heatmap across key flow dynamics."""
    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor('#0E1626')
    ax.set_facecolor('#0B1120')
    candidate_cols = [
        "fwd_num_pkts", "bwd_num_pkts", "fwd_num_bytes", "bwd_num_bytes",
        "fwd_mean_iat", "bwd_mean_iat", "pkt_ratio", "byte_ratio",
        "bytes_per_pkt", "fwd_num_psh_flags", "bwd_num_rst_flags"
    ]
    present_cols = [c for c in candidate_cols if c in df.columns]
    if len(present_cols) >= 3 and len(df) >= 3:
        corr = df[present_cols].corr(method='spearman')
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f', cmap='vlag', center=0,
            linewidths=0.5, linecolor='#0E1626', ax=ax,
            cbar_kws={'label': 'Spearman Correlation'},
            annot_kws={'size': 7, 'color': '#FFFFFF'}
        )
        ax.tick_params(colors='#94A3B8', labelsize=8)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='#94A3B8')
        cbar.ax.yaxis.label.set_color('#E2E8F0')
    ax.set_title(title, color='#FFFFFF', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title="Ground Truth vs Model Verdict"):
    """Render confusion matrix comparison heatmap."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.patch.set_facecolor('#0E1626')
    ax.set_facecolor('#0B1120')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Benign (0)', 'Attack (1)'],
        yticklabels=['Benign (0)', 'Attack (1)'],
        cbar=False, annot_kws={'size': 11, 'fontweight': 'bold'}
    )
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.set_xlabel('Model Predicted Label', color='#E2E8F0', fontsize=10, fontweight='bold')
    ax.set_ylabel('Ground Truth Label', color='#E2E8F0', fontsize=10, fontweight='bold')
    ax.set_title(title, color='#FFFFFF', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


@st.cache_resource(show_spinner="Loading ML-NIDS Pipeline and SHAP Explainer...")
def load_pipeline() -> NIDSPredictionPipeline:
    return NIDSPredictionPipeline()


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.sidebar.title("ML-NIDS SOC Console")
    st.sidebar.markdown("**Explainable Threat Detection**")
    st.sidebar.markdown("---")

    # Load Pipeline
    try:
        pipeline = load_pipeline()
        st.sidebar.success(f"Model Active (v1.0.0)")
        st.sidebar.caption(f"34 features • 150 trees • TreeExplainer")
    except Exception as e:
        st.sidebar.error(f"Initialization Error: {e}")
        st.stop()

    # Sidebar Operational Controls
    st.sidebar.subheader("Detection Controls")
    threshold = st.sidebar.slider(
        "Decision Threshold (tau)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Alert probability cut-off. Lower threshold (<0.5) prioritizes high recall; higher threshold (>0.5) suppresses false positives."
    )

    # Primary Navigation Tabs (Matching Stages A-F)
    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs([
        "A. Overview",
        "B. Dataset Analysis",
        "C. Threat Detection",
        "D. Alert Investigation & XAI",
        "E. SOC Reports",
        "F. Model Information"
    ])

    # ========================================================
    # TAB A: OVERVIEW & SYSTEM TELEMETRY
    # ========================================================
    with tab_a:
        st.header("ML-NIDS & Explainable AI Architecture")
        st.markdown(
            "An academically rigorous, reproducible Machine Learning Network Intrusion Detection System "
            "powered by a **Random Forest ensemble** with **SHAP TreeExplainer** directional attribution."
        )

        # Top KPI Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("""<div class="metric-card"><div class="metric-lbl">Holdout Accuracy</div><div class="metric-val" style="color:#10B981">99.94%</div><span style="font-size:0.75rem;color:#64748B">38,907 test flows</span></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card"><div class="metric-lbl">Precision</div><div class="metric-val">99.89%</div><span style="font-size:0.75rem;color:#64748B">12 False Positives</span></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card"><div class="metric-lbl">Recall (Sensitivity)</div><div class="metric-val">99.89%</div><span style="font-size:0.75rem;color:#64748B">12 False Negatives</span></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="metric-card"><div class="metric-lbl">ROC-AUC Score</div><div class="metric-val" style="color:#A855F7">0.999997</div><span style="font-size:0.75rem;color:#64748B">PR-AUC: 0.999992</span></div>""", unsafe_allow_html=True)
        with col5:
            st.markdown("""<div class="metric-card"><div class="metric-lbl">Inference Latency</div><div class="metric-val" style="color:#F59E0B">5.03 &mu;s</div><span style="font-size:0.75rem;color:#64748B">198,700 flows/sec</span></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Operational Distinction Disclaimer
        st.info(
            "**Operational Distinction Notice:** "
            "• **Offline Evaluation:** Metrics above reflect held-out test traffic with fixed class balance (72.6% Benign / 27.4% Malicious).\n"
            "• **Live Network Detection:** In production, base-rate fallacies apply (attacks typically <0.1% of flows). "
            "A model with 0.04% FPR will still produce occasional false alerts that require analyst verification via the XAI investigation tab."
        )

        st.subheader("Evaluated Attack Taxonomy")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**MQTT Brute Force**\n* Automated credential attacks targeting IoT brokers on TCP port 1883.")
        c2.markdown("**Nmap ACK Scan (-sA)**\n* TCP ACK probes mapping firewall statefulness, eliciting 40-byte RST packets.")
        c3.markdown("**Nmap UDP Scan (-sU)**\n* Connectionless sweeps against UDP ports provoking ICMP port unreachables.")
        c4.markdown("**SPARTA Reconnaissance**\n* Automated vulnerability assessment generating high-frequency PSH/RST bursts.")

    # ========================================================
    # TAB B: DATASET ANALYSIS & QUALITY AUDITING
    # ========================================================
    with tab_b:
        st.header("Dataset Intelligence, Quality Audit & Dynamic Visual Analysis")
        st.markdown(
            "Upload any flow dataset, select the bundled benchmark (`sample_flows.csv`), or paste flow records. "
            "The system automatically verifies schema conformance, executes machine learning inference, "
            "and generates comprehensive analytical diagrams and visualizations based on the given data."
        )

        eda_mode = st.radio(
            "Select Dataset Source:",
            [
                "⚡ Built-In Sample Flows (`sample_flows.csv`)",
                "📂 Upload Feature CSV File",
                "📋 Paste CSV Flow Data"
            ],
            horizontal=True,
            key="eda_mode_selection"
        )

        df_eda = None
        source_name = ""

        if eda_mode == "⚡ Built-In Sample Flows (`sample_flows.csv`)":
            sample_path = ROOT_DIR / "sample_flows.csv"
            if sample_path.exists():
                try:
                    df_eda = pd.read_csv(sample_path)
                    source_name = "sample_flows.csv (Bundled Benchmark)"
                    st.success(f"Loaded bundled benchmark `{source_name}` with {len(df_eda)} network flows.")
                except Exception as e:
                    st.error(f"Error reading sample_flows.csv: {e}")
            else:
                st.error("sample_flows.csv not found in repository root.")

        elif eda_mode == "📂 Upload Feature CSV File":
            uploaded_eda = st.file_uploader("Upload Feature CSV for Quality Audit & Visualization", type=["csv"], key="eda_upload")
            if uploaded_eda is not None:
                try:
                    df_eda = pd.read_csv(uploaded_eda)
                    source_name = uploaded_eda.name
                    st.success(f"Loaded `{source_name}`: {len(df_eda):,} rows, {len(df_eda.columns)} columns.")
                except Exception as e:
                    st.error(f"Error reading uploaded CSV: {e}")

        elif eda_mode == "📋 Paste CSV Flow Data":
            pasted_csv = st.text_area(
                "Paste CSV Flow Rows (including header line):",
                height=150,
                placeholder="bwd_max_iat,bwd_max_pkt_len,bwd_mean_iat,...\n0.00092,60,0.00035,..."
            )
            if pasted_csv.strip():
                try:
                    df_eda = pd.read_csv(io.StringIO(pasted_csv.strip()))
                    source_name = "Pasted Flow Dataset"
                    st.success(f"Parsed pasted flow dataset: {len(df_eda):,} rows, {len(df_eda.columns)} columns.")
                except Exception as e:
                    st.error(f"Error parsing pasted CSV text: {e}")

        if df_eda is not None and not df_eda.empty:
            # 1. Quality & Schema Auditing
            st.subheader(f"1. Schema Conformance & Telemetry ({source_name})")
            
            missing_cells = int(df_eda.isna().sum().sum())
            inf_cells = int(np.isinf(df_eda.select_dtypes(include=np.number)).sum().sum())
            dup_rows = int(df_eda.duplicated().sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df_eda):,}")
            c2.metric("Missing (NaN) Cells", f"{missing_cells:,}")
            c3.metric("Infinite Values", f"{inf_cells:,}")
            c4.metric("Duplicate Flows", f"{dup_rows:,}")

            # Schema Verification
            missing_feats = [f for f in pipeline.expected_features if f not in df_eda.columns]
            has_raw_biflow = all(c in df_eda.columns for c in ["fwd_num_pkts", "bwd_num_pkts", "fwd_num_bytes"])

            if not missing_feats:
                st.success("✅ **Schema Verification Clean:** Exact 34 engineered model features present.")
            elif has_raw_biflow:
                st.info("ℹ️ **Raw Biflow Schema Detected:** Asymmetry ratios (pkt_ratio, byte_ratio, etc.) will be dynamically calculated.")
            else:
                st.warning(f"⚠️ Missing model features: {missing_feats[:5]}... Pipeline will attempt automated imputation.")

            # 2. Run Pipeline Prediction for Complete Threat Analytics
            with st.spinner("Analyzing dataset dynamics and evaluating threat classifications..."):
                try:
                    results_eda = pipeline.predict_dataframe(df_eda, threshold=threshold)
                except Exception as e:
                    st.error(f"Pipeline evaluation error: {e}")
                    results_eda = None

            if results_eda is not None and not results_eda.empty:
                tot_flows = len(results_eda)
                alert_count = int((results_eda["is_alert"] == 1).sum())
                normal_count = tot_flows - alert_count
                alert_rate = (alert_count / tot_flows) * 100 if tot_flows > 0 else 0
                crit_count = int((results_eda["severity"] == "CRITICAL").sum())
                high_count = int((results_eda["severity"] == "HIGH").sum())

                # Threat Telemetry Cards
                st.markdown("---")
                st.subheader("2. AI Threat Intelligence & Risk Summary")
                t_c1, t_c2, t_c3, t_c4 = st.columns(4)
                t_c1.metric("Normal / Benign Flows", f"{normal_count:,}", f"{(normal_count/tot_flows)*100:.1f}%")
                t_c2.metric("Flagged Malicious Alerts", f"{alert_count:,}", f"{alert_rate:.1f}% rate", delta_color="inverse")
                t_c3.metric("Critical & High Alerts", f"{crit_count + high_count:,}")
                top_port = results_eda["prt_dst"].mode().iloc[0] if "prt_dst" in results_eda.columns and not results_eda["prt_dst"].empty else "N/A"
                top_port_name = PORT_NAME_MAP.get(int(top_port), str(top_port)) if top_port != "N/A" else "N/A"
                t_c4.metric("Dominant Target Port", f"{top_port_name}")

                # 3. Dynamic Visualizations & Diagrams
                st.markdown("---")
                st.subheader("3. Comprehensive Visual Analysis & Telemetry Diagrams")
                
                vtab1, vtab2, vtab3, vtab4 = st.tabs([
                    "🎯 Threat Verdict & Subtypes",
                    "🌐 Protocols & Target Ports",
                    "⚡ Traffic Volume & Asymmetry Ratios",
                    "🔬 TCP Flags & Feature Correlations"
                ])

                with vtab1:
                    st.markdown("#### Threat Detection Breakdown & Confidence Spectrum")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        fig_donut = plot_verdict_donut(normal_count, alert_count, title="Threat Classification Verdict")
                        st.pyplot(fig_donut)
                        plt.close(fig_donut)
                    with v_col2:
                        sev_counts = results_eda["severity"].value_counts().to_dict()
                        fig_sev = plot_severity_bar(sev_counts, title="Alert Severity Distribution")
                        st.pyplot(fig_sev)
                        plt.close(fig_sev)

                    v_col3, v_col4 = st.columns(2)
                    with v_col3:
                        st_counts = results_eda["attack_subtype"].value_counts().to_dict()
                        fig_sub = plot_subtypes_bar(st_counts, title="Predicted Attack Subtypes")
                        st.pyplot(fig_sub)
                        plt.close(fig_sub)
                    with v_col4:
                        fig_prob = plot_prob_distribution(results_eda["prob_attack"], threshold=threshold, title="Threat Probability Density P(Attack)")
                        st.pyplot(fig_prob)
                        plt.close(fig_prob)

                    # Ground truth comparison if label column present
                    gt_col = "target_is_attack" if "target_is_attack" in df_eda.columns else ("is_attack" if "is_attack" in df_eda.columns else None)
                    if gt_col:
                        st.markdown("---")
                        st.markdown("#### Ground Truth Label vs Model Predictions")
                        gt_c1, gt_c2 = st.columns(2)
                        with gt_c1:
                            val_counts = df_eda[gt_col].value_counts()
                            fig_gt, ax_gt = plt.subplots(figsize=(5, 3.2))
                            style_dark_axis(fig_gt, ax_gt, title="Ground Truth Class Distribution", xlabel="Class", ylabel="Flow Count")
                            bars_gt = ax_gt.bar(["Benign (0)", "Attack (1)"], [val_counts.get(0, 0), val_counts.get(1, 0)], color=["#10B981", "#EF4444"])
                            for bar, v in zip(bars_gt, [val_counts.get(0, 0), val_counts.get(1, 0)]):
                                ax_gt.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:,}", ha='center', va='bottom', color='#FFFFFF', fontsize=9, fontweight='bold')
                            st.pyplot(fig_gt)
                            plt.close(fig_gt)
                        with gt_c2:
                            fig_cm = plot_confusion_matrix(df_eda[gt_col], results_eda["is_alert"], title="Model Evaluation Matrix on Provided Dataset")
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)

                with vtab2:
                    st.markdown("#### Transport Layer & Service Telemetry")
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        if "proto" in results_eda.columns:
                            fig_proto = plot_protocol_donut(results_eda["proto"], title="Transport Layer Protocols")
                            st.pyplot(fig_proto)
                            plt.close(fig_proto)
                        else:
                            st.info("Protocol column not present in dataset.")
                    with p_col2:
                        if "prt_dst" in results_eda.columns:
                            fig_port = plot_top_ports_bar(results_eda["prt_dst"], title="Top Targeted Destination Ports")
                            st.pyplot(fig_port)
                            plt.close(fig_port)
                        else:
                            st.info("Destination port column not present in dataset.")

                with vtab3:
                    st.markdown("#### Flow Volume, Packet Dynamics & Asymmetry Ratios")
                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        fig_pkt = plot_packet_lengths(results_eda, title="Packet Length Dynamics")
                        st.pyplot(fig_pkt)
                        plt.close(fig_pkt)
                    with r_col2:
                        fig_ratios = plot_ratios_distribution(results_eda, title="Flow Asymmetry Ratios (Log Scale)")
                        st.pyplot(fig_ratios)
                        plt.close(fig_ratios)

                with vtab4:
                    st.markdown("#### TCP Control Flags & Feature Correlations")
                    c_col1, c_col2 = st.columns([1, 1])
                    with c_col1:
                        fig_flags = plot_tcp_flags(results_eda, title="TCP Control Flag Activity")
                        st.pyplot(fig_flags)
                        plt.close(fig_flags)
                    with c_col2:
                        if len(results_eda) >= 3:
                            fig_corr = plot_correlation_heatmap(results_eda, title="Feature Correlation Matrix")
                            st.pyplot(fig_corr)
                            plt.close(fig_corr)
                        else:
                            st.info("At least 3 flow records are required to calculate correlation matrix.")

                # 4. Interactive Data Table & Seamless Handoff
                st.markdown("---")
                st.subheader("4. Analyzed Flow Records & Investigation Queue")
                filter_choice = st.selectbox(
                    "Filter Records:",
                    ["All Flows", "Flagged Attacks Only", "Normal Traffic Only"],
                    key="eda_filter"
                )
                filtered_df = results_eda
                if filter_choice == "Flagged Attacks Only":
                    filtered_df = results_eda[results_eda["is_alert"] == 1]
                elif filter_choice == "Normal Traffic Only":
                    filtered_df = results_eda[results_eda["is_alert"] == 0]

                display_cols = [c for c in ["predicted_label", "prob_attack", "severity", "attack_subtype", "prt_dst", "proto", "total_pkts", "total_bytes"] if c in filtered_df.columns]
                st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)

                # Button to transfer to active detection session
                col_btn, _ = st.columns([2, 3])
                with col_btn:
                    if st.button("🚀 Load Analyzed Dataset into Threat Detection & XAI Investigation", type="primary"):
                        st.session_state["active_results"] = results_eda
                        st.session_state["active_input_df"] = df_eda
                        st.session_state["inference_completed"] = True
                        st.success("✅ Dataset loaded! Switch to Tab C (Threat Detection) or Tab D (Alert Investigation & XAI) to inspect any individual flow!")
        else:
            st.info("Select or upload a dataset above to generate complete analysis diagrams and threat evaluations.")
            summary_path = ROOT_DIR / "artifacts" / "reports" / "eda_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    eda_json = json.load(f)
                st.subheader("Pre-Computed Benchmark Dataset Inventory")
                st.json(eda_json.get("biflow_summary", {}))

    # ========================================================
    # TAB C: THREAT DETECTION & INFERENCE
    # ========================================================
    with tab_c:
        st.header("Threat Detection & Stream Ingestion")
        st.markdown("Run live inference using pre-loaded curated test cases, uploaded feature CSVs, raw PCAP captures, or benchmark datasets.")

        ingest_mode = st.radio(
            "Select Ingestion Source:",
            [
                "Curated Test Cases",
                "⚡ Load Built-In Sample Flows (`sample_flows.csv`)",
                "Upload CSV Features",
                "📋 Paste CSV Flow Data",
                "Upload Raw PCAP Capture"
            ],
            horizontal=True
        )

        results_df = None
        input_df = None

        if ingest_mode == "Curated Test Cases":
            st.subheader("Select Curated Test Case")
            scenarios = {
                "Normal MQTT Telemetry": {
                    "prt_dst": 1883, "proto": 6, "fwd_num_pkts": 7, "bwd_num_pkts": 5,
                    "fwd_mean_pkt_len": 63.4, "bwd_mean_pkt_len": 54.0, "fwd_min_pkt_len": 52.0, "bwd_min_pkt_len": 52.0,
                    "fwd_max_pkt_len": 98.0, "bwd_max_pkt_len": 60.0, "fwd_num_bytes": 444, "bwd_num_bytes": 270,
                    "fwd_mean_iat": 0.00024, "bwd_mean_iat": 0.00031, "fwd_std_iat": 0.0001, "bwd_std_iat": 0.0001,
                    "fwd_min_iat": 0.00005, "bwd_min_iat": 0.00008, "fwd_max_iat": 0.0008, "bwd_max_iat": 0.0009,
                    "fwd_std_pkt_len": 18.2, "bwd_std_pkt_len": 3.8, "fwd_num_psh_flags": 3, "bwd_num_psh_flags": 1,
                    "fwd_num_rst_flags": 0, "bwd_num_rst_flags": 0, "fwd_num_urg_flags": 0, "bwd_num_urg_flags": 0
                },
                "MQTT Credential Brute Force": {
                    "prt_dst": 53, "proto": 17, "fwd_num_pkts": 2, "bwd_num_pkts": 2,
                    "fwd_mean_pkt_len": 83.0, "bwd_mean_pkt_len": 83.0, "fwd_min_pkt_len": 83.0, "bwd_min_pkt_len": 83.0,
                    "fwd_max_pkt_len": 83.0, "bwd_max_pkt_len": 83.0, "fwd_num_bytes": 166, "bwd_num_bytes": 166,
                    "fwd_mean_iat": 0.000008, "bwd_mean_iat": 0.000008, "fwd_std_iat": 0.0, "bwd_std_iat": 0.0,
                    "fwd_min_iat": 0.000008, "bwd_min_iat": 0.000008, "fwd_max_iat": 0.000008, "bwd_max_iat": 0.000008,
                    "fwd_std_pkt_len": 0.0, "bwd_std_pkt_len": 0.0, "fwd_num_psh_flags": 0, "bwd_num_psh_flags": 0,
                    "fwd_num_rst_flags": 0, "bwd_num_rst_flags": 0, "fwd_num_urg_flags": 0, "bwd_num_urg_flags": 0
                },
                "Nmap TCP ACK Scan (-sA)": {
                    "prt_dst": 443, "proto": 6, "fwd_num_pkts": 1, "bwd_num_pkts": 1,
                    "fwd_mean_pkt_len": 44.0, "bwd_mean_pkt_len": 40.0, "fwd_min_pkt_len": 44.0, "bwd_min_pkt_len": 40.0,
                    "fwd_max_pkt_len": 44.0, "bwd_max_pkt_len": 40.0, "fwd_num_bytes": 44, "bwd_num_bytes": 40,
                    "fwd_mean_iat": 0.0, "bwd_mean_iat": 0.0, "fwd_std_iat": 0.0, "bwd_std_iat": 0.0,
                    "fwd_min_iat": 0.0, "bwd_min_iat": 0.0, "fwd_max_iat": 0.0, "bwd_max_iat": 0.0,
                    "fwd_std_pkt_len": 0.0, "bwd_std_pkt_len": 0.0, "fwd_num_psh_flags": 0, "bwd_num_psh_flags": 0,
                    "fwd_num_rst_flags": 0, "bwd_num_rst_flags": 1, "fwd_num_urg_flags": 0, "bwd_num_urg_flags": 0
                },
                "SPARTA Reconnaissance & Brute Force": {
                    "prt_dst": 22, "proto": 6, "fwd_num_pkts": 737, "bwd_num_pkts": 691,
                    "fwd_mean_pkt_len": 110.6, "bwd_mean_pkt_len": 95.2, "fwd_min_pkt_len": 40.0, "bwd_min_pkt_len": 40.0,
                    "fwd_max_pkt_len": 1420.0, "bwd_max_pkt_len": 1420.0, "fwd_num_bytes": 81512, "bwd_num_bytes": 65783,
                    "fwd_mean_iat": 8.05, "bwd_mean_iat": 7.91, "fwd_std_iat": 12.4, "bwd_std_iat": 11.8,
                    "fwd_min_iat": 0.0001, "bwd_min_iat": 0.0001, "fwd_max_iat": 45.2, "bwd_max_iat": 44.9,
                    "fwd_std_pkt_len": 240.5, "bwd_std_pkt_len": 210.1, "fwd_num_psh_flags": 404, "bwd_num_psh_flags": 312,
                    "fwd_num_rst_flags": 18, "bwd_num_rst_flags": 24, "fwd_num_urg_flags": 0, "bwd_num_urg_flags": 0
                }
            }

            selected_sc = st.selectbox("Choose Traffic Scenario:", list(scenarios.keys()))
            sample_feats = scenarios[selected_sc]
            input_df = pd.DataFrame([sample_feats])
            results_df = pipeline.predict_dataframe(input_df, threshold=threshold)

        elif ingest_mode == "⚡ Load Built-In Sample Flows (`sample_flows.csv`)":
            sample_path = ROOT_DIR / "sample_flows.csv"
            if sample_path.exists():
                try:
                    input_df = pd.read_csv(sample_path)
                    results_df = pipeline.predict_dataframe(input_df, threshold=threshold)
                    st.success(f"Successfully evaluated {len(results_df):,} flows from bundled `sample_flows.csv`.")
                except Exception as e:
                    st.error(f"Inference error on sample_flows.csv: {e}")
            else:
                st.error("sample_flows.csv not found in repository root.")

        elif ingest_mode == "Upload CSV Features":
            uploaded_file = st.file_uploader("Upload CSV Flow File", type=["csv"], key="infer_csv")
            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    results_df = pipeline.predict_dataframe(input_df, threshold=threshold)
                    st.success(f"Successfully processed {len(results_df):,} flows from `{uploaded_file.name}`.")
                except Exception as e:
                    st.error(f"Inference error: {e}")

        elif ingest_mode == "📋 Paste CSV Flow Data":
            pasted_infer = st.text_area(
                "Paste Flow CSV Text to Ingest:",
                height=150,
                placeholder="bwd_max_iat,bwd_max_pkt_len,bwd_mean_iat,...\n0.00092,60,0.00035,..."
            )
            if pasted_infer.strip():
                try:
                    input_df = pd.read_csv(io.StringIO(pasted_infer.strip()))
                    results_df = pipeline.predict_dataframe(input_df, threshold=threshold)
                    st.success(f"Successfully evaluated {len(results_df):,} pasted flow records.")
                except Exception as e:
                    st.error(f"Pasted CSV evaluation error: {e}")

        elif ingest_mode == "Upload Raw PCAP Capture":
            uploaded_pcap = st.file_uploader("Upload PCAP File (.pcap)", type=["pcap"], key="infer_pcap")
            max_pkts = st.number_input("Max Packets to Read", min_value=100, max_value=50000, value=5000, step=500)
            if uploaded_pcap is not None:
                with st.spinner("Dissecting PCAP packets into bidirectional flows..."):
                    try:
                        pcap_bytes = uploaded_pcap.read()
                        results_df, input_df = pipeline.process_pcap(pcap_bytes, max_packets=int(max_pkts), threshold=threshold)
                        if results_df.empty:
                            st.warning("No IP/TCP/UDP sessions detected in PCAP.")
                        else:
                            st.success(f"Dissected {len(results_df)} flows from PCAP.")
                    except Exception as e:
                        st.error(f"PCAP Extraction Error: {e}")

        # Display Inference Output & Telemetry Diagrams
        if results_df is not None and not results_df.empty:
            st.session_state["active_results"] = results_df
            if input_df is not None and not input_df.empty:
                st.session_state["active_input_df"] = input_df

            # Telemetry Summary Cards
            tot = len(results_df)
            alerts = int((results_df["is_alert"] == 1).sum())
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Total Flows", f"{tot:,}")
            c_b.metric("Flagged Alerts", f"{alerts:,}")
            c_c.metric("Alert Rate", f"{alerts/tot*100:.2f}%")

            # Threat Visual Analysis Diagrams
            st.markdown("#### Threat Ingestion Telemetry & Diagrams")
            diag_c1, diag_c2 = st.columns(2)
            with diag_c1:
                fig_verdict = plot_verdict_donut(tot - alerts, alerts, title="Classification Verdict")
                st.pyplot(fig_verdict)
                plt.close(fig_verdict)
            with diag_c2:
                sev_counts = results_df["severity"].value_counts().to_dict()
                fig_sev_ingest = plot_severity_bar(sev_counts, title="Alert Severity Tiers")
                st.pyplot(fig_sev_ingest)
                plt.close(fig_sev_ingest)

            diag_c3, diag_c4 = st.columns(2)
            with diag_c3:
                st_counts = results_df["attack_subtype"].value_counts().to_dict()
                fig_st_ingest = plot_subtypes_bar(st_counts, title="Attack Subtype Distribution")
                st.pyplot(fig_st_ingest)
                plt.close(fig_st_ingest)
            with diag_c4:
                fig_p_ingest = plot_prob_distribution(results_df["prob_attack"], threshold=threshold, title="Threat Probability Density")
                st.pyplot(fig_p_ingest)
                plt.close(fig_p_ingest)

            # Result Dataframe View
            st.subheader("Inference Results Table")
            display_cols = [c for c in ["predicted_label", "prob_attack", "severity", "attack_subtype", "prt_dst", "proto", "total_pkts", "total_bytes"] if c in results_df.columns]
            st.dataframe(results_df[display_cols].head(100), use_container_width=True)

            # Store for XAI tab
            st.session_state["inference_completed"] = True

    # ========================================================
    # TAB D: ALERT INVESTIGATION & LOCAL SHAP XAI
    # ========================================================
    with tab_d:
        st.header("Alert Investigation & Local Explainable AI")
        st.markdown(
            "Select any flow record to calculate exact **SHAP TreeExplainer attributions**, "
            "inspect positive and negative driving features, and review an automated SOC briefing."
        )

        active_df = st.session_state.get("active_results", None)
        active_input = st.session_state.get("active_input_df", None)

        if active_df is not None and not active_df.empty:
            alert_indices = active_df.index.tolist()
            selected_idx = st.selectbox(
                "Select Flow Record by Index:",
                alert_indices,
                format_func=lambda i: f"Row {i} — [{active_df.loc[i, 'predicted_label']}] P(Attack)={active_df.loc[i, 'prob_attack']*100:.1f}% ({active_df.loc[i, 'attack_subtype']})"
            )

            if st.button("Generate Local SHAP Explanation", type="primary"):
                with st.spinner("Computing Shapley values across 150 decision trees..."):
                    try:
                        # Prefer input schema if available, otherwise fall back to active_df (now carrying 34 features)
                        eval_source = active_input if active_input is not None and not active_input.empty else active_df
                        explanation = pipeline.explain_flow_by_index(eval_source, index=selected_idx, top_k=5)
                        st.session_state["current_explanation"] = explanation
                    except Exception as e:
                        st.error(f"Explanation failed: {e}")

            if "current_explanation" in st.session_state:
                exp = st.session_state["current_explanation"]

                # Verdict Banner
                is_atk = exp["predicted_class"] == 1
                badge_style = "alert-badge-critical" if is_atk else "alert-badge-benign"
                st.markdown(f"""
                <div style="padding:12px; border-radius:6px; margin:12px 0;" class="{badge_style}">
                    <h3>{exp['verdict']} — Confidence: {exp['confidence_scores']['attack_probability']*100:.1f}%</h3>
                    <p style="margin-top:4px;">{exp['executive_summary']}</p>
                </div>
                """, unsafe_allow_html=True)

                col_x, col_y = st.columns([3, 2])

                with col_x:
                    st.subheader("Local SHAP Attribution Plot (ΔP Contribution)")
                    top_items = exp["all_feature_contributions"][:10]
                    top_items = sorted(top_items, key=lambda x: x["shap_value"])

                    labels = [f"{item['feature']} ({item['value']})" for item in top_items]
                    values = [item["shap_value"] for item in top_items]
                    colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.barh(labels, values, color=colors)
                    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
                    ax.set_xlabel("Local SHAP Contribution to Threat Probability (phi_i)")
                    ax.set_title("Top 10 Feature Contributions for this Flow")
                    st.pyplot(fig)
                    plt.close()

                with col_y:
                    st.subheader("Automated SOC Analyst Briefing")
                    st.markdown("**Executive Interpretation:**")
                    for pt in exp["analyst_briefing_points"]:
                        st.markdown(f"• {pt}")

                    st.markdown("---")
                    st.caption(f"**Baseline Model Prior:** {exp['base_value_attack']*100:.1f}%")
                    st.caption(exp["disclaimer"])

                st.subheader("Feature vs Benign Baseline Comparison")
                comp_table = exp.get("baseline_comparison_table", [])
                if comp_table:
                    df_comp = pd.DataFrame(comp_table)
                    st.dataframe(df_comp, use_container_width=True)

        else:
            st.info("Run threat detection in Tab C first to generate flows for XAI investigation.")

    # ========================================================
    # TAB E: SOC REPORTS & SERIALIZATION
    # ========================================================
    with tab_e:
        st.header("SOC Incident Reports & Serialization")
        st.markdown("Export sanitized audit summaries adhering to cybersecurity data protection best practices.")

        active_df = st.session_state.get("active_results", None)

        if active_df is not None and not active_df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Export Predictions as CSV")
                csv_data = active_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Predictions",
                    data=csv_data,
                    file_name="nids_incident_predictions.csv",
                    mime="text/csv"
                )

            with c2:
                st.subheader("Export Executive JSON Triage Report")
                exps = [st.session_state.get("current_explanation")] if "current_explanation" in st.session_state else []
                report_json = pipeline.export_triage_report(active_df, explanations=exps, output_format="json")
                json_str = json.dumps(report_json, indent=2)
                st.download_button(
                    label="Download JSON Triage Report",
                    data=json_str,
                    file_name="nids_triage_report.json",
                    mime="application/json"
                )

            st.markdown("---")
            st.subheader("Report Preview")
            st.json(report_json)

        else:
            st.info("No active inference results available to export. Ingest traffic in Tab C.")

    # ========================================================
    # TAB F: MODEL INFORMATION & GOVERNANCE
    # ========================================================
    with tab_f:
        st.header("Model Information, Provenance & Governance")
        st.markdown("Transparent technical documentation detailing features, training parameters, and known operational limits.")

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.subheader("Model Specifications")
            st.markdown("""
            - **Algorithm:** Random Forest Classifier (Scikit-Learn)
            - **Ensemble Estimators:** 150 Decision Trees
            - **Criterion:** Gini Impurity with balanced sub-sample class weighting
            - **Feature Dimension:** 34 numerical features (28 sanitized baseline + 6 engineered asymmetry ratios)
            - **Leakage Prevention:** Complete exclusion of `ip_src`, `ip_dst`, and `prt_src`
            - **Explainability Backend:** SHAP TreeExplainer (Polynomial tree traversal)
            """)

            st.subheader("Operational Limitations & Attack Surface")
            st.markdown("""
            1. **Encrypted Payloads:** Operates purely on transport-layer metadata and flow dynamics; cannot inspect plaintext encrypted payload contents (TLS, SSH).
            2. **Adversarial Timing Attacks:** Attackers could introduce artificial packet delays to inflate `mean_iat` to mimic legitimate human traffic.
            3. **Base-Rate Sensitivity:** While FPR is 0.0425%, on high-throughput 10 Gbps links, even tiny false-alarm percentages require secondary analyst confirmation.
            """)

        with col_f2:
            st.subheader("Active 34-Feature Schema Dictionary")
            schema_data = []
            for feat in pipeline.expected_features:
                if feat in ["pkt_ratio", "byte_ratio", "bytes_per_pkt", "pkt_len_ratio", "total_pkts", "total_bytes"]:
                    ftype = "Engineered Asymmetry Ratio"
                elif "flags" in feat:
                    ftype = "TCP Control Flags"
                elif "iat" in feat:
                    ftype = "Inter-Arrival Timing (sec)"
                elif "pkt_len" in feat or "bytes" in feat:
                    ftype = "Packet Length / Volume (bytes)"
                else:
                    ftype = "Network Protocol Metadata"
                schema_data.append({"Feature Name": feat, "Type": ftype})

            st.dataframe(pd.DataFrame(schema_data), height=380, use_container_width=True)


if __name__ == "__main__":
    main()
