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
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
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
        st.header("Dataset Intelligence & Schema Auditing")
        st.markdown("Upload any pre-extracted flow feature CSV to inspect schema conformance, missing values, and class balance.")

        uploaded_eda = st.file_uploader("Upload Feature CSV for Quality Audit", type=["csv"], key="eda_upload")

        if uploaded_eda is not None:
            try:
                df_eda = pd.read_csv(uploaded_eda)
                st.success(f"Loaded `{uploaded_eda.name}`: {len(df_eda):,} rows, {len(df_eda.columns)} columns.")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Rows", f"{len(df_eda):,}")
                col_b.metric("Missing (NaN) Cells", f"{df_eda.isna().sum().sum():,}")
                col_c.metric("Infinite Values", f"{np.isinf(df_eda.select_dtypes(include=np.number)).sum().sum():,}")

                # Schema verification
                missing_feats = [f for f in pipeline.expected_features if f not in df_eda.columns]
                has_raw_biflow = all(c in df_eda.columns for c in ["fwd_num_pkts", "bwd_num_pkts", "fwd_num_bytes"])

                if not missing_feats:
                    st.success("Schema Conformance: Clean match with exact 34 engineered features.")
                elif has_raw_biflow:
                    st.info("Schema Conformance: Raw biflow format detected. Ratio features will be synthesized automatically.")
                else:
                    st.warning(f"Schema Warning: File is missing required feature columns: {missing_feats[:5]}...")

                # Class Distribution if Ground Truth Available
                if "is_attack" in df_eda.columns or "target_is_attack" in df_eda.columns:
                    lbl_col = "target_is_attack" if "target_is_attack" in df_eda.columns else "is_attack"
                    st.subheader("Trustworthy Label Distribution")
                    val_counts = df_eda[lbl_col].value_counts()

                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    colors = ["#10B981" if x == 0 else "#EF4444" for x in val_counts.index]
                    ax.bar(["Benign (0)", "Attack (1)"], val_counts.values, color=colors, width=0.5)
                    ax.set_ylabel("Flow Count")
                    ax.set_title(f"Ground Truth Class Balance in {uploaded_eda.name}")
                    for i, v in enumerate(val_counts.values):
                        ax.text(i, v + (max(val_counts.values)*0.02), f"{v:,} ({v/len(df_eda)*100:.1f}%)", ha='center', fontsize=9)
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        else:
            st.info("Upload a CSV file or load the benchmark summary below.")
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
        st.markdown("Run live inference using pre-loaded curated test cases, uploaded feature CSVs, or raw PCAP captures.")

        ingest_mode = st.radio(
            "Select Ingestion Source:",
            ["Curated Test Cases", "Upload CSV Features", "Upload Raw PCAP Capture"],
            horizontal=True
        )

        results_df = None

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
            df_curated = pd.DataFrame([sample_feats])
            results_df = pipeline.predict_dataframe(df_curated, threshold=threshold)

        elif ingest_mode == "Upload CSV Features":
            uploaded_file = st.file_uploader("Upload CSV Flow File", type=["csv"], key="infer_csv")
            if uploaded_file is not None:
                try:
                    df_raw = pd.read_csv(uploaded_file)
                    results_df = pipeline.predict_dataframe(df_raw, threshold=threshold)
                    st.success(f"Successfully processed {len(results_df):,} flows from `{uploaded_file.name}`.")
                except Exception as e:
                    st.error(f"Inference error: {e}")

        elif ingest_mode == "Upload Raw PCAP Capture":
            uploaded_pcap = st.file_uploader("Upload PCAP File (.pcap)", type=["pcap"], key="infer_pcap")
            max_pkts = st.number_input("Max Packets to Read", min_value=100, max_value=50000, value=5000, step=500)
            if uploaded_pcap is not None:
                with st.spinner("Dissecting PCAP packets into bidirectional flows..."):
                    try:
                        pcap_bytes = uploaded_pcap.read()
                        results_df, flows_df = pipeline.process_pcap(pcap_bytes, max_packets=int(max_pkts), threshold=threshold)
                        if results_df.empty:
                            st.warning("No IP/TCP/UDP sessions detected in PCAP.")
                        else:
                            st.success(f"Dissected {len(results_df)} flows from PCAP.")
                    except Exception as e:
                        st.error(f"PCAP Extraction Error: {e}")

        # Display Inference Output
        if results_df is not None and not results_df.empty:
            st.session_state["active_results"] = results_df

            # Telemetry Summary Cards
            tot = len(results_df)
            alerts = int((results_df["is_alert"] == 1).sum())
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Total Flows", f"{tot:,}")
            c_b.metric("Flagged Alerts", f"{alerts:,}")
            c_c.metric("Alert Rate", f"{alerts/tot*100:.2f}%")

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
                        explanation = pipeline.explain_flow_by_index(active_df, index=selected_idx, top_k=5)
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
