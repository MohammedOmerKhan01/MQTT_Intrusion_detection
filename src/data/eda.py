"""
eda.py - Comprehensive Exploratory Data Analysis & Quality Audit Pipeline

Performs:
1. Statistical profiling across Benign vs. Attack flows
2. Null, infinite, and out-of-bounds validity checks
3. Exact duplicate and 5-tuple consistency audit
4. Port distribution and IP leakage vulnerability assessment
5. Publication-ready visualization generation
6. Machine-readable summary export to artifacts/reports/eda_summary.json
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for automation
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure workspace root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.loaders import load_flow_dataset
from src.data.schema import get_uniflow_columns, get_biflow_columns


def run_data_quality_audit(df: pd.DataFrame, flow_type: str) -> Dict[str, Any]:
    """Execute rigorous data quality and anomaly audit."""
    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_dict = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    
    num_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(num_df).sum()
    inf_dict = {col: int(cnt) for col, cnt in inf_counts.items() if cnt > 0}

    # Duplicate audit
    cols_for_dup = [c for c in df.columns if c not in ['scenario', 'attack_type']]
    exact_duplicates = int(df.duplicated(subset=cols_for_dup, keep=False).sum())
    
    # 5-tuple consistency audit: Check if any 5-tuple has conflicting labels
    tuple_cols = ['ip_src', 'ip_dst', 'prt_src', 'prt_dst', 'proto']
    tuple_conflicts = int((df.groupby(tuple_cols)['is_attack'].nunique() > 1).sum())

    # Range & Plausibility checks
    invalid_ports = int(((df['prt_src'] < 0) | (df['prt_src'] > 65535) | (df['prt_dst'] < 0) | (df['prt_dst'] > 65535)).sum())
    invalid_proto = int((~df['proto'].isin([6, 17])).sum())
    
    # Check non-negative packet counts and IATs
    neg_checks = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'is_attack':
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                neg_checks[col] = neg_count

    return {
        "flow_type": flow_type,
        "total_records": total_rows,
        "null_counts": null_dict,
        "infinite_counts": inf_dict,
        "exact_duplicate_rows": exact_duplicates,
        "conflicting_5tuples": tuple_conflicts,
        "out_of_bound_ports": invalid_ports,
        "unrecognized_protocols": invalid_proto,
        "negative_measurement_counts": neg_checks
    }


def analyze_leakage_and_identifiers(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze whether IP addresses, ports, or protocols introduce predictive leakage."""
    normal_df = df[df['is_attack'] == 0]
    attack_df = df[df['is_attack'] == 1]

    norm_src_ips = set(normal_df['ip_src'].unique())
    atk_src_ips = set(attack_df['ip_src'].unique())
    overlapping_src_ips = list(norm_src_ips.intersection(atk_src_ips))

    norm_dst_ports = normal_df['prt_dst'].value_counts().head(5).to_dict()
    atk_dst_ports = attack_df['prt_dst'].value_counts().head(5).to_dict()

    return {
        "unique_src_ips_normal": len(norm_src_ips),
        "unique_src_ips_attack": len(atk_src_ips),
        "overlapping_src_ips_count": len(overlapping_src_ips),
        "overlapping_src_ips": overlapping_src_ips,
        "top_normal_dst_ports": {str(k): int(v) for k, v in norm_dst_ports.items()},
        "top_attack_dst_ports": {str(k): int(v) for k, v in atk_dst_ports.items()},
        "ip_overlap_ratio": round(len(overlapping_src_ips) / len(norm_src_ips.union(atk_src_ips)), 4)
    }


def generate_eda_visualizations(biflow_df: pd.DataFrame, output_dir: Path):
    """Generate high-contrast, publication-quality EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class Distribution by Scenario
    plt.figure(figsize=(10, 5))
    sc_counts = pd.crosstab(biflow_df['scenario'], biflow_df['is_attack'])
    sc_counts.columns = ['Benign (0)', 'Attack (1)']
    ax = sc_counts.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], figsize=(10, 5.5))
    plt.title('Biflow Class Composition Across Attack Scenarios', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Scenario File', fontsize=11, labelpad=8)
    plt.ylabel('Flow Count', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Ground Truth Label', frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_balance.png', dpi=300)
    plt.close()

    # 2. Key Numerical Feature Distributions (Boxplots on Log Scale)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    feature_pairs = [
        ('fwd_num_pkts', 'Forward Packet Count', axes[0, 0]),
        ('bwd_num_pkts', 'Backward Packet Count', axes[0, 1]),
        ('fwd_mean_iat', 'Forward Mean IAT (sec)', axes[1, 0]),
        ('fwd_mean_pkt_len', 'Forward Mean Packet Length (bytes)', axes[1, 1])
    ]

    for col, title, ax in feature_pairs:
        sns.boxplot(
            x='is_attack',
            y=col,
            hue='is_attack',
            data=biflow_df,
            ax=ax,
            palette=['#337ab7', '#d9534f'],
            showfliers=False,
            legend=False
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Benign (0)', 'Attack (1)'])
        ax.set_xlabel('')
        if col in ['fwd_num_pkts', 'bwd_num_pkts', 'fwd_mean_iat']:
            ax.set_yscale('log')
            ax.set_ylabel('Log Scale Value')
    
    plt.suptitle('Comparison of Key Flow Features: Benign vs. Attack (Outliers Hidden)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300)
    plt.close()

    # 3. TCP Flags Analysis
    flag_cols = ['fwd_num_psh_flags', 'bwd_num_psh_flags', 'fwd_num_rst_flags', 'bwd_num_rst_flags']
    flag_means = biflow_df.groupby('is_attack')[flag_cols].mean().T
    flag_means.columns = ['Benign (0)', 'Attack (1)']
    
    plt.figure(figsize=(9, 5))
    flag_means.plot(kind='bar', color=['#2b5c8f', '#e67e22'], figsize=(9, 5))
    plt.title('Mean TCP Flag Counts per Flow: Benign vs. Attack', fontsize=13, fontweight='bold')
    plt.ylabel('Average Flags per Flow', fontsize=11)
    plt.xticks(
        ticks=range(len(flag_cols)),
        labels=['Fwd PSH', 'Bwd PSH', 'Fwd RST', 'Bwd RST'],
        rotation=0,
        fontsize=10
    )
    plt.yscale('log')
    plt.legend(title='Traffic Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'tcp_flags_comparison.png', dpi=300)
    plt.close()

    # 4. Correlation Heatmap of Flow Features
    plt.figure(figsize=(11, 9))
    numeric_features = [
        'fwd_num_pkts', 'bwd_num_pkts', 'fwd_mean_iat', 'bwd_mean_iat',
        'fwd_std_iat', 'fwd_mean_pkt_len', 'bwd_mean_pkt_len',
        'fwd_num_bytes', 'bwd_num_bytes', 'fwd_num_psh_flags', 'bwd_num_rst_flags'
    ]
    corr_matrix = biflow_df[numeric_features].corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Spearman Correlation'}
    )
    plt.title('Spearman Feature Correlation Matrix (Biflow Features)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
    plt.close()

    # 5. Identifier & Port Leakage Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Source IP flow counts by class
    ip_class = pd.crosstab(biflow_df['ip_src'], biflow_df['is_attack'])
    ip_class.columns = ['Benign', 'Attack']
    ip_class = ip_class.sort_values(by=['Attack', 'Benign'], ascending=False).head(12)
    ip_class.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=ax1)
    ax1.set_title('Top Source IP Addresses by Traffic Class (Severe Leakage Risk)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Source IP Address')
    ax1.set_ylabel('Flow Count')
    ax1.tick_params(axis='x', rotation=45)

    # Top Destination Ports
    port_class = pd.crosstab(biflow_df['prt_dst'], biflow_df['is_attack'])
    port_class.columns = ['Benign', 'Attack']
    port_class = port_class.sort_values(by=['Attack', 'Benign'], ascending=False).head(10)
    port_class.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=ax2)
    ax2.set_title('Top Destination Ports by Traffic Class', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Destination Port')
    ax2.set_ylabel('Flow Count')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle('Network Identifier Audit: Evaluating Predictive Data Leakage', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'leakage_analysis.png', dpi=300)
    plt.close()


def run_full_eda(base_dir: Path) -> Dict[str, Any]:
    """Execute complete Stage 2 EDA workflow."""
    print("=" * 70)
    print("STARTING STAGE 2: EXPLORATORY DATA ANALYSIS & DATA QUALITY AUDIT")
    print("=" * 70)

    # 1. Load Biflow Data
    print("\n[1/4] Loading Biflow Dataset...")
    biflow_df = load_flow_dataset(flow_type="biflow", base_dir=base_dir)
    print(f"Loaded {len(biflow_df):,} biflow records across {biflow_df['scenario'].nunique()} scenarios.")

    # 2. Load Uniflow Data
    print("[2/4] Loading Uniflow Dataset for Quality Verification...")
    uniflow_df = load_flow_dataset(flow_type="uniflow", base_dir=base_dir)
    print(f"Loaded {len(uniflow_df):,} uniflow records.")

    # 3. Run Quality Audits
    print("[3/4] Running Quality Audits & Leakage Assessment...")
    biflow_audit = run_data_quality_audit(biflow_df, "biflow")
    uniflow_audit = run_data_quality_audit(uniflow_df, "uniflow")
    leakage_audit = analyze_leakage_and_identifiers(biflow_df)

    # 4. Generate Visualizations
    figures_dir = base_dir / "artifacts" / "figures"
    print(f"[4/4] Generating Publication Figures in: {figures_dir} ...")
    generate_eda_visualizations(biflow_df, figures_dir)

    # Export EDA Summary Report
    reports_dir = base_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "eda_summary.json"

    eda_summary = {
        "biflow_quality": biflow_audit,
        "uniflow_quality": uniflow_audit,
        "leakage_assessment": leakage_audit,
        "figures_generated": [
            "class_balance.png",
            "feature_distributions.png",
            "tcp_flags_comparison.png",
            "correlation_heatmap.png",
            "leakage_analysis.png"
        ]
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(eda_summary, f, indent=2)

    print(f"\nEDA Summary successfully written to: {summary_path}")
    print("=" * 70)
    print("STAGE 2 QUALITY AUDIT SUMMARY")
    print("=" * 70)
    print(f"Biflow Records:          {biflow_audit['total_records']:,}")
    print(f"Null Values:             {sum(biflow_audit['null_counts'].values())}")
    print(f"Infinite Values:         {sum(biflow_audit['infinite_counts'].values())}")
    print(f"Exact Duplicate Rows:    {biflow_audit['exact_duplicate_rows']} ({round(biflow_audit['exact_duplicate_rows']/biflow_audit['total_records']*100, 3)}%)")
    print(f"Conflicting 5-tuples:    {biflow_audit['conflicting_5tuples']}")
    print(f"Out-of-bounds Ports:     {biflow_audit['out_of_bound_ports']}")
    print(f"Overlapping Host IPs:    {leakage_audit['overlapping_src_ips_count']} of {leakage_audit['unique_src_ips_normal'] + leakage_audit['unique_src_ips_attack']} unique IPs")
    print("=" * 70)

    return eda_summary


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent.parent
    run_full_eda(workspace_root)
