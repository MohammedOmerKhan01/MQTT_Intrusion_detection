"""
loaders.py - Memory-conscious dataset loader for Uniflow and Biflow features.

Supports streaming reads directly from ZIP archives to prevent unnecessary
multi-gigabyte disk extraction while adding provenance metadata.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np


def get_archive_path(archive_name: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve archive path relative to workspace root."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / archive_name


def load_flow_dataset(
    flow_type: str = "biflow",
    base_dir: Optional[Path] = None,
    include_scenario: bool = True,
    sample_frac: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load all CSV files from the specified flow archive (uniflow or biflow).

    Args:
        flow_type: 'uniflow' or 'biflow'
        base_dir: Workspace root directory
        include_scenario: If True, adds 'scenario' and 'attack_type' columns
        sample_frac: Optional fraction to sample (e.g., 0.1 for 10% sample)
        random_state: Random seed for sampling reproducibility

    Returns:
        pd.DataFrame containing concatenated records from all scenarios.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent

    archive_filename = f"{flow_type}_features.zip"
    archive_path = base_dir / archive_filename

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    dfs: List[pd.DataFrame] = []

    with zipfile.ZipFile(archive_path, 'r') as zf:
        csv_files = sorted([f for f in zf.namelist() if f.endswith('.csv')])
        for fname in csv_files:
            # Derive scenario name from filename, e.g., 'biflow_scan_A.csv' -> 'scan_A'
            basename = os.path.basename(fname).replace('.csv', '')
            prefix = f"{flow_type}_"
            scenario_name = basename[len(prefix):] if basename.startswith(prefix) else basename

            with zf.open(fname) as f:
                df_curr = pd.read_csv(f)

            if include_scenario:
                df_curr['scenario'] = scenario_name
                # Detailed attack type: 'normal' for benign, scenario name for attacks
                df_curr['attack_type'] = np.where(
                    df_curr['is_attack'] == 1,
                    scenario_name,
                    'normal'
                )

            if sample_frac is not None and 0.0 < sample_frac < 1.0:
                df_curr = df_curr.sample(frac=sample_frac, random_state=random_state)

            dfs.append(df_curr)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_single_scenario(
    flow_type: str,
    scenario: str,
    base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a single scenario CSV file from the flow archive.

    Args:
        flow_type: 'uniflow' or 'biflow'
        scenario: Scenario name, e.g., 'normal', 'scan_A', 'scan_sU', 'mqtt_bruteforce', 'sparta'
        base_dir: Workspace root directory

    Returns:
        pd.DataFrame for the requested scenario
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent

    archive_path = base_dir / f"{flow_type}_features.zip"
    target_filename = f"{flow_type}_{scenario}.csv"

    with zipfile.ZipFile(archive_path, 'r') as zf:
        matches = [f for f in zf.namelist() if os.path.basename(f) == target_filename]
        if not matches:
            available = [os.path.basename(f) for f in zf.namelist() if f.endswith('.csv')]
            raise KeyError(f"File {target_filename} not in archive. Available: {available}")
        
        with zf.open(matches[0]) as f:
            df = pd.read_csv(f)
            df['scenario'] = scenario
            df['attack_type'] = np.where(df['is_attack'] == 1, scenario, 'normal')
            return df
