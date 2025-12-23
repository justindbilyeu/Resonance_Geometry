"""
Reporting utilities for F_AI results.

Generates CSV and JSON outputs for episodes, sessions, and topologies.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .types import EpisodeMetrics, CalibrationData


def write_episode_metrics_csv(
    metrics: List[EpisodeMetrics], filepath: str
) -> None:
    """
    Write per-episode metrics to CSV.

    Args:
        metrics: List of EpisodeMetrics objects
        filepath: Output CSV path
    """
    rows = [m.to_dict() for m in metrics]
    df = pd.DataFrame(rows)

    # Column order
    columns = [
        "session_id",
        "episode_id",
        "topology_id",
        "F_AI",
        "C",
        "U",
        "D",
        "P",
        "N",
        "C_z",
        "U_z",
        "D_z",
        "P_z",
        "N_z",
        "n_messages",
        "n_tokens",
    ]

    # Reorder columns (if they exist)
    columns = [c for c in columns if c in df.columns]
    df = df[columns]

    # Sort by session and episode
    df = df.sort_values(["session_id", "episode_id"])

    # Write to CSV
    df.to_csv(filepath, index=False, float_format="%.6f")


def compute_session_summary(
    metrics: List[EpisodeMetrics],
) -> pd.DataFrame:
    """
    Compute per-session summary statistics.

    Args:
        metrics: List of EpisodeMetrics objects

    Returns:
        DataFrame with session-level statistics
    """
    rows = [m.to_dict() for m in metrics]
    df = pd.DataFrame(rows)

    # Group by session
    grouped = df.groupby(["session_id", "topology_id"])

    summary = grouped.agg(
        {
            "episode_id": "count",  # n_episodes
            "F_AI": ["mean", "median", "std"],
            "C": ["mean", "std"],
            "U": ["mean", "std"],
            "D": ["mean", "std"],
            "P": ["mean", "std"],
            "N": ["mean", "std"],
            "n_messages": "sum",
            "n_tokens": "sum",
        }
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    # Rename
    summary = summary.rename(
        columns={
            "episode_id_count": "n_episodes",
            "F_AI_mean": "F_AI_mean",
            "F_AI_median": "F_AI_median",
            "F_AI_std": "F_AI_std",
            "C_mean": "C_mean",
            "U_mean": "U_mean",
            "D_mean": "D_mean",
            "P_mean": "P_mean",
            "N_mean": "N_mean",
            "n_messages_sum": "total_messages",
            "n_tokens_sum": "total_tokens",
        }
    )

    summary = summary.reset_index()

    return summary


def write_session_summary_csv(
    metrics: List[EpisodeMetrics], filepath: str
) -> None:
    """
    Write per-session summary to CSV.

    Args:
        metrics: List of EpisodeMetrics objects
        filepath: Output CSV path
    """
    summary = compute_session_summary(metrics)
    summary.to_csv(filepath, index=False, float_format="%.6f")


def compute_topology_summary(
    metrics: List[EpisodeMetrics],
) -> pd.DataFrame:
    """
    Compute per-topology summary statistics.

    Args:
        metrics: List of EpisodeMetrics objects

    Returns:
        DataFrame with topology-level statistics
    """
    rows = [m.to_dict() for m in metrics]
    df = pd.DataFrame(rows)

    # Group by topology
    grouped = df.groupby("topology_id")

    summary = grouped.agg(
        {
            "episode_id": "count",
            "session_id": "nunique",  # n_sessions
            "F_AI": ["mean", "median", "std"],
            "C": ["mean", "std"],
            "U": ["mean", "std"],
            "D": ["mean", "std"],
            "P": ["mean", "std"],
            "N": ["mean", "std"],
        }
    )

    # Flatten
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    # Rename
    summary = summary.rename(
        columns={
            "episode_id_count": "n_episodes",
            "session_id_nunique": "n_sessions",
        }
    )

    summary = summary.reset_index()

    return summary


def write_topology_summary_csv(
    metrics: List[EpisodeMetrics], filepath: str
) -> None:
    """
    Write per-topology summary to CSV.

    Args:
        metrics: List of EpisodeMetrics objects
        filepath: Output CSV path
    """
    summary = compute_topology_summary(metrics)
    summary.to_csv(filepath, index=False, float_format="%.6f")


def write_normalization_json(
    calibration_by_topology: Dict[str, CalibrationData],
    filepath: str,
    calibration_method: str = "first_n",
    calibration_n: int = 50,
) -> None:
    """
    Write normalization parameters to JSON.

    Args:
        calibration_by_topology: Dict mapping topology_id to CalibrationData
        filepath: Output JSON path
        calibration_method: Calibration method used
        calibration_n: Number of episodes per topology (for first_n method)
    """
    output = {
        "calibration_method": calibration_method,
        "calibration_n": calibration_n,
        "per_topology": {},
    }

    for topology_id, calib in calibration_by_topology.items():
        output["per_topology"][topology_id] = calib.to_dict()

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


def write_all_reports(
    metrics: List[EpisodeMetrics],
    calibration_by_topology: Dict[str, CalibrationData],
    output_dir: str,
    calibration_method: str = "first_n",
    calibration_n: int = 50,
) -> None:
    """
    Write all standard reports.

    Args:
        metrics: List of EpisodeMetrics objects
        calibration_by_topology: Calibration data
        output_dir: Output directory
        calibration_method: Calibration method
        calibration_n: Calibration n
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Episode-level
    write_episode_metrics_csv(
        metrics, str(output_path / "episodes.csv")
    )

    # Session-level
    write_session_summary_csv(
        metrics, str(output_path / "sessions.csv")
    )

    # Topology-level
    write_topology_summary_csv(
        metrics, str(output_path / "topology_summary.csv")
    )

    # Normalization parameters
    write_normalization_json(
        calibration_by_topology,
        str(output_path / "normalization.json"),
        calibration_method=calibration_method,
        calibration_n=calibration_n,
    )

    print(f"Reports written to {output_dir}/")
    print(f"  - episodes.csv: {len(metrics)} episodes")
    print(f"  - sessions.csv")
    print(f"  - topology_summary.csv")
    print(f"  - normalization.json")
