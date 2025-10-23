#!/usr/bin/env python3
"""Lightweight CI smoke checks for committed paper artifacts."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.resonance_geometry.hallucination.phase_dynamics import simulate_trajectory


ROOT = REPO_ROOT
SUMMARY_PATH = ROOT / "docs/analysis/eigs_scan_summary.json"
CSV_PATH = ROOT / "docs/analysis/eigs_scan_alpha.csv"
TRACES_DIR = ROOT / "results/phase/traces"


def _check_artifact_paths() -> None:
    for path in (SUMMARY_PATH, CSV_PATH):
        if not path.exists():
            raise SystemExit(f"Missing required artifact: {path}")
        if path.stat().st_size == 0:
            raise SystemExit(f"Artifact appears empty: {path}")

    trace_paths = sorted(TRACES_DIR.glob("traj_alpha_*.json"))
    if len(trace_paths) != 7:
        raise SystemExit("Expected seven trajectory JSON files")

    for trace in trace_paths:
        with trace.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        required = {"alpha", "t", "S1"}
        missing = required - payload.keys()
        if missing:
            raise SystemExit(f"Trace missing keys {missing} in {trace}")
        t = payload["t"]
        s1 = payload["S1"]
        if not isinstance(t, list) or not isinstance(s1, list):
            raise SystemExit(f"Trace lists malformed in {trace}")
        if not t or len(t) != len(s1):
            raise SystemExit(f"Trace length mismatch in {trace}")
        if not all(math.isfinite(float(v)) for v in s1):
            raise SystemExit(f"Non-finite S1 values in {trace}")


def _check_summary_and_csv() -> None:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    if {"alpha_grid", "max_real"} - summary.keys():
        raise SystemExit("Summary JSON missing required keys")
    alpha_grid = summary["alpha_grid"]
    max_real = summary["max_real"]
    if len(alpha_grid) == 0 or len(alpha_grid) != len(max_real):
        raise SystemExit("Summary JSON arrays have inconsistent lengths")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise SystemExit("Eigenvalue CSV is empty")
    expected_cols = {"alpha", "phi_eq", "eig_real_max", "eig_imag_mean"}
    if reader.fieldnames is None or not expected_cols <= set(reader.fieldnames):
        raise SystemExit("Eigenvalue CSV missing expected columns")


def _check_simulator() -> None:
    params = {
        "lambda": 1.0,
        "gamma": 0.5,
        "k": 1.0,
        "beta": 0.02,
        "skew": 0.12,
        "alpha": 0.35,
        "omega_anchor": [0.0, 0.0, 0.0],
        "mi_window": 10,
        "mi_ema": 0.2,
        "eta": 2.0,
    }
    traj = simulate_trajectory(params, T=2.0, dt=0.01)
    if len(traj["t"]) == 0 or len(traj["norm"]) != len(traj["t"]):
        raise SystemExit("Smoke trajectory failed to produce samples")
    if not all(math.isfinite(float(v)) for v in traj["norm"]):
        raise SystemExit("Smoke trajectory contains non-finite values")


def main() -> None:
    _check_artifact_paths()
    _check_summary_and_csv()
    _check_simulator()
    print("ci-smoke checks passed (artifacts + integrator)")


if __name__ == "__main__":
    main()
