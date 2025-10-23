from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.resonance_geometry.hallucination.phase_dynamics import simulate_trajectory


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs/analysis/eigs_scan_summary.json"
CSV_PATH = ROOT / "docs/analysis/eigs_scan_alpha.csv"
TRACES_DIR = ROOT / "results/phase/traces"


@pytest.mark.parametrize("path", [SUMMARY_PATH, CSV_PATH])
def test_artifact_presence(path: Path) -> None:
    assert path.exists(), f"Expected artifact to exist: {path}"
    assert path.stat().st_size > 0, f"Artifact is empty: {path}"


def test_summary_schema() -> None:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    assert {"alpha_grid", "max_real"} <= summary.keys()
    alpha_grid = summary["alpha_grid"]
    max_real = summary["max_real"]
    assert isinstance(alpha_grid, list)
    assert isinstance(max_real, list)
    assert len(alpha_grid) > 0
    assert len(alpha_grid) == len(max_real)
    assert all(isinstance(a, (int, float)) for a in alpha_grid)
    assert any(v is not None for v in max_real)


def test_csv_has_expected_columns() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert rows, "Expected eigenvalue CSV to contain rows"
    assert reader.fieldnames is not None
    expected_cols = {"alpha", "phi_eq", "eig_real_max", "eig_imag_mean"}
    assert expected_cols <= set(reader.fieldnames)


@pytest.mark.parametrize("trace_path", sorted(TRACES_DIR.glob("traj_alpha_*.json")))
def test_trace_schema(trace_path: Path) -> None:
    with trace_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    required = {"alpha", "t", "S1"}
    assert required <= payload.keys()
    t = payload["t"]
    s1 = payload["S1"]
    assert isinstance(t, list)
    assert isinstance(s1, list)
    assert len(t) > 0
    assert len(t) == len(s1)
    assert all(math.isfinite(float(v)) for v in s1)
    assert math.isclose(float(payload["alpha"]), float(trace_path.stem.split("_")[-1]))


@pytest.mark.parametrize("ci_flag", [True, False])
def test_simulator_seed_and_ci(ci_flag: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_value = 4242
    if ci_flag:
        monkeypatch.setenv("RG_CI", "1")
        monkeypatch.setenv("RG_CI_MAX_STEPS", "64")
    else:
        monkeypatch.delenv("RG_CI", raising=False)
        monkeypatch.delenv("RG_CI_MAX_STEPS", raising=False)
    monkeypatch.setenv("RG_SEED", str(seed_value))

    params = {
        "lambda": 1.0,
        "gamma": 0.5,
        "k": 1.0,
        "beta": 0.02,
        "skew": 0.12,
        "alpha": 0.35,
        "omega_anchor": np.zeros(3),
        "mi_window": 12,
        "mi_ema": 0.2,
        "eta": 2.0,
    }
    traj = simulate_trajectory(params)
    assert len(traj["t"]) == len(traj["norm"])
    assert len(traj["t"]) > 0
    if ci_flag:
        assert len(traj["t"]) <= 64
    assert all(math.isfinite(float(v)) for v in traj["norm"])
