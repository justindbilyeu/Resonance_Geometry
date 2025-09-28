"""Smoke tests for the topological forcing CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import pytest


@pytest.fixture()
def fake_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    outputs_dir = tmp_path / "outputs" / "topo"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = outputs_dir / "coverage.npz"

    cells = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.3, 0.3],
            [0.4, 0.2, 0.1],
        ]
    )
    np.savez(coverage_path, candidate_cells=cells, grid_spacing=0.5)

    monkeypatch.chdir(tmp_path)
    return coverage_path


def test_force_probe_dry_run(fake_coverage: Path) -> None:
    output_path = Path("outputs/topo/forbidden.json")
    repo_root = Path(__file__).resolve().parents[1]

    command = [
        "python",
        str(repo_root / "experiments" / "topo_constraint_cli.py"),
        "force-probe",
        "--dry-run",
        "--coverage",
        str(fake_coverage),
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert completed.returncode == 0

    data = json.loads(output_path.read_text())
    assert set(data) == {"cells", "summary"}

    summary = data["summary"]
    for key in ["total_cells", "counts", "time_budgets", "strategies"]:
        assert key in summary

    expected_labels = {"REACHABLE", "RARE_BUT_REACHABLE", "FORBIDDEN"}
    assert expected_labels.issubset(summary["counts"].keys())

    cells = data["cells"]
    assert cells, "At least one cell should be processed."
    for entry in cells:
        assert entry["label"] in expected_labels
        assert isinstance(entry.get("strategy_attempts"), list)
        assert entry["strategy_attempts"], "Each cell should record attempts."

        attempt = entry["strategy_attempts"][0]
        for key in ["strategy", "time_multiplier", "distance", "hit"]:
            assert key in attempt
