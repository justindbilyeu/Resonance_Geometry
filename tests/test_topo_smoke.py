"""Smoke test for the topological accessibility CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_map_accessible_cli_creates_artifacts() -> None:
    coverage_path = Path("outputs/topo/coverage.npz")
    samples_path = Path("data/topo/samples.npz")

    for target in (coverage_path, samples_path):
        if target.exists():
            target.unlink()

    result = subprocess.run(
        ["python", "experiments/topo_constraint_cli.py", "map-accessible", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert coverage_path.exists()
    assert samples_path.exists()
