"""Smoke tests for the optimized spin-foam Monte Carlo driver."""

from __future__ import annotations

import pytest

from spin_foam_mc_optimized import optimized_spin_foam_mc


@pytest.mark.parametrize(
    "steps,size,runs,seed",
    [
        (200, 8, 2, 7),
        (128, 4, 1, 0),
    ],
)
def test_spin_foam_summary_shape(steps: int, size: int, runs: int, seed: int) -> None:
    """The sampler should produce deterministic, well-formed summaries."""

    summary = optimized_spin_foam_mc(steps=steps, size=size, runs=runs, seed=seed)

    assert summary["steps"] == steps
    assert summary["size"] == size
    assert summary["runs"] == runs
    assert len(summary["per_run"]) == runs
    assert all(entry["run"] == idx for idx, entry in enumerate(summary["per_run"]))
    assert all("mean_amplitude" in entry for entry in summary["per_run"])
    assert all("mean_energy" in entry for entry in summary["per_run"])
    assert all("acceptance" in entry for entry in summary["per_run"])


@pytest.mark.parametrize(
    "seed,expected",
    [
        (7, {
            "mean_amplitude": 0.005615327393209304,
            "mean_energy": 0.3924015469987234,
            "acceptance": 0.1425,
        }),
        (0, {
            "mean_amplitude": 0.01270063215400305,
            "mean_energy": 0.4323862557531094,
            "acceptance": 0.1625,
        }),
    ],
)
def test_spin_foam_summary_values(seed: int, expected: dict[str, float]) -> None:
    """Validate a couple of deterministic summaries for regression coverage."""

    summary = optimized_spin_foam_mc(steps=200, size=8, runs=2, seed=seed)

    for key, value in expected.items():
        assert summary[key] == pytest.approx(value, rel=1e-6)
