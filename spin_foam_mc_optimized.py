"""Optimized spin-foam Monte Carlo toy simulation.

This module implements a lightweight Monte Carlo driver that mimics a
spin-foam sampler.  The implementation is intentionally simple – the CI
smoke test only needs to make sure the numerical core executes without
errors and produces stable summary statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SpinFoamRun:
    """Container for per-run metrics produced by the sampler."""

    run: int
    mean_amplitude: float
    mean_energy: float
    acceptance: float


def _simulate_single_run(
    rng: np.random.Generator, steps: int, size: int
) -> SpinFoamRun:
    """Run a single Monte Carlo sweep over a synthetic spin-foam lattice.

    The synthetic model is inspired by simplified Ising-like dynamics.  We
    draw Gaussian noise to perturb each lattice site, estimate a local
    action and aggregate a few summary statistics.  The goal is to provide a
    deterministic workload that exercises vectorised numpy code without
    depending on any heavy physics libraries.
    """

    # Randomly initialise a lattice of spin amplitudes in [-1, 1].
    lattice = rng.uniform(-1.0, 1.0, size=(size, size))

    accepted = 0
    amp_accum = 0.0
    energy_accum = 0.0

    for _ in range(steps):
        proposal = lattice + 0.25 * rng.normal(size=lattice.shape)

        # Simple Metropolis acceptance rule with quadratic action.
        action_old = np.sum(lattice * lattice)
        action_new = np.sum(proposal * proposal)
        accept_prob = float(np.exp(min(0.0, action_old - action_new)))

        if rng.random() < accept_prob:
            lattice = proposal
            accepted += 1
            action_old = action_new

        amp_accum += float(np.mean(lattice))
        energy_accum += float(action_old / lattice.size)

    mean_amp = amp_accum / steps
    mean_energy = energy_accum / steps
    acceptance = accepted / steps

    return SpinFoamRun(
        run=0,  # placeholder; caller assigns run index
        mean_amplitude=mean_amp,
        mean_energy=mean_energy,
        acceptance=acceptance,
    )


def optimized_spin_foam_mc(
    *,
    steps: int = 50_000,
    size: int = 32,
    runs: int = 8,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run the synthetic spin-foam Monte Carlo sampler.

    Parameters
    ----------
    steps:
        Number of Metropolis updates per run.  The smoke test keeps this
        small so that the job completes quickly.
    size:
        Linear lattice dimension (the lattice has ``size**2`` sites).
    runs:
        Number of independent Monte Carlo runs to execute.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with aggregate statistics and per-run metrics.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")

    rng = np.random.default_rng(seed)

    runs_data: List[SpinFoamRun] = []
    for idx in range(runs):
        run_stats = _simulate_single_run(rng, steps, size)
        runs_data.append(
            SpinFoamRun(
                run=idx,
                mean_amplitude=run_stats.mean_amplitude,
                mean_energy=run_stats.mean_energy,
                acceptance=run_stats.acceptance,
            )
        )

    mean_amp = float(np.mean([r.mean_amplitude for r in runs_data]))
    mean_energy = float(np.mean([r.mean_energy for r in runs_data]))
    acceptance = float(np.mean([r.acceptance for r in runs_data]))

    summary = {
        "steps": steps,
        "size": size,
        "runs": runs,
        "mean_amplitude": mean_amp,
        "mean_energy": mean_energy,
        "acceptance": acceptance,
        "per_run": [r.__dict__ for r in runs_data],
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for manual experimentation."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Spin-foam MC toy model")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    summary = optimized_spin_foam_mc(
        steps=args.steps, size=args.size, runs=args.runs, seed=args.seed
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
