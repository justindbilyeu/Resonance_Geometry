#!/usr/bin/env python3
"""Generate simple figures illustrating hysteresis-style sweeps.

The original project produces a rich set of plots from expensive simulations.
For bootstrapping the documentation pipeline we instead generate deterministic
synthetic data that exercises the plotting code paths and yields stable figures
for CI.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from rg.sims.meta_flow_min_pair_v2 import batch_simulate


@dataclass
class SweepResult:
    eta: float
    trajectory: np.ndarray
    regime: int


def make_parameter_grid(lam: float, eta_values: Iterable[float], base: dict) -> List[dict]:
    grid = []
    for eta in eta_values:
        params = base.copy()
        params.update({'lambda': lam, 'eta': eta})
        grid.append(params)
    return grid


def run_sweep(lam: float, eta_values: np.ndarray, base: dict) -> List[SweepResult]:
    grid = make_parameter_grid(lam, eta_values, base)
    results: List[SweepResult] = []
    for eta, traj in zip(eta_values, batch_simulate(grid)):
        lambda_max = np.asarray(traj.lambda_max)
        results.append(SweepResult(eta=float(eta), trajectory=lambda_max, regime=int(traj.regime)))
    return results


def plot_results(results: List[SweepResult], lam: float, fig_dir: pathlib.Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    etas = np.array([r.eta for r in results])
    responses = np.array([r.trajectory[-1] for r in results])
    regimes = np.array([r.regime for r in results])

    fig, ax = plt.subplots(figsize=(6, 4))
    colours = np.array(['tab:blue', 'tab:orange', 'tab:red'])
    ax.scatter(etas, responses, c=colours[regimes], s=18)
    ax.set_xlabel(r"Feedback strength $\eta$")
    ax.set_ylabel(r"Re($\lambda_{\max}$) at $t_{\mathrm{final}}$")
    ax.set_title(rf"Synthetic sweep at $\lambda={lam:.2f}$")
    ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    scatter_path = fig_dir / 'hysteresis_scatter.png'
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(etas, responses, color='black')
    ax.fill_between(etas, 0.0, responses, color='tab:purple', alpha=0.2)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Response")
    ax.set_title("Synthetic hysteresis envelope")
    fig.tight_layout()
    ribbon_path = fig_dir / 'hysteresis_ribbon.png'
    fig.savefig(ribbon_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--eta_min', type=float, default=0.2)
    ap.add_argument('--eta_max', type=float, default=5.0)
    ap.add_argument('--eta_steps', type=int, default=41)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--beta', type=float, default=0.02)
    ap.add_argument('--skew', type=float, default=0.12)
    ap.add_argument('--mi_window', type=int, default=30)
    ap.add_argument('--mi_ema', type=float, default=0.1)
    ap.add_argument('--output', type=pathlib.Path, default=pathlib.Path('docs/papers/neurips/figures'))
    args = ap.parse_args()

    eta_values = np.linspace(args.eta_min, args.eta_max, args.eta_steps)
    base = dict(
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        skew=args.skew,
        mi_window=args.mi_window,
        mi_ema=args.mi_ema,
        k=1.0,
        omega_anchor=np.zeros(3),
    )

    results = run_sweep(args.lam, eta_values, base)
    plot_results(results, args.lam, args.output)


if __name__ == '__main__':
    main()
