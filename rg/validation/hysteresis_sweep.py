#!/usr/bin/env python3
"""Generate simple figures illustrating hysteresis-style sweeps."""
from __future__ import annotations

import argparse
import json
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
    lambda_final: float
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
        lambda_final = float(lambda_max[-1]) if lambda_max.size else 0.0
        results.append(
            SweepResult(
                eta=float(eta),
                trajectory=lambda_max,
                lambda_final=lambda_final,
                regime=int(traj.regime),
            )
        )
    return results


def _critical_eta(results: List[SweepResult]) -> float:
    for result in results:
        if result.regime == 2 or result.lambda_final > 0.0:
            return float(result.eta)
    return float('nan')


def compute_report(
    forward: List[SweepResult],
    backward: List[SweepResult],
    lam: float,
    flags: dict,
) -> dict:
    etas = np.array([r.eta for r in forward])
    forward_vals = np.array([r.lambda_final for r in forward])
    backward_vals = np.array([r.lambda_final for r in backward[::-1]])  # align with ascending eta

    max_gap = float(np.max(np.abs(forward_vals - backward_vals)))
    loop_area = float(np.trapezoid(forward_vals - backward_vals, etas))
    eta_crit_forward = _critical_eta(forward)
    eta_crit_backward = _critical_eta(list(reversed(backward)))

    report = {
        'lambda': float(lam),
        'max_gap': max_gap,
        'loop_area': loop_area,
        'eta_crit_forward': eta_crit_forward,
        'eta_crit_backward': eta_crit_backward,
        'flags': flags,
    }
    return report


def plot_results(
    forward: List[SweepResult],
    backward: List[SweepResult],
    lam: float,
    fig_dir: pathlib.Path,
    label_suffix: str,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    etas = np.array([r.eta for r in forward])
    forward_resp = np.array([r.lambda_final for r in forward])
    backward_resp = np.array([r.lambda_final for r in backward[::-1]])
    regimes = np.array([r.regime for r in forward])

    fig, ax = plt.subplots(figsize=(6, 4))
    colours = np.array(['tab:blue', 'tab:orange', 'tab:red'])
    ax.scatter(etas, forward_resp, c=colours[regimes], s=18, label=f'forward ({label_suffix})')
    ax.scatter(etas, backward_resp, marker='x', color='black', s=16, label='backward')
    ax.set_xlabel(r"Feedback strength $\eta$")
    ax.set_ylabel(r"Re($\lambda_{\max}$) at $t_{\mathrm{final}}$")
    ax.set_title(rf"Synthetic sweep at $\lambda={lam:.2f}$")
    ax.grid(True, which='both', alpha=0.2)
    ax.legend()
    fig.tight_layout()
    scatter_path = fig_dir / 'hysteresis_scatter.png'
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(etas, forward_resp, color='tab:purple', label=f'forward ({label_suffix})')
    ax.plot(etas, backward_resp, color='tab:green', linestyle='--', label='backward')
    ax.fill_between(etas, backward_resp, forward_resp, color='tab:purple', alpha=0.15)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Response")
    ax.set_title("Synthetic hysteresis envelope")
    ax.legend()
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
    ap.add_argument('--algebra', choices=['su2', 'so3'], default='su2')
    ap.add_argument('--antisym_coupling', action='store_true', default=False)
    ap.add_argument('--noise_std', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mi_est', choices=['corr', 'svd'], default='corr')
    ap.add_argument('--mi_scale', type=float, default=1.0)
    ap.add_argument('--output', type=pathlib.Path, default=pathlib.Path('papers/neurips/figures'))
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
        algebra=args.algebra,
        antisym_coupling=args.antisym_coupling,
        noise_std=args.noise_std,
        seed=args.seed,
        mi_est=args.mi_est,
        mi_scale=args.mi_scale,
    )

    forward = run_sweep(args.lam, eta_values, base)
    backward = run_sweep(args.lam, eta_values[::-1], base)

    label_suffix = f"{args.algebra}, noise={args.noise_std:.3f}, MI={args.mi_est}"
    plot_results(forward, backward, args.lam, args.output, label_suffix)

    flags = {
        'algebra': args.algebra,
        'antisym_coupling': bool(args.antisym_coupling),
        'noise_std': float(args.noise_std),
        'mi_est': args.mi_est,
        'mi_scale': float(args.mi_scale),
        'seed': int(args.seed),
    }
    report = compute_report(forward, backward, args.lam, flags)

    output_dir = pathlib.Path('rg/results/sage_corrected')
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'hysteresis_report.json'
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
