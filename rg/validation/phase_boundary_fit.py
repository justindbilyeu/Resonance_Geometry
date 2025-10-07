#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

from rg.sims.meta_flow_min_pair_v2 import simulate_trajectory


def hallucinatory_eta_for_lambda(lam: float, eta_grid: np.ndarray, base: Dict[str, float]) -> float:
    """Smallest eta with hallucinatory regime (or positive lambda_max)."""
    for eta in eta_grid:
        params = base.copy()
        params.update({'lambda': lam, 'eta': float(eta)})
        traj = simulate_trajectory(params, T_max=3.0, dt=0.01)
        regime = traj.get('regime', None)
        lam_arr = np.asarray(traj.get('lambda_max', [0.0]))
        lam_max = float(lam_arr[-1]) if lam_arr.size else float(lam_arr)
        if regime == 2 or lam_max > 0.0:
            return float(eta)
    return float('nan')


def linear_fit(lam_grid: np.ndarray, eta_c: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(eta_c)
    if not np.any(mask):
        return {
            'slope': float('nan'),
            'intercept': float('nan'),
            'R2': float('nan'),
            'I_hat': float('nan'),
        }
    lam_valid = lam_grid[mask]
    eta_valid = eta_c[mask]
    slope, intercept = np.polyfit(lam_valid, eta_valid, 1)
    predictions = slope * lam_valid + intercept
    residuals = eta_valid - predictions
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((eta_valid - np.mean(eta_valid)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    I_hat = float('inf') if slope == 0 else float(1.0 / slope)
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'R2': float(R2),
        'I_hat': float(I_hat),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--lam_min', type=float, default=0.1)
    ap.add_argument('--lam_max', type=float, default=5.0)
    ap.add_argument('--lam_steps', type=int, default=11)
    ap.add_argument('--eta_min', type=float, default=0.2)
    ap.add_argument('--eta_max', type=float, default=5.0)
    ap.add_argument('--eta_steps', type=int, default=101)
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
    args = ap.parse_args()

    base = {
        'gamma': args.gamma,
        'alpha': args.alpha,
        'beta': args.beta,
        'skew': args.skew,
        'mi_window': args.mi_window,
        'mi_ema': args.mi_ema,
        'k': 1.0,
        'omega_anchor': np.zeros(3),
        'eta': 1.0,
        'lambda': 1.0,
        'algebra': args.algebra,
        'antisym_coupling': args.antisym_coupling,
        'noise_std': args.noise_std,
        'seed': args.seed,
        'mi_est': args.mi_est,
        'mi_scale': args.mi_scale,
    }

    lam_grid = np.linspace(args.lam_min, args.lam_max, args.lam_steps)
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.eta_steps)

    eta_c = np.array([hallucinatory_eta_for_lambda(lam, eta_grid, base) for lam in lam_grid])

    fit = linear_fit(lam_grid, eta_c)
    slope = fit['slope']
    intercept = fit['intercept']
    R2 = fit['R2']
    I_hat = fit['I_hat']

    gamma = args.gamma
    b_diff = abs(intercept - gamma)
    if not np.isnan(intercept) and b_diff > 0.3:
        print(f"[warning] Intercept deviates from γ by {b_diff:.3f}")

    output_dir = Path('rg/results/sage_corrected')
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'phase_boundary_fit.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['lambda', 'eta_critical'])
        for lam_val, eta_val in zip(lam_grid, eta_c):
            writer.writerow([f"{lam_val:.6f}", f"{eta_val:.6f}"])

    flags = {
        'algebra': args.algebra,
        'antisym_coupling': bool(args.antisym_coupling),
        'noise_std': float(args.noise_std),
        'mi_est': args.mi_est,
        'mi_scale': float(args.mi_scale),
        'seed': int(args.seed),
    }

    json_path = output_dir / 'phase_boundary_fit.json'
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(
            {
                'slope': slope,
                'intercept': intercept,
                'R2': R2,
                'I_hat': I_hat,
                'gamma': gamma,
                'flags': flags,
            },
            handle,
            indent=2,
        )

    print(f"Linear fit: eta_c ≈ {slope:.3f} * lambda + {intercept:.3f} (R²={R2:.3f})")
    print(f"I_hat ≈ {I_hat:.3f}, gamma={gamma:.3f}")


if __name__ == '__main__':
    main()
