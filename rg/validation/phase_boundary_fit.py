#!/usr/bin/env python3
import argparse
import numpy as np

from rg.sims.meta_flow_min_pair_v2 import simulate_trajectory

def hallucinatory_eta_for_lambda(lam, eta_grid, base):
    """Smallest eta with hallucinatory regime (or positive lambda_max)."""
    for eta in eta_grid:
        params = base.copy()
        params.update({'lambda': lam, 'eta': eta})
        traj = simulate_trajectory(params, T_max=3.0, dt=0.01)
        regime = traj.get('regime', None)
        lam_arr = traj.get('lambda_max', [0.0])
        lam_max = float(lam_arr[-1] if hasattr(lam_arr, '__len__') and len(lam_arr) > 0 else lam_arr)
        if regime == 2 or lam_max > 0.0:
            return eta
    return np.nan


def main():
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
    }

    lam_grid = np.linspace(args.lam_min, args.lam_max, args.lam_steps)
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.eta_steps)

    for lam in lam_grid:
        eta_c = hallucinatory_eta_for_lambda(lam, eta_grid, base)
        print(f"λ={lam:.2f} -> η_c≈{eta_c:.3f}")


if __name__ == '__main__':
    main()
