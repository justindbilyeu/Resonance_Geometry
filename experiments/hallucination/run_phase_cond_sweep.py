#!/usr/bin/env python3
"""
Phase conditioning sweep: vary η, λ and conditioning to observe boundary shifts.
Outputs: results/phase_cond/phase_cond_sweep.csv
"""
import sys
import os
import csv
import numpy as np
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from resonance_geometry.hallucination.phase_dynamics import simulate_trajectory, classify_regime
from resonance_geometry.hallucination.adaptive_gain import compute_effective_eta


def run_sweep():
    """Run parameter sweep with conditioning injection."""
    etas = np.linspace(0.2, 5.0, 10)  # Reduced for CI speed
    lambdas = np.linspace(0.1, 5.0, 8)

    results = []

    for eta in etas:
        for lam in lambdas:
            params = {
                'eta': eta,
                'lambda': lam,
                'gamma': 0.5,
                'k': 1.0,
                'alpha': 0.6,
                'beta': 0.02,
                'skew': 0.12,
                'mu': 0.0,
                'mi_window': 30,
                'mi_ema': 0.1,
                'omega_anchor': np.zeros(3),
                'adaptive_eta': {'enabled': True, 'epsilon': 1e-12}
            }

            traj = simulate_trajectory(params, T=5.0, dt=0.01)

            # Compute average kappa and eta_eff
            kappa_avg = float(np.mean(traj.get('kappa', [1.0])))
            eta_eff_avg = float(np.mean(traj.get('eta_eff', [eta])))
            i_bar = traj['MI_bar'][-1] if len(traj['MI_bar']) else 0.0
            norm_final = traj['norm'][-1] if len(traj['norm']) else 0.0
            lam_max = traj['lambda_max'][-1] if len(traj['lambda_max']) else 0.0
            regime = classify_regime(traj)

            results.append({
                'eta': eta,
                'lam': lam,
                'kappa_avg': kappa_avg,
                'eta_eff': eta_eff_avg,
                'i_bar': i_bar,
                'norm_final': norm_final,
                'lam_max': lam_max,
                'regime': regime
            })

            print(f"η={eta:.2f}, λ={lam:.2f}, κ_avg={kappa_avg:.2f}, regime={regime}")

    # Save CSV
    os.makedirs('results/phase_cond', exist_ok=True)
    with open('results/phase_cond/phase_cond_sweep.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['eta', 'lam', 'kappa_avg', 'eta_eff', 'i_bar', 'norm_final', 'lam_max', 'regime'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to results/phase_cond/phase_cond_sweep.csv")


if __name__ == '__main__':
    run_sweep()
