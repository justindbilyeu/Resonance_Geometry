#!/usr/bin/env python3
"""
Phase map sweep for hallucination regime boundaries.

Reads v2 config, sweeps (η, λ) grid, outputs:
- experiments/hallucination/results/phase_map.csv
- docs/papers/neurips/figures/Geometric Theory of AI Hallucination/phase_diagram.png
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Add src to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from resonance_geometry.hallucination.phase_dynamics import (
    simulate_trajectory,
    classify_regime,
)

import matplotlib.pyplot as plt


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_phase_sweep(config):
    """
    Sweep (η, λ) grid and classify regimes.

    Returns:
        results: List of dicts with (eta, lambda, regime, lambda_max, mi_bar, norm)
    """
    etas = config['grid']['eta']
    lambdas = config['grid']['lambda']

    # Build base params from config
    params_base = {
        'gamma': config['gamma'],
        'alpha': config['alpha'],
        'beta': config['beta'],
        'k': 1.0,  # grounding stiffness
        'skew': config['kappa'],
        'mu': 0.0,
        'mi_window': config['window'],
        'mi_ema': config['ema_alpha'],
        'omega_anchor': np.zeros(3),
        'use_adaptive_gain': config.get('use_adaptive_gain', False),
    }

    results = []
    for eta in etas:
        for lam in lambdas:
            params = params_base.copy()
            params['eta'] = eta
            params['lambda'] = lam

            # Run simulation
            traj = simulate_trajectory(
                params,
                T=config['t_horizon'],
                dt=config['dt'],
                seed=config['seed'],
            )

            regime = classify_regime(traj)
            regime_name = ['grounded', 'creative', 'hallucinatory'][regime]

            results.append({
                'eta': eta,
                'lambda': lam,
                'regime': regime,
                'regime_name': regime_name,
                'lambda_max': traj['lambda_max'][-1] if len(traj['lambda_max']) else 0.0,
                'mi_bar': traj['MI_bar'][-1] if len(traj['MI_bar']) else 0.0,
                'norm': traj['norm'][-1] if len(traj['norm']) else 0.0,
            })

            print(f"η={eta:.2f}, λ={lam:.2f} → {regime_name} (λ_max={results[-1]['lambda_max']:.3f}, Ī={results[-1]['mi_bar']:.3f})")

    return results, etas, lambdas


def save_csv(results, output_path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['eta', 'lambda', 'regime', 'regime_name', 'lambda_max', 'mi_bar', 'norm'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {output_path}")


def compute_boundary_fit(results):
    """
    Compute linear fit for phase boundary: η·Ī ≈ λ + γ

    Find points near creative regime (|λ_max| < 0.1) and fit η·Ī vs λ.
    """
    boundary_points = [r for r in results if abs(r['lambda_max']) < 0.1]

    if len(boundary_points) < 3:
        print("Warning: Too few boundary points for fit")
        return None, None, []

    # Extract η·Ī and λ
    eta_i = np.array([r['eta'] * r['mi_bar'] for r in boundary_points])
    lam = np.array([r['lambda'] for r in boundary_points])

    # Linear fit: η·Ī = m·λ + b
    coeffs = np.polyfit(lam, eta_i, deg=1)
    m, b = coeffs

    # R²
    eta_i_pred = m * lam + b
    ss_res = np.sum((eta_i - eta_i_pred)**2)
    ss_tot = np.sum((eta_i - np.mean(eta_i))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"\nPhase Boundary Fit: η·Ī ≈ {m:.3f}·λ + {b:.3f}")
    print(f"R² = {r_squared:.3f}")

    return m, b, boundary_points


def plot_phase_diagram(results, etas, lambdas, config, output_path, boundary_fit=None):
    """Generate phase diagram plot."""
    # Build 2D grid
    phase_grid = np.zeros((len(lambdas), len(etas)), dtype=int)
    for r in results:
        i = lambdas.index(r['lambda'])
        j = etas.index(r['eta'])
        phase_grid[i, j] = r['regime']

    # Plot
    plt.figure(figsize=(10, 7))
    extent = [min(etas), max(etas), min(lambdas), max(lambdas)]
    plt.imshow(phase_grid, origin='lower', extent=extent, cmap='RdYlGn_r',
               vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Regime', ticks=[0, 1, 2],
                 format=plt.FuncFormatter(lambda x, p: ['Grounded', 'Creative', 'Hallucinatory'][int(x)]))

    # Overlay boundary fit if available
    if boundary_fit:
        m, b = boundary_fit
        gamma = config['gamma']
        # From η·Ī ≈ m·λ + b, assuming typical Ī~1, plot guideline
        eta_line = np.linspace(min(etas), max(etas), 100)
        # Guideline: λ ≈ η - γ (assuming Ī≈1 for visualization)
        lam_guideline = eta_line - gamma
        plt.plot(eta_line, lam_guideline, 'w--', lw=2, alpha=0.7,
                label=f'Guideline: λ ≈ η - γ (Ī≈1)')

    plt.xlabel('η (Resonance Gain)', fontsize=12)
    plt.ylabel('λ (Grounding Strength)', fontsize=12)
    plt.title('Hallucination Phase Diagram (v2 with Adaptive Gain)', fontsize=14)
    plt.grid(alpha=0.3, color='white', linestyle='--')
    plt.legend(loc='upper left')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved phase diagram to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run phase map sweep')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Adaptive gain: {config.get('use_adaptive_gain', False)}")

    # Run sweep
    print("\nRunning phase sweep...")
    results, etas, lambdas = run_phase_sweep(config)

    # Save CSV
    csv_path = Path('experiments/hallucination/results/phase_map.csv')
    save_csv(results, csv_path)

    # Compute boundary fit
    m, b, boundary_points = compute_boundary_fit(results)
    boundary_fit = (m, b) if m is not None else None

    # Plot
    fig_path = Path('docs/papers/neurips/figures/Geometric Theory of AI Hallucination/phase_diagram.png')
    plot_phase_diagram(results, etas, lambdas, config, fig_path, boundary_fit)

    print("\nPhase map complete!")


if __name__ == '__main__':
    main()
