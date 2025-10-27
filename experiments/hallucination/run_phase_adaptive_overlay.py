#!/usr/bin/env python3
"""
Generate overlay figure comparing base and adaptive phase boundaries.
Outputs: docs/papers/neurips/figures/Geometric Theory of AI Hallucination/phase_adaptive_overlay.svg
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_overlay():
    """Create overlay visualization."""
    # Load sweep data
    csv_path = Path('results/phase_cond/phase_cond_sweep.csv')
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run phase-cond-sweep first.")
        return

    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Extract data
    etas = sorted(set(float(r['eta']) for r in data))
    lams = sorted(set(float(r['lam']) for r in data))

    # Create grid
    kappa_grid = np.zeros((len(lams), len(etas)))
    regime_grid = np.zeros((len(lams), len(etas)))

    for row in data:
        i = lams.index(float(row['lam']))
        j = etas.index(float(row['eta']))
        kappa_grid[i, j] = float(row['kappa_avg'])
        regime_grid[i, j] = int(row['regime'])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Regime map
    extent = [min(etas), max(etas), min(lams), max(lams)]
    im1 = ax1.imshow(regime_grid, origin='lower', extent=extent, cmap='RdYlGn_r', vmin=0, vmax=2, aspect='auto')
    ax1.set_xlabel('η (Resonance Gain)')
    ax1.set_ylabel('λ (Grounding)')
    ax1.set_title('Phase Regimes (Adaptive)')
    plt.colorbar(im1, ax=ax1, label='Regime')

    # Kappa heatmap
    im2 = ax2.imshow(kappa_grid, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    ax2.set_xlabel('η (Resonance Gain)')
    ax2.set_ylabel('λ (Grounding)')
    ax2.set_title('Conditioning κ(Σ)')
    plt.colorbar(im2, ax=ax2, label='κ_avg')

    plt.tight_layout()

    # Save SVG
    output_path = Path('docs/papers/neurips/figures/Geometric Theory of AI Hallucination/phase_adaptive_overlay.svg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg')
    print(f"Saved overlay to {output_path}")


if __name__ == '__main__':
    create_overlay()
