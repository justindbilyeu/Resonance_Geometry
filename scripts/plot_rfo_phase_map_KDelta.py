"""
Plot RFO K-Delta Phase Map

Creates the "hero" phase map figure for the paper showing:
- Color regions by regime (unstable=white, overdamped=blue, ringing=red)
- Green contour: Ring Threshold (discriminant = 0)
- Black horizontal line: DC instability boundary (K = B)

Input: results/rfo/rfo_cubic_scan_KDelta.csv
Output: figures/rfo/phase_map_KDelta.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import csv


def load_scan_data(csv_path):
    """
    Load K-Delta scan data from CSV.

    Returns:
        dict: {'Delta': array, 'K': array, 'disc': array, 'regime': array}
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    Delta = np.array([float(r['Delta']) for r in data])
    K = np.array([float(r['K']) for r in data])
    disc = np.array([float(r['disc']) for r in data])
    regime = np.array([r['regime'] for r in data])

    return {
        'Delta': Delta,
        'K': K,
        'disc': disc,
        'regime': regime,
        'A': float(data[0]['A']),
        'B': float(data[0]['B'])
    }


def create_phase_map(data):
    """
    Create phase map visualization.
    """
    # Extract parameters
    Delta_vals = np.unique(data['Delta'])
    K_vals = np.unique(data['K'])
    A = data['A']
    B = data['B']

    N_delta = len(Delta_vals)
    N_K = len(K_vals)

    print(f"Grid dimensions: {N_delta} × {N_K}")
    print(f"Parameters: A = {A} s⁻¹, B = {B} s⁻¹")
    print()

    # Create 2D grid
    grid = np.zeros((N_K, N_delta))
    disc_grid = np.zeros((N_K, N_delta))

    # Map regimes to integers: unstable=0, overdamped=1, ringing=2
    regime_map = {'unstable': 0, 'overdamped': 1, 'ringing': 2}

    # Fill grid
    for i, (delta, k, regime, disc) in enumerate(zip(data['Delta'], data['K'],
                                                       data['regime'], data['disc'])):
        i_delta = np.argmin(np.abs(Delta_vals - delta))
        i_k = np.argmin(np.abs(K_vals - k))

        grid[i_k, i_delta] = regime_map[regime]
        disc_grid[i_k, i_delta] = disc

    # Count regimes
    regime_counts = {}
    for regime in data['regime']:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    total = len(data['regime'])
    print("Regime distribution:")
    for regime in ['overdamped', 'ringing', 'unstable']:
        count = regime_counts.get(regime, 0)
        print(f"  {regime:15s}: {count:6d} ({100*count/total:5.1f}%)")
    print()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map: unstable=white, overdamped=blue, ringing=red
    colors = ['#FFFFFF', '#3498DB', '#E74C3C']  # White, Blue, Red
    cmap = ListedColormap(colors)

    # Plot phase map
    extent = [Delta_vals.min(), Delta_vals.max(), K_vals.min(), K_vals.max()]
    im = ax.imshow(grid, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, interpolation='nearest', vmin=0, vmax=2)

    # Overlay: DC instability boundary K = B (horizontal black line)
    ax.axhline(y=B, color='black', linestyle='-', linewidth=2.5,
               label=f'DC Limit: K = B = {B} s$^{{-1}}$', zorder=10)

    # Overlay: Ring Threshold (discriminant = 0) - green contour
    # Find contour by plotting discriminant grid
    X, Y = np.meshgrid(Delta_vals, K_vals)

    # Plot zero contour of discriminant
    contour = ax.contour(X, Y, disc_grid, levels=[0], colors='lime',
                        linewidths=3.0, zorder=11)

    # Add label for ring threshold
    ax.plot([], [], color='lime', linestyle='-', linewidth=3.0,
            label='Ring Threshold (disc = 0)', zorder=11)

    # Labels and title
    ax.set_xlabel(r'Delay $\Delta$ [s]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Loop Gain $K$ [s$^{-1}$]', fontsize=14, fontweight='bold')
    ax.set_title(f'RFO K-$\Delta$ Phase Map: Stability Regimes (A={A}, B={B})',
                fontsize=16, fontweight='bold')

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.set_label('Stability Regime', fontsize=13, fontweight='bold')
    cbar.ax.set_yticklabels(['Unstable', 'Overdamped', 'Ringing (Motif)'])
    cbar.ax.tick_params(labelsize=11)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='gray')

    # Tight layout
    plt.tight_layout()

    return fig


def main():
    """
    Generate hero phase map plot.
    """
    print("=" * 70)
    print("RFO K-Delta Phase Map Plotter")
    print("=" * 70)
    print()

    # Load data
    csv_path = Path(__file__).parent.parent / 'results' / 'rfo' / 'rfo_cubic_scan_KDelta.csv'

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        print("Run scripts/rfo_cubic_scan_KDelta.py first.")
        return

    print(f"Loading data from: {csv_path}")
    data = load_scan_data(csv_path)
    print(f"Loaded {len(data['Delta'])} data points")
    print()

    # Create plot
    print("Creating phase map...")
    fig = create_phase_map(data)

    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures' / 'rfo'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'phase_map_KDelta.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\nHero plot saved to: {output_path}")
    print()
    print("=" * 70)
    print("Plot complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
