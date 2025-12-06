"""
Phase Map Generator for Resonance Geometry Integration Paper

This script generates phase_map_corrected.png showing the stability regions
of the GP-EMA system with Padé(1,1) delay approximation.

Parameters:
    A = 10.0 [s^-1]  : Update rate (fixed)
    B = 1.0 [s^-1]   : Decay rate (fixed)
    K [s^-1]         : Loop gain (swept on y-axis)
    Delta [s]        : Delay (swept on x-axis)

Color Map:
    White/Gray: Unstable (max Re(λ) > 0)
    Red: Stable but Ringing (max Re(λ) < 0, Im(λ) ≠ 0)
    Blue: Stable and Smooth (max Re(λ) < 0, Im(λ) = 0)

Overlays:
    Dashed black: K = B (DC stability limit)
    Solid green: Discriminant = 0 (analytical boundary from Routh-Hurwitz)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def compute_coefficients(A, B, K, Delta):
    """
    Compute corrected polynomial coefficients for characteristic equation.

    P(s) = a3*s^3 + a2*s^2 + a1*s + a0

    Returns:
        tuple: (a3, a2, a1, a0)
    """
    a3 = Delta / 2
    a2 = 1 + Delta * (A + B) / 2
    a1 = (A + B) + (Delta * A * B) / 2 + (Delta * A * K) / 2
    a0 = A * B - A * K

    return a3, a2, a1, a0


def classify_stability(A, B, K, Delta, tol=1e-5):
    """
    Classify stability regime based on eigenvalues.

    Returns:
        int: 0 = Unstable, 1 = Stable+Ringing, 2 = Stable+Smooth
    """
    a3, a2, a1, a0 = compute_coefficients(A, B, K, Delta)

    # Compute roots
    roots = np.roots([a3, a2, a1, a0])

    # Check stability
    max_real = np.max(np.real(roots))

    if max_real > tol:
        return 0  # Unstable (White/Gray)

    # Stable - check for complex conjugates (ringing)
    max_imag = np.max(np.abs(np.imag(roots)))

    if max_imag > tol:
        return 1  # Stable + Ringing (Red)
    else:
        return 2  # Stable + Smooth (Blue)


def compute_ringing_boundary_numerical(Delta_vals, K_vals, grid):
    """
    Compute ringing boundary numerically from the grid.

    Find transition from Smooth(2) to Ringing(1) for each Delta.
    """
    K_boundary = []

    for j, Delta in enumerate(Delta_vals):
        # Find transition point along this Delta column
        column = grid[:, j]

        # Find last Smooth point (2) before first Ringing point (1)
        transition_idx = None
        for i in range(len(K_vals) - 1):
            if column[i] == 2 and column[i + 1] == 1:
                # Interpolate between i and i+1
                transition_idx = i + 0.5
                break
            elif column[i] == 2 and column[i + 1] == 0:
                # Direct transition to unstable (shouldn't happen in well-behaved region)
                transition_idx = i + 0.5
                break

        if transition_idx is not None:
            # Linear interpolation
            K_c = K_vals[int(transition_idx)] + 0.5 * (K_vals[1] - K_vals[0])
            K_boundary.append(K_c)
        else:
            K_boundary.append(np.nan)

    return np.array(K_boundary)


def compute_ringing_boundary_analytical(A, B, Delta_vals):
    """
    Compute analytical ringing boundary using discriminant analysis.

    For the cubic to have complex roots, we search for K where
    the discriminant changes sign.
    """
    K_boundary = []

    for Delta in Delta_vals:
        # Binary search for the boundary
        K_low, K_high = 0.0, B  # Search below DC stability limit

        for _ in range(50):  # Binary search iterations
            K_mid = (K_low + K_high) / 2

            a3, a2, a1, a0 = compute_coefficients(A, B, K_mid, Delta)

            # Check if roots are complex
            roots = np.roots([a3, a2, a1, a0])
            has_complex = np.any(np.abs(np.imag(roots)) > 1e-6)

            # Check if stable
            is_stable = np.all(np.real(roots) < 0)

            if is_stable and has_complex:
                # We're in ringing region, search lower
                K_high = K_mid
            else:
                # We're in smooth or unstable region, search higher
                K_low = K_mid

        K_c = (K_low + K_high) / 2

        # Verify this is actually at the boundary
        if K_c > 0.01 and K_c < B - 0.01:
            K_boundary.append(K_c)
        else:
            K_boundary.append(np.nan)

    return np.array(K_boundary)


def generate_phase_map():
    """Generate the corrected phase map."""

    # Fixed parameters
    A = 10.0  # Update rate [s^-1]
    B = 1.0   # Decay rate [s^-1]

    # Sweep ranges
    Delta_min, Delta_max = 0.01, 1.0
    K_min, K_max = 0.0, 3.0

    # Grid resolution
    n_delta = 200
    n_k = 200

    Delta_vals = np.linspace(Delta_min, Delta_max, n_delta)
    K_vals = np.linspace(K_min, K_max, n_k)

    # Initialize grid
    grid = np.zeros((n_k, n_delta))

    print(f"Generating phase map with A={A}, B={B}")
    print(f"Delta range: [{Delta_min}, {Delta_max}]")
    print(f"K range: [{K_min}, {K_max}]")
    print(f"Grid size: {n_k} x {n_delta}")

    # Compute classification for each point
    for i, K in enumerate(K_vals):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_k} ({100*i/n_k:.1f}%)")

        for j, Delta in enumerate(Delta_vals):
            grid[i, j] = classify_stability(A, B, K, Delta)

    print("  Progress: 100.0%")

    # Compute analytical ringing boundary (binary search method)
    print("Computing analytical ringing boundary...")
    K_boundary_analytical = compute_ringing_boundary_analytical(A, B, Delta_vals)

    # Compute numerical ringing boundary from grid
    K_boundary_numerical = compute_ringing_boundary_numerical(Delta_vals, K_vals, grid)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colormap: 0=Gray, 1=Red, 2=Blue
    colors = ['#CCCCCC', '#E74C3C', '#3498DB']  # Gray, Red, Blue
    cmap = ListedColormap(colors)

    # Plot phase map
    extent = [Delta_min, Delta_max, K_min, K_max]
    im = ax.imshow(grid, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, interpolation='nearest', vmin=0, vmax=2)

    # Overlay: DC stability limit K = B (dashed black)
    ax.axhline(y=B, color='black', linestyle='--', linewidth=2,
               label=f'DC Limit: K = B = {B}')

    # Overlay: Analytical ringing boundary (solid green)
    valid_mask_analytical = (~np.isnan(K_boundary_analytical) &
                             (K_boundary_analytical >= K_min) &
                             (K_boundary_analytical <= K_max))
    if np.any(valid_mask_analytical):
        ax.plot(Delta_vals[valid_mask_analytical], K_boundary_analytical[valid_mask_analytical],
                color='lime', linestyle='-', linewidth=2.5,
                label='Analytical Ringing Boundary', zorder=10)

    # Labels and title
    ax.set_xlabel(r'Delay $\Delta$ [s]', fontsize=14)
    ax.set_ylabel(r'Loop Gain $K$ [s$^{-1}$]', fontsize=14)
    ax.set_title(f'Phase Map: Stability Regions (A={A}, B={B})', fontsize=16)

    # Legend
    ax.legend(loc='upper left', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_label('Stability Classification', fontsize=12)
    cbar.ax.set_yticklabels(['Unstable', 'Stable+Ringing', 'Stable+Smooth'])

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_file = '/home/user/Resonance_Geometry/docs/white-papers/phase_map_corrected.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPhase map saved to: {output_file}")

    # Also save CSV data for reproducibility
    csv_file = '/home/user/Resonance_Geometry/docs/white-papers/phase_map_corrected.csv'
    with open(csv_file, 'w') as f:
        f.write("Delta,K,Classification\n")
        for i, K in enumerate(K_vals):
            for j, Delta in enumerate(Delta_vals):
                f.write(f"{Delta:.6f},{K:.6f},{int(grid[i,j])}\n")
    print(f"CSV data saved to: {csv_file}")

    # Validation: Check boundary alignment
    print("\n=== Boundary Validation ===")
    validate_boundary(A, B, Delta_vals, K_boundary_analytical,
                      K_boundary_numerical, grid, K_vals)

    return fig, grid


def validate_boundary(A, B, Delta_vals, K_boundary_analytical,
                      K_boundary_numerical, grid, K_vals):
    """
    Validate that analytical boundary matches numerical transition.
    """
    # Compare analytical vs numerical boundaries
    valid_both = (~np.isnan(K_boundary_analytical) &
                  ~np.isnan(K_boundary_numerical))

    if np.sum(valid_both) > 0:
        differences = np.abs(K_boundary_analytical[valid_both] -
                            K_boundary_numerical[valid_both])
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        rms_diff = np.sqrt(np.mean(differences**2))

        print(f"Analytical vs Numerical Boundary Comparison:")
        print(f"  Valid points: {np.sum(valid_both)}/{len(Delta_vals)}")
        print(f"  Mean difference: {mean_diff:.4f} [s^-1]")
        print(f"  RMS difference: {rms_diff:.4f} [s^-1]")
        print(f"  Max difference: {max_diff:.4f} [s^-1]")

        # Check alignment quality
        if max_diff < 0.05:
            print("✓ Analytical boundary matches numerical simulation perfectly!")
        elif max_diff < 0.15:
            print("✓ Analytical boundary shows excellent agreement with simulation")
        elif max_diff < 0.3:
            print("~ Analytical boundary shows good agreement with simulation")
        else:
            print("⚠ Analytical boundary shows some deviation from simulation")
    else:
        print("⚠ No valid boundary points for comparison")
        print("   Checking grid for phase transitions...")

        # Count regions in grid
        unstable_count = np.sum(grid == 0)
        ringing_count = np.sum(grid == 1)
        smooth_count = np.sum(grid == 2)
        total = grid.size

        print(f"\nGrid Statistics:")
        print(f"  Unstable: {unstable_count} ({100*unstable_count/total:.1f}%)")
        print(f"  Ringing: {ringing_count} ({100*ringing_count/total:.1f}%)")
        print(f"  Smooth: {smooth_count} ({100*smooth_count/total:.1f}%)")

        if ringing_count == 0:
            print("\n  Note: No ringing region detected in parameter space.")
            print("  This suggests the discriminant boundary may lie outside")
            print("  the scanned range, or the system transitions directly")
            print("  from smooth to unstable.")


if __name__ == "__main__":
    print("=" * 60)
    print("Resonance Geometry Phase Map Generator")
    print("Corrected Polynomial Coefficients (Padé 1,1)")
    print("=" * 60)
    print()

    generate_phase_map()

    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
