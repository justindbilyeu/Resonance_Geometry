"""
RFO Cubic Scan: K-Delta Phase Map (Analytical)

Builds the K-Δ stability map using the Padé(1,1) cubic characteristic equation.
No time-domain simulation - purely analytical classification.

This generates the "hero" phase map for the paper.

Model:
    Characteristic polynomial: a₃s³ + a₂s² + a₁s + a₀ = 0

    Coefficients:
        a₃ = Δ/2
        a₂ = 1 + Δ(A+B)/2
        a₁ = (A+B) + Δ(AB + AK)/2
        a₀ = AB - AK

Classification:
    - unstable: K > B or max(Re(s)) ≥ 0
    - ringing: stable and discriminant < 0 (complex conjugates)
    - overdamped: stable and discriminant ≥ 0 (all real roots)
"""

import numpy as np
from pathlib import Path
import csv


def compute_coefficients(A, B, K, Delta):
    """
    Compute Padé(1,1) cubic coefficients.

    Returns:
        tuple: (a3, a2, a1, a0)
    """
    a3 = Delta / 2.0
    a2 = 1.0 + Delta * (A + B) / 2.0
    a1 = (A + B) + Delta * (A*B + A*K) / 2.0
    a0 = A*B - A*K

    return a3, a2, a1, a0


def compute_cubic_discriminant(a3, a2, a1, a0):
    """
    Compute discriminant of cubic polynomial ax³ + bx² + cx + d.

    Δ = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²

    Interpretation:
        Δ > 0: three distinct real roots (overdamped)
        Δ = 0: repeated root
        Δ < 0: one real + two complex conjugates (ringing)
    """
    disc = (18*a3*a2*a1*a0
            - 4*a2**3*a0
            + a2**2*a1**2
            - 4*a3*a1**3
            - 27*a3**2*a0**2)

    return disc


def classify_regime(K, B, max_real, disc):
    """
    Classify stability regime.

    Priority:
        1. If K > B or max_real >= 0 → unstable (DC explosion)
        2. Else if disc < 0 → ringing (stable underdamped)
        3. Else → overdamped (stable, no ringing)

    Returns:
        str: 'unstable', 'ringing', or 'overdamped'
    """
    if K > B or max_real >= 0:
        return 'unstable'
    elif disc < 0:
        return 'ringing'
    else:
        return 'overdamped'


def scan_k_delta_space():
    """
    Scan K-Delta parameter space and classify stability regimes.
    """
    # Fixed parameters
    A = 10.0  # s⁻¹
    B = 1.0   # s⁻¹

    # Sweep ranges
    N_delta = 100
    N_K = 200

    Delta_vals = np.linspace(0.01, 0.5, N_delta)
    K_vals = np.linspace(0.0, 5.0, N_K)

    print("=" * 70)
    print("RFO Cubic Scan: K-Delta Phase Map (Analytical)")
    print("=" * 70)
    print(f"Fixed parameters: A = {A} s⁻¹, B = {B} s⁻¹")
    print(f"Sweep ranges:")
    print(f"  Delta ∈ [{Delta_vals.min():.2f}, {Delta_vals.max():.2f}] s")
    print(f"  K ∈ [{K_vals.min():.2f}, {K_vals.max():.2f}] s⁻¹")
    print(f"Grid size: {N_delta} × {N_K} = {N_delta * N_K} points")
    print()

    # Storage
    results = []

    # Scan grid
    for i, Delta in enumerate(Delta_vals):
        if i % 10 == 0:
            print(f"Progress: {i}/{N_delta} ({100*i/N_delta:.1f}%)")

        for K in K_vals:
            # Compute coefficients
            a3, a2, a1, a0 = compute_coefficients(A, B, K, Delta)

            # Compute roots
            roots = np.roots([a3, a2, a1, a0])
            max_real = np.max(np.real(roots))

            # Compute discriminant
            disc = compute_cubic_discriminant(a3, a2, a1, a0)

            # Classify regime
            regime = classify_regime(K, B, max_real, disc)

            # Store result
            results.append({
                'Delta': Delta,
                'K': K,
                'A': A,
                'B': B,
                'a0': a0,
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'max_real': max_real,
                'disc': disc,
                'regime': regime
            })

    print(f"Progress: 100.0%")
    print()

    # Summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Count regimes
    regime_counts = {}
    for r in results:
        regime = r['regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    total = len(results)
    for regime in ['overdamped', 'ringing', 'unstable']:
        count = regime_counts.get(regime, 0)
        print(f"  {regime:15s}: {count:6d} ({100*count/total:5.1f}%)")

    # Discriminant statistics
    all_disc = [r['disc'] for r in results]
    print(f"\nDiscriminant range: [{min(all_disc):.4e}, {max(all_disc):.4e}]")

    # Max real eigenvalue statistics
    all_max_real = [r['max_real'] for r in results]
    print(f"Max Re(s) range: [{min(all_max_real):.4f}, {max(all_max_real):.4f}]")
    print()

    # Save to CSV
    output_path = Path(__file__).parent.parent / 'results' / 'rfo' / 'rfo_cubic_scan_KDelta.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['Delta', 'K', 'A', 'B', 'a0', 'a1', 'a2', 'a3',
                     'max_real', 'disc', 'regime']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")
    print(f"Total rows: {len(results)}")
    print()

    # Report Ring Threshold boundary points
    print("=" * 70)
    print("Ring Threshold Boundary (disc = 0)")
    print("=" * 70)

    # Find points near disc = 0
    boundary_points = []
    for r in results:
        if r['regime'] in ['ringing', 'overdamped'] and abs(r['disc']) < 0.01:
            boundary_points.append(r)

    if boundary_points:
        print(f"Found {len(boundary_points)} points near ring threshold")
        # Show a few examples
        for i, pt in enumerate(boundary_points[:5]):
            print(f"  Δ = {pt['Delta']:.4f} s, K = {pt['K']:.4f} s⁻¹, "
                  f"disc = {pt['disc']:.4e}")
    else:
        print("No points found near ring threshold in scanned region")

    print()
    print("=" * 70)
    print("Scan complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    scan_k_delta_space()
