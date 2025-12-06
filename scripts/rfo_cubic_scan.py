"""
RFO Cubic Scan: Analytical Stability Analysis

Scans A-K parameter space using the corrected Padé(1,1) cubic characteristic
polynomial. Computes roots, discriminant, and classifies stability regimes.

Polynomial: P(s) = a3*s³ + a2*s² + a1*s + a0

Coefficients (corrected):
- a3 = Delta/2
- a2 = 1 + Delta*(A+B)/2
- a1 = (A+B) + Delta*(A*B + A*K)/2
- a0 = A*(B - K)

Regime Classification:
- unstable_dc: max(Re(λ)) > 0
- stable_ringing: max(Re(λ)) < 0 and Im(λ) ≠ 0 (complex conjugates)
- stable_overdamped: max(Re(λ)) < 0 and Im(λ) = 0 (all real)
"""

import numpy as np
from pathlib import Path
import csv


def compute_coefficients(A, B, K, Delta):
    """
    Compute corrected polynomial coefficients.

    Returns:
        tuple: (a3, a2, a1, a0)
    """
    a3 = Delta / 2
    a2 = 1 + Delta * (A + B) / 2
    a1 = (A + B) + Delta * (A * B + A * K) / 2
    a0 = A * (B - K)

    return a3, a2, a1, a0


def compute_cubic_discriminant(a3, a2, a1, a0):
    """
    Compute discriminant of cubic polynomial.

    For ax³ + bx² + cx + d:
    Δ = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²

    Discriminant interpretation:
    - Δ > 0: three distinct real roots
    - Δ = 0: repeated root
    - Δ < 0: one real root + two complex conjugates (RINGING)
    """
    a, b, c, d = a3, a2, a1, a0

    discriminant = (18*a*b*c*d - 4*b**3*d + b**2*c**2 -
                   4*a*c**3 - 27*a**2*d**2)

    return discriminant


def classify_regime(roots, tol=1e-5):
    """
    Classify stability regime based on eigenvalues.

    Returns:
        str: 'unstable_dc', 'stable_ringing', or 'stable_overdamped'
    """
    max_real = np.max(np.real(roots))

    if max_real > tol:
        return 'unstable_dc'

    # Stable - check for complex conjugates
    max_imag = np.max(np.abs(np.imag(roots)))

    if max_imag > tol:
        return 'stable_ringing'
    else:
        return 'stable_overdamped'


def scan_parameter_space():
    """
    Scan A-K parameter space and classify stability regimes.
    """
    # Fixed parameters
    B = 1.0
    Delta = 0.1

    # Scan ranges
    A_vals = np.linspace(0.2, 3.0, 100)
    K_vals = np.linspace(0.0, 3.0, 100)

    print("=" * 70)
    print("RFO Cubic Scan: Analytical Stability Analysis")
    print("=" * 70)
    print(f"Fixed parameters: B = {B}, Delta = {Delta}")
    print(f"Scan ranges: A ∈ [{A_vals.min():.2f}, {A_vals.max():.2f}]")
    print(f"             K ∈ [{K_vals.min():.2f}, {K_vals.max():.2f}]")
    print(f"Grid size: {len(A_vals)} × {len(K_vals)} = {len(A_vals)*len(K_vals)} points")
    print()

    # Storage
    results = []

    # Scan grid
    for i, A in enumerate(A_vals):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(A_vals)} ({100*i/len(A_vals):.1f}%)")

        for K in K_vals:
            # Compute coefficients
            a3, a2, a1, a0 = compute_coefficients(A, B, K, Delta)

            # Compute roots
            roots = np.roots([a3, a2, a1, a0])

            # Extract eigenvalue properties
            max_real = np.max(np.real(roots))
            max_imag = np.max(np.abs(np.imag(roots)))

            # Compute discriminant
            discriminant = compute_cubic_discriminant(a3, a2, a1, a0)

            # Classify regime
            regime = classify_regime(roots)

            # Store result
            results.append({
                'A': A,
                'K': K,
                'a3': a3,
                'a2': a2,
                'a1': a1,
                'a0': a0,
                'max_real': max_real,
                'max_imag': max_imag,
                'discriminant': discriminant,
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
    for regime, count in sorted(regime_counts.items()):
        print(f"{regime:20s}: {count:6d} ({100*count/total:5.1f}%)")

    # Compute ranges
    all_discriminants = [r['discriminant'] for r in results]
    all_max_real = [r['max_real'] for r in results]

    print()
    print(f"Discriminant range: [{min(all_discriminants):.2e}, "
          f"{max(all_discriminants):.2e}]")
    print(f"Max real eigenvalue range: [{min(all_max_real):.4f}, "
          f"{max(all_max_real):.4f}]")
    print()

    # Save to CSV
    output_path = Path(__file__).parent.parent / 'results' / 'rfo' / 'rfo_cubic_scan_AK.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['A', 'K', 'a3', 'a2', 'a1', 'a0',
                     'max_real', 'max_imag', 'discriminant', 'regime']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")
    print(f"Total rows: {len(results)}")
    print()

    # Report representative points (one per regime)
    print("=" * 70)
    print("Representative Points (one per regime)")
    print("=" * 70)

    for regime in ['stable_overdamped', 'stable_ringing', 'unstable_dc']:
        subset = [r for r in results if r['regime'] == regime]
        if len(subset) > 0:
            # Pick point near center of regime
            idx = len(subset) // 2
            row = subset[idx]
            print(f"\n{regime}:")
            print(f"  A = {row['A']:.4f}, K = {row['K']:.4f}")
            print(f"  max_real = {row['max_real']:.6f}")
            print(f"  max_imag = {row['max_imag']:.6f}")
            print(f"  discriminant = {row['discriminant']:.4e}")

    print()
    print("=" * 70)
    print("Scan complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    scan_parameter_space()
