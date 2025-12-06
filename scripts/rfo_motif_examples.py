"""
Generate motif examples showing impulse responses across the K-Δ wedge.

Uses analytical construction from Padé poles.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

A = 10.0
B = 1.0
Delta_fixed = 0.15  # Fixed delay

def cubic_coeffs(K, Delta):
    a3 = Delta / 2.0
    a2 = 1.0 + Delta * (A + B) / 2.0
    a1 = (A + B) + (Delta / 2.0) * (A * B + A * K)
    a0 = A * B - A * K
    return [a3, a2, a1, a0]

def construct_impulse_response(K, Delta, t):
    """
    Construct approximate impulse response from pole locations.

    For a 3rd-order system with poles s1, s2, s3:
    g(t) ≈ Σ c_i * exp(s_i * t)

    For complex conjugates: exp(σt)[c*cos(ωt) + d*sin(ωt)]
    """
    coeffs = cubic_coeffs(K, Delta)
    poles = np.roots(coeffs)

    # Sort poles
    complex_poles = [p for p in poles if abs(p.imag) > 1e-6]
    real_poles = [p.real for p in poles if abs(p.imag) <= 1e-6]

    g = np.zeros_like(t)

    # Contribution from complex conjugate pair
    if len(complex_poles) >= 2:
        sigma = complex_poles[0].real
        omega = abs(complex_poles[0].imag)

        # Empirical amplitude and phase for visualization
        amp = 0.8
        phase = 0.0

        g += amp * np.exp(sigma * t) * np.cos(omega * t + phase)

    # Contribution from real poles
    for i, s_real in enumerate(real_poles):
        amp = 0.4 if i == 0 else 0.2
        g += amp * np.exp(s_real * t)

    # Normalize to start near 1.0
    if len(g) > 0 and abs(g[0]) > 1e-10:
        g = g / g[0]

    return g

def generate_motif_examples():
    """Generate motif example plots."""

    print("="*70)
    print("Generating RFO Motif Examples")
    print("="*70)
    print(f"Fixed parameters: A={A}, B={B}, Δ={Delta_fixed}")
    print()

    # K values across the wedge
    # From earlier analysis, threshold at Δ=0.15 is K≈0.119
    K_threshold = 0.119

    test_cases = [
        {'K': 0.05, 'label': 'Deep Overdamped (K≪K_c)', 'color': 'blue'},
        {'K': 0.30, 'label': 'Mid-Wedge Ringing (K>K_c)', 'color': 'red'},
        {'K': 0.70, 'label': 'Strong Ringing (K→B)', 'color': 'darkred'},
        {'K': 1.05, 'label': 'Unstable (K>B)', 'color': 'black'},
    ]

    # Time array
    t = np.linspace(0, 12, 2000)

    # Create figure
    fig, axes = plt.subplots(len(test_cases), 1, figsize=(12, 10), sharex=True)

    print(f"{'K':<8} | {'Regime':<20} | {'Poles':<40}")
    print("-"*70)

    for idx, case in enumerate(test_cases):
        K = case['K']
        label = case['label']
        color = case['color']

        # Get poles
        coeffs = cubic_coeffs(K, Delta_fixed)
        poles = np.roots(coeffs)
        max_real = np.max(np.real(poles))
        has_complex = np.any(np.abs(np.imag(poles)) > 1e-6)

        # Classify
        if K >= B or max_real >= 0:
            regime = "Unstable"
        elif has_complex:
            regime = "Ringing"
        else:
            regime = "Overdamped"

        # Construct response
        g = construct_impulse_response(K, Delta_fixed, t)

        # Plot
        ax = axes[idx]
        ax.plot(t, g, color=color, linewidth=2, label=label)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_ylabel('g(t)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Annotate with regime
        ax.text(0.02, 0.05, f'{regime}', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Print pole info
        pole_str = ", ".join([f"{p.real:.2f}+{p.imag:.2f}j" if abs(p.imag)>1e-6
                             else f"{p.real:.2f}" for p in poles])
        print(f"{K:<8.2f} | {regime:<20} | {pole_str}")

    axes[-1].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    axes[0].set_title(f'RFO Impulse Response: Motif Examples (A={A}, B={B}, Δ={Delta_fixed} s)',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'figures' / 'rfo'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'motif_examples.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print()
    print(f"Saved: {output_path}")
    print()
    print("="*70)
    print("Motif examples complete!")
    print("="*70)

if __name__ == "__main__":
    generate_motif_examples()
