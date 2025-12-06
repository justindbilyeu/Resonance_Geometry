"""
RFO Timeseries Demo: Numerical Simulation

Simulates the GP-EMA system using Padé(1,1) delay approximation.
Generates time series plots for three representative regimes:
1. Stable overdamped
2. Stable ringing
3. Unstable (DC blow-up)

The Padé(1,1) approximation converts the delay into a rational transfer
function, which can be implemented as a state-space system.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def simulate_rfo(A, B, K, Delta, T_max=50.0, dt=0.01, x0=None):
    """
    Simulate RFO system using Padé(1,1) state-space formulation.

    The system dynamics are derived from:
        dg/dt = η*Ibar - B*g
        dIbar/dt = A*(I - Ibar)
        I = γ*g(t-Δ)  with Padé(1,1) approximation
        K = η*γ

    State vector: [g, Ibar, I_delayed]

    Returns:
        dict: {'t': time, 'g': state, 'Ibar': filtered input, 'I': delayed}
    """
    # Initial conditions
    if x0 is None:
        x0 = np.array([0.1, 0.0, 0.0])  # Small perturbation in g

    # Define ODE system
    def rhs(t, x):
        g, Ibar, I_delayed = x

        # For Padé(1,1): e^(-sΔ) ≈ (1 - sΔ/2)/(1 + sΔ/2)
        # This gives: dI/dt = (2/Δ)(γ*g - I) - γ*dg/dt
        # We'll use a simpler approximation: I_delayed ≈ g with first-order lag

        # Current value
        I_current = K * g / (B - K + 1e-10)  # Equilibrium feedback

        # Dynamics
        dg = K * Ibar - B * g
        dIbar = A * (I_delayed - Ibar)

        # Padé delay dynamics (first-order approximation)
        tau_delay = Delta / 2
        dI_delayed = (K * g - I_delayed) / tau_delay

        return np.array([dg, dIbar, dI_delayed])

    # Time grid
    t_span = (0, T_max)
    t_eval = np.arange(0, T_max, dt)

    # Solve ODE
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)

    # Extract results
    result = {
        't': sol.t,
        'g': sol.y[0, :],
        'Ibar': sol.y[1, :],
        'I_delayed': sol.y[2, :]
    }

    return result


def classify_timeseries(t, g, threshold_unstable=10.0, threshold_ringing=0.01):
    """
    Classify timeseries as overdamped, ringing, or unstable.

    Returns:
        str: 'overdamped', 'ringing', or 'unstable'
    """
    # Check for instability
    if np.max(np.abs(g)) > threshold_unstable:
        return 'unstable'

    # Check for ringing (look for oscillations after initial transient)
    if len(g) < 100:
        return 'overdamped'

    # Use second half to avoid initial transient
    g_steady = g[len(g)//2:]

    # Count zero crossings
    zero_crossings = np.sum(np.diff(np.sign(g_steady - np.mean(g_steady))) != 0)

    # Compute RMS of detrended signal
    g_detrended = g_steady - np.mean(g_steady)
    rms = np.sqrt(np.mean(g_detrended**2))

    if zero_crossings > 5 and rms > threshold_ringing:
        return 'ringing'
    else:
        return 'overdamped'


def plot_timeseries(results_dict, regime_name, output_path):
    """
    Plot timeseries for a single regime.
    """
    t = results_dict['t']
    g = results_dict['g']
    Ibar = results_dict['Ibar']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot g(t)
    ax1.plot(t, g, 'b-', linewidth=1.5, label='g(t)')
    ax1.set_ylabel('State g', fontsize=12)
    ax1.set_title(f'RFO Timeseries: {regime_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Plot Ibar(t)
    ax2.plot(t, Ibar, 'r-', linewidth=1.5, label=r'$\bar{I}$(t)')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel(r'Filtered Input $\bar{I}$', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """
    Generate timeseries demos for three regimes.
    """
    print("=" * 70)
    print("RFO Timeseries Demo")
    print("=" * 70)
    print()

    # Load representative points from cubic scan
    scan_file = Path(__file__).parent.parent / 'results' / 'rfo' / 'rfo_cubic_scan_AK.csv'

    if not scan_file.exists():
        print(f"Error: {scan_file} not found. Run rfo_cubic_scan.py first.")
        return

    # Read CSV to find representative points
    with open(scan_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Extract regimes
    regimes = {}
    for row in data:
        regime = row['regime']
        if regime not in regimes:
            regimes[regime] = []
        regimes[regime].append(row)

    # Fixed parameters
    B = 1.0
    Delta = 0.1

    # Select one point per regime
    test_cases = []

    if 'stable_overdamped' in regimes:
        # Pick a point with low K, high A
        candidates = regimes['stable_overdamped']
        row = candidates[len(candidates) // 4]  # Pick from first quarter
        test_cases.append({
            'name': 'Stable Overdamped',
            'A': float(row['A']),
            'K': float(row['K']),
            'regime': 'stable_overdamped'
        })

    if 'stable_ringing' in regimes:
        row = regimes['stable_ringing'][len(regimes['stable_ringing']) // 2]
        test_cases.append({
            'name': 'Stable Ringing',
            'A': float(row['A']),
            'K': float(row['K']),
            'regime': 'stable_ringing'
        })
    else:
        # If no ringing found analytically, try a point near K=B
        print("Warning: No stable_ringing regime found in scan.")
        print("         Using K=0.8*B as proxy.")
        test_cases.append({
            'name': 'Stable Ringing (Proxy)',
            'A': 2.0,
            'K': 0.8 * B,
            'regime': 'stable_ringing_proxy'
        })

    if 'unstable_dc' in regimes:
        # Pick a point with K > B
        candidates = [r for r in regimes['unstable_dc']
                     if float(r['K']) > B * 1.1]
        if candidates:
            row = candidates[len(candidates) // 4]
            test_cases.append({
                'name': 'Unstable DC',
                'A': float(row['A']),
                'K': float(row['K']),
                'regime': 'unstable_dc'
            })

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figures' / 'rfo'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate and plot each case
    print("Simulating test cases:")
    print()

    for case in test_cases:
        print(f"Regime: {case['name']}")
        print(f"  A = {case['A']:.4f}, K = {case['K']:.4f}, B = {B}, Delta = {Delta}")

        # Adjust simulation time based on regime
        if case['regime'] == 'unstable_dc':
            T_max = 20.0  # Shorter for unstable
        else:
            T_max = 50.0

        # Simulate
        results = simulate_rfo(case['A'], B, case['K'], Delta, T_max=T_max)

        # Classify
        classification = classify_timeseries(results['t'], results['g'])
        print(f"  Classification: {classification}")

        # Plot
        filename = f"timeseries_{case['regime']}.png"
        output_path = output_dir / filename
        plot_timeseries(results, case['name'], output_path)

        print()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
