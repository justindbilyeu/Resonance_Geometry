"""
RFO Motif Phase Map: Simulation-Based Classification

Performs full A-K parameter sweep with numerical simulation.
Classifies ringing behavior using RMS and overshoot thresholds.
Combines with analytical regime classification for validation.

Output: results/rfo/rfo_motif_phase_map_AK.csv
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import csv


def simulate_rfo_fast(A, B, K, Delta, T_max=30.0, dt=0.05):
    """
    Fast simulation for phase map generation.

    Returns:
        dict: {'t': time, 'g': state}
    """
    # Initial condition
    x0 = np.array([0.1, 0.0, 0.0])

    # ODE system (same as rfo_timeseries_demo.py)
    def rhs(t, x):
        g, Ibar, I_delayed = x

        dg = K * Ibar - B * g
        dIbar = A * (I_delayed - Ibar)

        tau_delay = Delta / 2
        dI_delayed = (K * g - I_delayed) / tau_delay

        return np.array([dg, dIbar, dI_delayed])

    # Time grid
    t_span = (0, T_max)
    t_eval = np.arange(0, T_max, dt)

    # Solve
    try:
        sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method='RK45',
                       rtol=1e-6, atol=1e-8)
        return {'t': sol.t, 'g': sol.y[0, :]}
    except:
        # Return NaN on failure
        return {'t': t_eval, 'g': np.full_like(t_eval, np.nan)}


def detect_ringing(t, g, rms_threshold=0.01, overshoot_threshold=2):
    """
    Detect ringing using RMS and overshoot criteria.

    Args:
        t: time array
        g: state array
        rms_threshold: minimum RMS for ringing
        overshoot_threshold: minimum number of overshoots

    Returns:
        bool: True if ringing detected
    """
    # Check for NaN
    if np.any(np.isnan(g)):
        return False

    # Check for blow-up
    if np.max(np.abs(g)) > 100.0:
        return False

    # Use second half for steady-state analysis
    n_half = len(g) // 2
    if n_half < 10:
        return False

    g_steady = g[n_half:]

    # Detrend
    g_mean = np.mean(g_steady)
    g_detrended = g_steady - g_mean

    # Compute RMS
    rms = np.sqrt(np.mean(g_detrended**2))

    # Count zero crossings (overshoots)
    zero_crossings = np.sum(np.diff(np.sign(g_detrended)) != 0)

    # Classify
    is_ringing = (rms > rms_threshold and zero_crossings > overshoot_threshold)

    return is_ringing


def generate_phase_map():
    """
    Generate full phase map with simulation data.
    """
    print("=" * 70)
    print("RFO Motif Phase Map: Simulation Sweep")
    print("=" * 70)
    print()

    # Load analytical results
    scan_file = Path(__file__).parent.parent / 'results' / 'rfo' / 'rfo_cubic_scan_AK.csv'

    if not scan_file.exists():
        print(f"Error: {scan_file} not found. Run rfo_cubic_scan.py first.")
        return

    # Read analytical data
    with open(scan_file, 'r') as f:
        reader = csv.DictReader(f)
        analytical_data = list(reader)

    print(f"Loaded {len(analytical_data)} analytical points")
    print()

    # Fixed parameters
    B = 1.0
    Delta = 0.1

    # Simulate each point
    print("Running simulations...")
    print()

    results = []
    total = len(analytical_data)

    for i, row in enumerate(analytical_data):
        if i % 500 == 0:
            print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")

        A = float(row['A'])
        K = float(row['K'])

        # Run simulation
        sim_result = simulate_rfo_fast(A, B, K, Delta)

        # Detect ringing
        is_ringing = detect_ringing(sim_result['t'], sim_result['g'])

        # Compute RMS
        g = sim_result['g']
        if len(g) > 0 and not np.any(np.isnan(g)):
            g_steady = g[len(g)//2:]
            g_detrended = g_steady - np.mean(g_steady)
            rms = np.sqrt(np.mean(g_detrended**2))
            max_abs = np.max(np.abs(g))
        else:
            rms = np.nan
            max_abs = np.nan

        # Combine with analytical data
        result = {
            'A': A,
            'K': K,
            'max_real': float(row['max_real']),
            'max_imag': float(row['max_imag']),
            'discriminant': float(row['discriminant']),
            'regime_analytical': row['regime'],
            'is_ringing_simulation': int(is_ringing),
            'rms': rms,
            'max_abs': max_abs
        }

        results.append(result)

    print(f"  Progress: 100.0%")
    print()

    # Save combined data
    output_file = Path(__file__).parent.parent / 'results' / 'rfo' / 'rfo_motif_phase_map_AK.csv'

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['A', 'K', 'max_real', 'max_imag', 'discriminant',
                     'regime_analytical', 'is_ringing_simulation', 'rms', 'max_abs']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_file}")
    print(f"Total rows: {len(results)}")
    print()

    # Summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Count analytical regimes
    regime_counts = {}
    for r in results:
        regime = r['regime_analytical']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print("\nAnalytical Regimes:")
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime:20s}: {count:6d} ({100*count/len(results):5.1f}%)")

    # Count simulation ringing
    ringing_count = sum(r['is_ringing_simulation'] for r in results)
    print(f"\nSimulation-Detected Ringing:")
    print(f"  Ringing: {ringing_count:6d} ({100*ringing_count/len(results):5.1f}%)")
    print(f"  No Ringing: {len(results)-ringing_count:6d} "
          f"({100*(len(results)-ringing_count)/len(results):5.1f}%)")

    # RMS statistics
    valid_rms = [r['rms'] for r in results if not np.isnan(r['rms'])]
    if valid_rms:
        print(f"\nRMS Statistics:")
        print(f"  Mean: {np.mean(valid_rms):.4f}")
        print(f"  Median: {np.median(valid_rms):.4f}")
        print(f"  Max: {np.max(valid_rms):.4f}")

    print()
    print("=" * 70)
    print("Phase map generation complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    generate_phase_map()
