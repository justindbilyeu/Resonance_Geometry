"""
RFO Validation: Analytical Ring Threshold vs Full DDE Simulation

Compares the Padé(1,1) cubic discriminant threshold with numerical DDE simulation.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from pathlib import Path

# --- Physics Parameters ---
A = 10.0   # Fast update rate [s^-1]
B = 1.0    # Decay rate [s^-1]

# --- 1. Analytic Functions ---

def cubic_coeffs(K, Delta):
    """Returns [a3, a2, a1, a0] for the Pade(1,1) characteristic eq."""
    a3 = Delta / 2.0
    a2 = 1.0 + Delta * (A + B) / 2.0
    a1 = (A + B) + (Delta / 2.0) * (A * B + A * K)
    a0 = A * B - A * K
    return [a3, a2, a1, a0]

def get_cubic_discriminant(K, Delta):
    """Calculates the discriminant of the cubic polynomial."""
    a3, a2, a1, a0 = cubic_coeffs(K, Delta)
    a, b, c, d = a3, a2, a1, a0
    disc = (18*a*b*c*d) - (4*(b**3)*d) + ((b**2)*(c**2)) - (4*a*(c**3)) - (27*(a**2)*(d**2))
    return disc

def find_analytic_threshold(Delta):
    """Finds K_c where Discriminant = 0 (using root finding)."""
    # Check if there's a sign change
    disc_low = get_cubic_discriminant(0.0, Delta)
    disc_high = get_cubic_discriminant(B - 1e-4, Delta)

    # If both same sign, no crossing
    if disc_low * disc_high > 0:
        return None

    try:
        root = brentq(lambda k: get_cubic_discriminant(k, Delta), 0.0, B - 1e-4)
        return root
    except ValueError:
        return None

# --- 2. Improved DDE Simulation ---

def simulate_dde_rk4(K, Delta, T_max=20.0, dt=0.0005):
    """
    Simulates the FULL DDE using RK4: g'' + (A+B)g' + ABg = AK g(t-Delta)

    State: x = [g, g']
    Returns: time array, g array
    """
    steps = int(T_max / dt)
    delay_steps = int(Delta / dt)

    # History buffer
    g_history = np.zeros(steps)
    gdot_history = np.zeros(steps)

    # Initial condition: impulse
    g_history[0] = 1.0
    gdot_history[0] = 0.0

    for i in range(steps - 1):
        # Current state
        g = g_history[i]
        gdot = gdot_history[i]

        # Delayed value
        if i >= delay_steps:
            g_delayed = g_history[i - delay_steps]
        else:
            g_delayed = 0.0

        # RK4 for the system:
        # g' = gdot
        # gdot' = -(A+B)*gdot - A*B*g + A*K*g_delayed

        def derivatives(g_curr, gdot_curr, g_del):
            dg = gdot_curr
            dgdot = -(A + B)*gdot_curr - A*B*g_curr + A*K*g_del
            return dg, dgdot

        # k1
        k1_g, k1_gdot = derivatives(g, gdot, g_delayed)

        # k2 (half-step delayed value - approximate)
        k2_g, k2_gdot = derivatives(g + 0.5*dt*k1_g,
                                    gdot + 0.5*dt*k1_gdot,
                                    g_delayed)

        # k3
        k3_g, k3_gdot = derivatives(g + 0.5*dt*k2_g,
                                    gdot + 0.5*dt*k2_gdot,
                                    g_delayed)

        # k4
        k4_g, k4_gdot = derivatives(g + dt*k3_g,
                                    gdot + dt*k3_gdot,
                                    g_delayed)

        # Update
        g_history[i+1] = g + (dt/6.0)*(k1_g + 2*k2_g + 2*k3_g + k4_g)
        gdot_history[i+1] = gdot + (dt/6.0)*(k1_gdot + 2*k2_gdot + 2*k3_gdot + k4_gdot)

        # Check for blow-up
        if abs(g_history[i+1]) > 1000.0:
            # Unstable - pad rest with NaN
            g_history[i+1:] = np.nan
            break

    return g_history

def check_ringing_improved(K, Delta, T_max=20.0, dt=0.0005):
    """
    Improved ringing detection.

    Ringing = underdamped oscillation with undershoot (g crosses below zero).
    """
    g = simulate_dde_rk4(K, Delta, T_max, dt)

    # Check for instability
    if np.any(np.isnan(g)) or np.max(np.abs(g)) > 100.0:
        return False

    # Find peak (initial overshoot)
    peak_idx = np.argmax(g[:int(5.0/dt)])  # Look in first 5 seconds

    # Check if there's a subsequent undershoot (g goes negative)
    if peak_idx < len(g) - 100:
        post_peak = g[peak_idx:]
        if np.min(post_peak) < -0.001:  # Clear undershoot
            return True

    return False

def find_sim_threshold_improved(Delta, tolerance=0.005):
    """
    Binary search for ringing threshold with improved detection.
    """
    # Search range
    low = 0.0
    high = B - 0.005  # Stay safely below instability

    # Check bounds
    if check_ringing_improved(low, Delta):
        return 0.0
    if not check_ringing_improved(high, Delta):
        return None

    # Binary search
    for iteration in range(20):  # Higher precision
        mid = (low + high) / 2.0

        if check_ringing_improved(mid, Delta):
            high = mid  # Ringing starts at lower K
        else:
            low = mid  # Still overdamped

        if (high - low) < tolerance:
            break

    return (low + high) / 2.0

# --- 3. Validation Run ---

def run_full_validation():
    """
    Run complete validation comparing analytical and simulation thresholds.
    """
    print("=" * 70)
    print("RFO VALIDATION: Analytical Ring Threshold vs DDE Simulation")
    print("=" * 70)
    print()

    # Delay range - adjusted to where ringing exists
    # From K-Delta scan: ringing emerges at Delta ≈ 0.104 s
    delays = np.linspace(0.12, 0.30, 15)

    results = []

    print(f"{'Delta [s]':<12} | {'K_analytic':<12} | {'K_sim':<12} | {'Error %':<12}")
    print("-" * 60)

    for i, delta in enumerate(delays):
        # Analytical threshold
        k_analytic = find_analytic_threshold(delta)

        # Simulation threshold
        k_sim = find_sim_threshold_improved(delta)

        if k_analytic is not None and k_sim is not None:
            error = abs(k_analytic - k_sim) / k_analytic * 100.0
            results.append({
                'delta': delta,
                'k_analytic': k_analytic,
                'k_sim': k_sim,
                'error': error
            })
            print(f"{delta:<12.4f} | {k_analytic:<12.6f} | {k_sim:<12.6f} | {error:<12.4f}")
        else:
            print(f"{delta:<12.4f} | {'---':<12} | {'---':<12} | {'---':<12}")

        # Progress indicator
        if (i+1) % 5 == 0:
            print(f"  [{i+1}/{len(delays)} complete]")

    print("-" * 60)
    print()

    # Statistics
    if len(results) > 0:
        errors = [r['error'] for r in results]
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        print("=" * 70)
        print("RESULTS FOR LATEX")
        print("=" * 70)
        print(f"Mean Relative Error: {mean_error:.2f}%")
        print(f"Max Relative Error:  {max_error:.2f}%")
        print(f"Delay Range:         [{delays[0]:.2f}, {delays[-1]:.2f}] s")
        print(f"Valid comparisons:   {len(results)}/{len(delays)}")
        print()

        # LaTeX sentence
        print("LATEX ABSTRACT SENTENCE:")
        print("-" * 70)
        print(f"Numerical validation of the full DDE demonstrates that the ")
        print(f"discriminant-based Ring Threshold predicts the onset of underdamped ")
        print(f"behavior with $\\bar{{\\varepsilon}} = {mean_error:.2f}\\%$ mean relative error ")
        print(f"and $\\varepsilon_{{\\max}} = {max_error:.2f}\\%$ across delays ")
        print(f"$\\Delta \\in [{delays[0]:.2f}, {delays[-1]:.2f}]~\\text{{s}}$.")
        print()

        return results, mean_error, max_error
    else:
        print("WARNING: No valid comparisons found")
        return [], None, None

if __name__ == "__main__":
    run_full_validation()
