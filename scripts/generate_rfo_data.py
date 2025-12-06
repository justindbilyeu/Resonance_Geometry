import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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
    # Standard formula: 18abcd - 4b^3d + b^2c^2 - 4ac^3 - 27a^2d^2
    # Map a3->a, a2->b, a1->c, a0->d
    a, b, c, d = a3, a2, a1, a0
    disc = (18*a*b*c*d) - (4*(b**3)*d) + ((b**2)*(c**2)) - (4*a*(c**3)) - (27*(a**2)*(d**2))
    return disc

def find_analytic_threshold(Delta):
    """Finds K_c where Discriminant = 0 (using root finding)."""
    # Search between K=0 and K=B (stable region)
    # If disc(0) and disc(B) have same sign, simplistic bisection fails,
    # but we know Disc > 0 at K=0 (overdamped) and Disc usually < 0 near K=B.
    try:
        root = brentq(lambda k: get_cubic_discriminant(k, Delta), 0.0, B - 1e-4)
        return root
    except ValueError:
        return None # No crossing found (always overdamped or always ringing)

# --- 2. Numerical Simulation (Ground Truth) ---

def simulate_impulse_response(K, Delta, T_max=10.0, dt=0.001):
    """
    Simulates the FULL DDE: g'' + (A+B)g' + ABg = AK g(t-Delta)
    Returns: time array, g array
    """
    # State: [g, g_dot]
    # History buffer for g
    steps = int(T_max / dt)
    delay_steps = int(Delta / dt)

    g = np.zeros(steps)
    g_dot = np.zeros(steps)

    # Impulse Initial Condition: g(0)=1, g'(0)=0 (or similar approximation)
    g[0] = 1.0

    for i in range(steps - 1):
        # Delayed term
        if i >= delay_steps:
            g_delayed = g[i - delay_steps]
        else:
            g_delayed = 0.0 # Pre-history is zero

        # DDE: g'' = - (A+B)g' - AB g + AK g_delayed
        acc = -(A + B)*g_dot[i] - (A * B)*g[i] + (A * K)*g_delayed

        # Euler integration (sufficient for threshold detection)
        g_dot[i+1] = g_dot[i] + acc * dt
        g[i+1] = g[i] + g_dot[i] * dt

    return g

def check_ringing_sim(K, Delta):
    """Returns True if simulation shows valid ringing (undershoot)."""
    # We define ringing as: having a zero crossing (g < 0 at some point)
    # AND not being unstable (g doesn't explode).
    g = simulate_impulse_response(K, Delta)

    if np.max(np.abs(g)) > 100.0:
        return False # Unstable/Explosion (technically ringing, but DC fail)

    if np.min(g) < -1e-4: # Tolerance for zero crossing
        return True
    return False

def find_sim_threshold(Delta):
    """Binary search for the onset of ringing in the simulation."""
    low = 0.0
    high = B - 0.01

    # Check bounds
    if check_ringing_sim(low, Delta): return 0.0 # Always rings
    if not check_ringing_sim(high, Delta): return None # Never rings

    for _ in range(15): # Precision ~ 1/2^15
        mid = (low + high) / 2
        if check_ringing_sim(mid, Delta):
            high = mid # Ringing starts lower
        else:
            low = mid # Still overdamped

    return (low + high) / 2

# --- 3. Main Execution & Plotting ---

def run_validation():
    print("--- Running Resonance Geometry Validation ---")

    # Based on K-Delta scan: ringing exists for Delta >= 0.104 s
    # Use range where Ring Threshold exists
    delays = np.linspace(0.10, 0.40, 15)
    errors = []

    print(f"{'Delta [s]':<10} | {'K_analytic':<10} | {'K_sim':<10} | {'Error %':<10}")
    print("-" * 50)

    for d in delays:
        k_a = find_analytic_threshold(d)
        k_s = find_sim_threshold(d)

        if k_a is not None and k_s is not None:
            err = abs(k_a - k_s) / k_a * 100.0
            errors.append(err)
            print(f"{d:<10.3f} | {k_a:<10.4f} | {k_s:<10.4f} | {err:<10.2f}")
        else:
            print(f"{d:<10.3f} | {'---':<10} | {'---':<10} | {'---'}")

    print("-" * 50)

    if len(errors) > 0:
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        print(f"RESULTS FOR LATEX:")
        print(f"Mean Relative Error: {mean_err:.2f}%")
        print(f"Max Relative Error:  {max_err:.2f}%")
        print(f"Range of Delays:     [{delays[0]:.2f}, {delays[-1]:.2f}] s")
        print(f"Valid comparisons: {len(errors)}/{len(delays)}")
    else:
        print("WARNING: No valid threshold comparisons found")
        print("This may indicate Ring Threshold doesn't exist in tested range")

    # --- Generate Phase Map Data (Placeholder for full plot script) ---
    # This just proves the 'Wedge' exists
    k_wedge = find_analytic_threshold(0.15)
    if k_wedge is not None:
        print(f"\nAt Delta=0.15s, the Wedge exists for K in [{k_wedge:.3f}, {B:.3f}]")
    else:
        print(f"\nAt Delta=0.15s, no Ring Threshold found (always overdamped)")

if __name__ == "__main__":
    run_validation()
