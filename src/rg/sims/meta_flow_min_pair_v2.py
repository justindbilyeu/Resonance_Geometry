"""Deprecated stub for meta flow minimal pair simulation.

Use `resonance_geometry.hallucination.phase_dynamics` instead.
"""

from resonance_geometry.hallucination.phase_dynamics import *  # noqa: F401,F403

if __name__ == "__main__":
    from resonance_geometry.hallucination.phase_dynamics import simulate_trajectory
    import numpy as np

    params = {
        "lambda": 1.0,
        "gamma": 0.5,
        "k": 1.0,
        "alpha": 0.6,
        "beta": 0.02,
        "skew": 0.12,
        "mu": 0.0,
        "mi_window": 30,
        "mi_ema": 0.1,
        "omega_anchor": np.zeros(3),
        "eta": 2.0,
    }
    trajectory = simulate_trajectory(params)
    print("Final norm:", trajectory["norm"][-1])
