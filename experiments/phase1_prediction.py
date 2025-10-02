experiments/phase1_prediction.py <<'PY'
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List


# -------------------- small helpers --------------------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(a); b = unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(c))

def finite_diff_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x); dx[i] = eps
        g[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return g


# -------------------- a proxy "curvature" field --------------------
# Smooth, periodic bumps in [0,1]^4 + bowl so the gradient isn't degenerate.
def curvature_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(
        0.6 * np.sum(np.cos(2 * np.pi * x)) +   # structured bumps
        0.4 * np.sum((x - 0.5) ** 2)            # weak quadratic bowl
    )

def geom_pred_direction(x0: np.ndarray) -> np.ndarray:
    # Predict "deflection" as downhill direction of curvature (negative gradient).
    g = finite_diff_grad(curvature_scalar, x0)
    return -unit(g)


# -------------------- a light surrogate trajectory --------------------
# To keep this CI/pilot-friendly (no heavy GP), we simulate a damped driven walk
# in [0,1]^4 with reflecting boundaries. This gives us an "actual" deflection.
def simulate_surrogate_trajectory(x0: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    x = x0.astype(float).copy()
    v = np.zeros_like(x)
    traj = [x.copy()]
    gamma = 0.08      # damping
    amp   = 0.04      # drive amplitude
    noise = 0.02
    for t in range(steps):
        drive = amp * np.sin(2 * np.pi * (t / 40.0))  # slow oscillation
        v = (1 - gamma) * v + drive * geom_pred_direction(x) + noise * rng.normal(size=x.shape)
        x = x + v
        # reflect at boundaries to keep inside [0,1]^4
        for d in range(x.size):
            if x[d] < 0:
                x[d] = -x[d]; v[d] = -v[d]
            elif x[d] > 1:
                x[d] = 2 - x[d]; v[d] = -v[d]
        traj.append(x.copy())
    return np.stack(traj, axis=0)  # shape (steps+1, 4)


# -------------------- analysis functions used by the runner --------------------
def _per_run_metrics(x0: np.ndarray, rng: np.random.Generator, steps: int = 200) -> Dict:
    traj = simulate_surrogate_trajectory(x0, steps=steps, rng=rng)

    # "Actual" deflection: compare early vs late velocity directions
    # Use short windows for robustness.
    w = min(20, len(traj)-1)
    v_start = unit(traj[w] - traj[0])
    v_end   = unit(traj[-1] - traj[-1-w])

    d_actual = v_end
    d_geom   = geom_pred_direction(x0)

    return {
        "sign_match": bool(np.sign(d_geom[0]) == np.sign(d_actual[0])),
        "angular_error": float(angle_between(d_geom, d_actual)),
    }


def run_phase1_analysis(n_runs: int, seed: int) -> List[Dict]:
    """
    Geometry ON: prediction uses curvature gradient; dynamics use surrogate.
    """
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    for _ in range(n_runs):
        x0 = rng.uniform(0, 1, size=4)
        out.append(_per_run_metrics(x0, rng))
    return out


def run_phase1_analysis_null(n_runs: int, seed: int) -> List[Dict]:
    """
    Null baseline: geometry OFF. Prediction is random unit vector.
    """
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    for _ in range(n_runs):
        x0 = rng.uniform(0, 1, size=4)
        # actual from surrogate trajectory
        traj = simulate_surrogate_trajectory(x0, steps=200, rng=rng)
        w = min(20, len(traj)-1)
        v_end = unit(traj[-1] - traj[-1-w])
        # random prediction
        rand = unit(rng.normal(size=4))
        out.append({
            "sign_match": bool(np.sign(rand[0]) == np.sign(v_end[0])),
            "angular_error": float(angle_between(rand, v_end)),
        })
    return out
PY
