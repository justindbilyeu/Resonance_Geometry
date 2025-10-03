# experiments/phase1_prediction.py
from __future__ import annotations
import numpy as np
from typing import List, Dict

# Try to import toy GP evolve; fall back to synthetic trajectory
try:
    from experiments.forbidden_region_detector import gp_toy_evolve
    HAVE_GP = True
except Exception:
    HAVE_GP = False
    gp_toy_evolve = None  # type: ignore

# -------- helpers --------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(a); b = unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(c))

def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x); dx[i] = eps
        g[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return g

# -------- a deterministic “curvature-like” scalar field (proxy) --------
def curvature_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(0.6 * np.sum(np.cos(2*np.pi*x)) + 0.4 * np.sum(x**2))

def geom_pred_direction(x0: np.ndarray) -> np.ndarray:
    g = -finite_diff_grad(curvature_scalar, x0, 1e-2)   # “move downhill”
    return unit(g)

# -------- synthetic trajectory if gp_toy_evolve is unavailable --------
def _synthetic_traj(seed: int, steps: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, 4), float)
    v = unit(rng.normal(size=4))
    for t in range(1, steps):
        x[t] = x[t-1] + 0.01 * v + 0.01 * rng.normal(size=4)
        # weak drift along grad of curvature
        v = unit(0.9*v + 0.1*geom_pred_direction(unit(x[t] % 1.0)))
    return unit(x % 1.0)

# -------- public analysis functions (used by the chunked runner) --------
def _compare_vectors(d_geom: np.ndarray, d_true: np.ndarray) -> Dict[str, float | bool]:
    ang = angle_between(d_geom, d_true)
    sign_ok = bool(np.sign(d_geom[0]) == np.sign(d_true[0]))
    return {"sign_match": sign_ok, "angular_error": float(ang)}

def run_phase1_analysis(n_runs: int, seed: int) -> List[Dict]:
    """
    Proxy geom predictor vs. empirical deflection from a trajectory.
    Returns list of dicts with 'sign_match' and 'angular_error'.
    """
    rng = np.random.default_rng(seed)
    results: List[Dict] = []
    for k in range(n_runs):
        s = int(rng.integers(0, 2**31-1))
        if HAVE_GP:
            # use the toy evolve from your repo
            traj = gp_toy_evolve(n_steps=600, seed=s)  # shape (T,4) in [0,1]^4
        else:
            traj = _synthetic_traj(s, steps=600)

        x0, x1, x2 = traj[0], traj[-2], traj[-1]
        d_true = unit(x2 - x1)             # last-step direction (empirical)
        d_geom = geom_pred_direction(x0)   # geometry-only prediction at start
        results.append(_compare_vectors(d_geom, d_true))
    return results

def run_phase1_analysis_null(n_runs: int, seed: int) -> List[Dict]:
    """
    Null baseline: random predictor vs. empirical deflection.
    Expected sign accuracy ≈ 0.5, angular error ≈ π/2.
    """
    rng = np.random.default_rng(seed)
    results: List[Dict] = []
    for k in range(n_runs):
        s = int(rng.integers(0, 2**31-1))
        traj = _synthetic_traj(s, steps=600)
        x1, x2 = traj[-2], traj[-1]
        d_true = unit(x2 - x1)
        d_rand = unit(rng.normal(size=4))
        results.append(_compare_vectors(d_rand, d_true))
    return results
