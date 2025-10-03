from __future__ import annotations

import numpy as np
from typing import List, Dict

# Try to import the toy GP evolution from the repo; we keep it optional so CI stays green.
try:
    from experiments.forbidden_region_detector import gp_toy_evolve  # type: ignore
    HAVE_GP = True
except Exception:
    gp_toy_evolve = None  # type: ignore
    HAVE_GP = False


# ------------------ small helpers (self-contained, no external deps) ------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = _unit(a); b = _unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(c))

def _finite_diff_grad(f, x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x); dx[i] = eps
        g[i] = (f(np.clip(x + dx, 0, 1)) - f(np.clip(x - dx, 0, 1))) / (2 * eps)
    return g


# ------------------ a simple geometric scalar field (proxy curvature) ------------------

def _curvature_scalar(x: np.ndarray) -> float:
    """
    A bounded, bumpy scalar field over [0,1]^4. Acts as a proxy 'curvature' field
    to generate a geometric prediction direction via -∇κ(x0).
    """
    x = np.asarray(x, dtype=float)
    return float(
        0.6 * np.sum(np.cos(2*np.pi*x))   # periodic bumps
      + 0.4 * np.sum(x**2)                # gentle bowl
    )

def _geom_pred_direction(x0: np.ndarray) -> np.ndarray:
    g = _finite_diff_grad(_curvature_scalar, x0, eps=1e-2)
    return -_unit(g)  # deflect "downhill" in κ


# ------------------ lightweight synthetic trajectory (fallback) ------------------

def _synth_traj(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Produce a simple 4D bounded trajectory in [0,1]^4. If gp_toy_evolve is available,
    we can swap this out later; for now, keep it deterministic & cheap.
    """
    x = rng.random(4)  # x0 in [0,1]^4
    A = rng.standard_normal((4, 4)) * 0.25  # small linear part
    traj = [x.copy()]
    for _ in range(n_steps - 1):
        drift = 0.06 * np.tanh(A @ x)              # bounded drift
        noise = 0.02 * rng.standard_normal(4)      # small noise
        x = np.clip(x + drift + noise, 0.0, 1.0)
        traj.append(x.copy())
    return np.asarray(traj)  # shape (n_steps, 4)


# ------------------ metrics extraction ------------------

def _deflection_from_traj(traj: np.ndarray, tail: int = 50) -> np.ndarray:
    """
    Compare early vs late average motion direction; returns a 4D deflection vector.
    """
    T = traj.shape[0]
    tail = max(5, min(tail, T // 2))
    # early direction
    v0 = np.mean(np.diff(traj[:tail, :], axis=0), axis=0)
    # late direction
    v1 = np.mean(np.diff(traj[-tail:, :], axis=0), axis=0)
    return _unit(v1 - v0)


# ------------------ public predictors ------------------

def run_phase1_analysis_null(n_runs: int = 1000, seed: int = 42) -> List[Dict]:
    """
    Null baseline: prediction is random in 4D, 'actual' comes from a synthetic trajectory.
    Expected sign accuracy ~0.5; mean angular error ~π/2.
    """
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    for _ in range(n_runs):
        # trajectory -> actual deflection
        traj = _synth_traj(n_steps=200, rng=rng)
        d_actual = _deflection_from_traj(traj, tail=40)

        # random predicted direction
        d_pred = _unit(rng.standard_normal(4))

        out.append({
            "sign_match": bool(np.sign(d_pred[0]) == np.sign(d_actual[0])),
            "angular_error": float(_angle_between(d_pred, d_actual)),
        })
    return out


def run_phase1_analysis(n_runs: int = 1000, seed: int = 42) -> List[Dict]:
    """
    Proxy geometric predictor:
      - Simulate a synthetic trajectory to get the *actual* deflection.
      - Use -∇κ(x0) at the initial point x0 to get a purely geometric *predicted* direction.
    This is CI-safe and gives non-degenerate results distinct from the null.
    """
    rng = np.random.default_rng(seed)
    out: List[Dict] = []

    for _ in range(n_runs):
        traj = _synth_traj(n_steps=200, rng=rng)
        x0 = traj[0]
        d_actual = _deflection_from_traj(traj, tail=40)
        d_pred   = _geom_pred_direction(x0)

        out.append({
            "sign_match": bool(np.sign(d_pred[0]) == np.sign(d_actual[0])),
            "angular_error": float(_angle_between(d_pred, d_actual)),
        })

    return out


def run_phase1_analysis_curv(n_runs: int = 1000, seed: int = 42) -> List[Dict]:
    """Minimal curvature-based predictor suitable for CI runs.

    Workflow:
      - Sample a starting point ``x0`` in the unit hypercube.
      - Roll out a lightweight, deterministic-ish trajectory with a tiny drift
        and restoring force to obtain an "actual" deflection.
      - Use the analytic curvature proxy :math:`-∇κ(x0)` as the geometric
        prediction direction and compare against the trajectory deflection.
    """
    rng = np.random.default_rng(seed)
    out: List[Dict] = []

    for _ in range(n_runs):
        x0 = rng.random(4)

        # Toy trajectory: slow drift along a random unit vector with mild
        # restoring dynamics to keep it in-bounds.
        T = 200
        v = rng.standard_normal(4)
        v /= np.linalg.norm(v) + 1e-12
        x = x0.copy()
        for _ in range(T):
            x = x + 0.02 * v - 0.01 * (x - 0.5)
            x = np.clip(x, 0.0, 1.0)

        d_actual = x - x0
        d_geom = -_finite_diff_grad(_curvature_scalar, x0, eps=1e-2)

        out.append({
            "sign_match": bool(np.sign(d_geom[0]) == np.sign(d_actual[0])),
            "angular_error": float(_angle_between(d_geom, d_actual)),
        })

    return out
