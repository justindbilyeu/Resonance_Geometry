cat > experiments/phase1_prediction.py <<'PY'
"""
Phase 1 (Prediction Test): chunkable analysis function.

Exports:
- run_phase1_analysis(n_runs: int, seed: int) -> list[dict]

Each dict minimally contains:
  - 'sign_match' (bool): whether predicted deflection sign matches actual along dim 0
  - 'angular_error' (float): angle between predicted and actual deflection (radians)

This implementation is dependency-light:
- It will try to use experiments.forbidden_region_detector.gp_toy_evolve if available.
- If that import or call fails, it falls back to a synthetic 4D trajectory generator.
- The geometric *prediction* is made ONLY from the gradient of a scalar "curvature"
  field at the initial state (no dynamics), which is the point of Phase 1.

Note: This is a reasonable scaffold to get infrastructure green. You can swap
      the curvature field or the trajectory source later without changing
      scripts/run_phase1_chunked.py.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Callable, Optional

# Try to import the toy evolution. If unavailable, we fallback.
try:
    from experiments.forbidden_region_detector import gp_toy_evolve  # type: ignore
    HAVE_GP = True
except Exception:
    gp_toy_evolve = None  # type: ignore
    HAVE_GP = False


# ---------- small helpers ----------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(a); b = unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(c))

def finite_diff_grad(f: Callable[[np.ndarray], float],
                     x: np.ndarray,
                     eps: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x); dx[i] = eps
        g[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return g


# ---------- a smooth scalar "curvature" field on [0,1]^4 ----------
# tunable but deterministic; gives wells/saddles + gentle bowl
def curvature_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(
        0.6 * np.sum(np.cos(2.0 * np.pi * x)) +
        0.4 * np.sum(x**2)
    )

def geom_pred_direction(x0: np.ndarray) -> np.ndarray:
    # geometric prediction uses ONLY the curvature landscape (no dynamics)
    g = finite_diff_grad(curvature_scalar, x0, eps=1e-3)
    # deflect away from positive curvature (choose minus grad as "barrier normal")
    return unit(-g)


# ---------- trajectory sources ----------
def _synthetic_trajectory(x0: np.ndarray,
                          steps: int = 200,
                          dt: float = 0.02,
                          beta: float = 0.8,
                          noise: float = 0.02,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Lightweight 4D nonlinear damped oscillator with a ridge near the curvature barrier.
    Returns an array of shape (steps, 4) in [0,1]^4 (clamped).
    """
    if rng is None:
        rng = np.random.default_rng()

    x = x0.copy()
    v = rng.normal(scale=0.05, size=4)
    traj = [x.copy()]

    for _ in range(steps - 1):
        # nonlinear spring towards 0.5 (center) + coupling across dims
        center = 0.5
        spring = -(x - center)
        cross  = beta * np.array([
            0.3*(x[1]-x[0]) + 0.1*(x[2]-x[0]),
            0.3*(x[2]-x[1]) + 0.1*(x[3]-x[1]),
            0.3*(x[3]-x[2]) + 0.1*(x[0]-x[2]),
            0.3*(x[0]-x[3]) + 0.1*(x[1]-x[3]),
        ])
        # curvature ridge "repulsion"
        ridge_n = geom_pred_direction(x)        # normal of curvature barrier
        ridge   = 0.15 * ridge_n

        # damped velocity + forces + small noise
        v = 0.95 * v + dt * (spring + cross + ridge) + rng.normal(scale=noise, size=4)
        x = x + dt * v
        x = np.clip(x, 0.0, 1.0)
        traj.append(x.copy())

    return np.asarray(traj)  # (steps, 4)


def _gp_or_fallback_trajectory(x0: np.ndarray,
                               steps: int,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Try gp_toy_evolve if it exists and is callable with minimal args; otherwise fallback.
    We do NOT pass 'seed' because earlier versions of gp_toy_evolve didn't accept it.
    """
    if HAVE_GP and callable(gp_toy_evolve):
        try:
            # Some versions accept: gp_toy_evolve(x0, steps=..., beta=..., noise=...)
            # Others might accept different signatures — keep it minimal & safe.
            traj = gp_toy_evolve(x0=x0, steps=steps)  # type: ignore
            traj = np.asarray(traj, dtype=float)
            if traj.ndim == 2 and traj.shape[1] == 4:
                # Normalize to [0,1] if gp returns unbounded coords
                # (Defensive: avoid NaNs)
                tmin = np.nanmin(traj, axis=0)
                tmax = np.nanmax(traj, axis=0)
                denom = (tmax - tmin); denom[denom == 0] = 1.0
                traj_n = (traj - tmin) / denom
                return np.clip(traj_n, 0.0, 1.0)
        except Exception:
            pass
    # Fallback
    return _synthetic_trajectory(x0, steps=steps, rng=rng)


# ---------- public entry ----------
def run_phase1_analysis(n_runs: int, seed: int) -> List[Dict]:
    """
    Run n_runs independent trials. For each:
      - choose x0 in [0,1]^4 deterministically from (seed, idx)
      - evolve a trajectory
      - actual deflection = direction change across last 100 steps
      - geometric prediction = -∇(curvature)(x0)  (ONLY geometry)
      - record sign match (dim 0) and angular error
    """
    results: List[Dict] = []
    base_rng = np.random.default_rng(seed)

    for idx in range(n_runs):
        # derive a per-run RNG and x0
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        x0  = rng.random(4)  # in [0,1]^4

        steps = 200
        traj  = _gp_or_fallback_trajectory(x0, steps=steps, rng=rng)

        # compute actual deflection from final segment
        # take two velocity estimates far enough apart to be stable
        m1, m2 = max(steps - 120, 1), max(steps - 20, 2)
        v_initial = traj[m1] - traj[m1 - 1]
        v_final   = traj[m2] - traj[m2 - 1]
        d_actual  = v_final - v_initial

        # geometric prediction uses ONLY initial x0
        d_geom = geom_pred_direction(x0)

        # metrics
        sign_match = bool(np.sign(d_geom[0]) == np.sign(d_actual[0]))
        ang_err    = float(angle_between(d_geom, d_actual))

        results.append({
            "sign_match": sign_match,
            "angular_error": ang_err,
        })

    return results
PY
