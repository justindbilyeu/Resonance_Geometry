# experiments/fluency_velocity.py
from __future__ import annotations
import numpy as np

# ---------- core coherence helpers ----------

def phase_coherence(phases: np.ndarray) -> np.ndarray:
    """
    Φ(t) for phase array shaped (T, N) in radians.
    Returns Φ series length T in [0,1].
    Φ(t) = |(1/N) Σ_i e^{i θ_i(t)}|
    """
    phases = np.asarray(phases, float)
    if phases.ndim != 2:
        raise ValueError("phases must be (T, N)")
    T, N = phases.shape
    z = np.exp(1j * phases)  # (T,N)
    order = np.abs(np.mean(z, axis=1))  # (T,)
    return order

def coherence_velocity(phases: np.ndarray) -> np.ndarray:
    """
    v_coh(t) = dΦ/dt (finite difference).
    """
    phi = phase_coherence(phases)
    return np.gradient(phi)

# ---------- relaxation velocity (post-perturb) ----------

def relaxation_velocity(
    phases: np.ndarray,
    perturb_start_idx: int,
    eq_window: int = 100,
    eps: float = 1e-8
) -> float:
    """
    v_relax ≈ - mean d/dt log |Φ(t) - Φ_eq| over post-perturb segment.
    phases: (T,N) radians
    perturb_start_idx: index t0 where perturbation begins (0-based)
    eq_window: how many final points to average for Φ_eq
    returns a scalar (1/steps); larger -> faster relaxation.
    """
    phi = phase_coherence(phases)
    T = len(phi)
    if T < max(perturb_start_idx+5, eq_window+5):
        raise ValueError("trajectory too short for chosen windows")

    phi_eq = float(np.mean(phi[-eq_window:]))
    post = phi[perturb_start_idx:]
    dist = np.log(np.abs(post - phi_eq) + eps)
    # Numerical gradient can be noisy; average a central band
    g = np.gradient(dist)
    return float(-np.mean(g))

# ---------- phase-space velocity ----------

def phase_space_velocity(omega_history: np.ndarray) -> float:
    """
    v_phase = mean || dω/dt || for ω(t) in R^D (history shaped (T,D)).
    This treats 'omega' as your system's state vector time-series.
    """
    X = np.asarray(omega_history, float)
    if X.ndim != 2:
        raise ValueError("omega_history must be (T, D)")
    dX = np.gradient(X, axis=0)
    speed = np.linalg.norm(dX, axis=1)  # (T,)
    return float(np.mean(speed))

# ---------- convenience: all metrics ----------

def compute_fluency_metrics(
    phases: np.ndarray,
    omega_history: np.ndarray | None = None,
    perturb_start_idx: int | None = None
) -> dict:
    """
    Returns:
      {
        'phi_mean': ..., 'phi_std': ...,
        'v_coh_mean': ..., 'v_coh_std': ...,
        'v_relax': (or None),
        'v_phase': (or None)
      }
    """
    phi = phase_coherence(phases)
    vcoh = np.gradient(phi)
    out = {
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "v_coh_mean": float(np.mean(vcoh)),
        "v_coh_std": float(np.std(vcoh)),
        "v_relax": None,
        "v_phase": None,
    }
    if perturb_start_idx is not None:
        out["v_relax"] = relaxation_velocity(phases, perturb_start_idx)

    if omega_history is not None:
        out["v_phase"] = phase_space_velocity(omega_history)

    return out

# ---------- tiny synthetic demo (optional) ----------

def _synthetic_phases(T=1000, N=16, noise=0.05, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-np.pi, np.pi, size=N)
    w = rng.normal(0, 0.02, size=N) + drift
    phases = base + np.outer(np.arange(T), w)
    phases += rng.normal(0, noise, size=(T, N))
    return (phases + np.pi) % (2*np.pi) - np.pi
