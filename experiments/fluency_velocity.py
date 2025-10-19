from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def _coherence_kuramoto(phases_row: np.ndarray) -> float:
    z = np.exp(1j * phases_row)
    return float(np.abs(np.mean(z)))


def coherence_series(states: np.ndarray) -> np.ndarray:
    """Return the Kuramoto-style coherence ``Φ(t)`` for ``states`` shaped ``(T, N)``."""

    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise AssertionError("states must be (T, N)")

    T, N = states.shape
    phases = np.empty_like(states)
    for j in range(N):
        phases[:, j] = np.angle(hilbert(states[:, j]))

    Phi = np.empty(T, dtype=float)
    for t in range(T):
        Phi[t] = _coherence_kuramoto(phases[t, :])
    return Phi


def fluency_velocity(states: np.ndarray, smooth: int = 5) -> dict:
    """Compute coherence ``Φ`` and velocity summary statistics from ``states``."""

    Phi = coherence_series(states)

    if smooth and smooth > 1:
        k = np.ones(smooth, dtype=float) / smooth
        Phi = np.convolve(Phi, k, mode="same")

    v_f = np.gradient(Phi)

    return {
        "phi_mean": float(np.mean(Phi)),
        "vf_mean": float(np.mean(v_f)),
        "vf_std": float(np.std(v_f)),
    }
