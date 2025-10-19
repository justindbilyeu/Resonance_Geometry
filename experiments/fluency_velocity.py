from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.signal import hilbert, savgol_filter


def _to_series_and_phases(x) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Accepts:
      - x: 1D array [T] or 2D array [T, N]
    Returns:
      - series y[t]: mean-centered observable over nodes (if 2D: node-avg)
      - phases θ[t, i] (if 2D) extracted via Hilbert per node; else None
    """

    a = np.asarray(x)
    if a.ndim == 1:
        y = a - np.mean(a)
        return y, None
    if a.ndim == 2:
        # mean over nodes as observable
        y = a.mean(axis=1)
        # per-node analytic signal → phase
        theta = []
        for i in range(a.shape[1]):
            sig = a[:, i] - np.mean(a[:, i])
            z = hilbert(sig)
            theta.append(np.angle(z))
        theta = np.stack(theta, axis=1)  # [T, N]
        return y, theta
    raise ValueError("Input must be 1D [T] or 2D [T, N] array.")


def coherence_series(x) -> np.ndarray:
    """
    Kuramoto-like order parameter magnitude:
      Φ(t) = | (1/N) Σ_i exp(i θ_i(t)) |
    If only a scalar series is provided (1D), we proxy coherence by
    normalized autocorrelation magnitude in a short window.
    """

    y, theta = _to_series_and_phases(x)
    T = len(y)
    if theta is None:
        # Scalar proxy: rolling normalized autocorr magnitude
        w = min(101, max(11, (T // 20) * 2 + 1))
        half = w // 2
        y0 = y - y.mean()
        lag = max(1, w // 10)
        phi = np.zeros(T)
        for t in range(T):
            start = max(0, t - half)
            end = min(T, t + half + 1)
            seg = y0[start:end]
            if len(seg) <= lag:
                phi[t] = 0.0
                continue
            seg0 = seg - seg.mean()
            local_lag = min(lag, len(seg0) - 1)
            if local_lag <= 0:
                phi[t] = 0.0
                continue
            num = np.sum(seg0[local_lag:] * seg0[:-local_lag])
            den = np.sum(seg0**2) + 1e-12
            phi[t] = np.clip(abs(num / den), 0.0, 1.0)
        return phi
    # Proper multi-node coherence
    z = np.exp(1j * theta)  # [T, N]
    r = np.abs(np.mean(z, axis=1))  # [T]
    return r


def fluency_velocity(phi: np.ndarray, smooth: bool = True) -> np.ndarray:
    """
    v_f(t) = dΦ/dt with optional Savitzky–Golay smoothing for robustness.
    """

    phi = np.asarray(phi).astype(float)
    if smooth and len(phi) >= 11:
        k = (len(phi) // 25) * 2 + 1
        k = np.clip(k, 11, 101)  # odd window
        phi_s = savgol_filter(phi, window_length=k, polyorder=2, mode="interp")
    else:
        phi_s = phi
    vf = np.gradient(phi_s)
    return vf


def relaxation_velocity(phi: np.ndarray, t0: int, eq_window: int = 50) -> float:
    """
    After a perturbation at index t0, measure relaxation rate:
      v_relax = -mean_t d/dt log(|Φ(t) - Φ_eq| + eps), t ≥ t0
    """

    phi = np.asarray(phi).astype(float)
    T = len(phi)
    t0 = int(np.clip(t0, 0, T - 2))
    tail = phi[max(T - eq_window, 0) :]
    phi_eq = float(np.mean(tail)) if len(tail) > 0 else float(phi[-1])
    post = phi[t0:]
    d = np.abs(post - phi_eq) + 1e-8
    logd = np.log(d)
    t_idx = np.arange(len(logd), dtype=float)
    # focus on region above numerical floor
    thresh = 1e-6 * np.max(d)
    valid = (d > thresh) & np.isfinite(logd)
    if valid.sum() < 2:
        return 0.0
    slope, _ = np.polyfit(t_idx[valid], logd[valid], 1)
    return float(-slope)


def summary_metrics(phi: np.ndarray, vf: np.ndarray) -> Dict[str, float]:
    return {
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "vf_mean": float(np.mean(vf)),
        "vf_std": float(np.std(vf)),
        "vf_p95": float(np.percentile(np.abs(vf), 95)),
    }
