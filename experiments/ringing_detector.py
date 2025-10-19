#!/usr/bin/env python3
"""
Amplitude-independent ringing detector using relative peak prominence.

Returns a dict:
  {
    "ringing": bool,
    "n_peaks": int,
    "peak_ratio": float,     # mean(prominence)/noise
    "overshoot_z": float     # max(|x|)/sigma(noise)
  }
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation as mad


def detect_ringing(
    series: np.ndarray,
    peak_factor: float = 60.0,
    overshoot_sigma: float = 6.0,
    min_peaks: int = 3,
) -> dict:
    x = np.asarray(series, dtype=float).ravel()
    if x.size < 10:
        return {"ringing": False, "n_peaks": 0, "peak_ratio": 0.0, "overshoot_z": 0.0}

    # Robust noise scale from MAD (guard zero)
    noise = float(mad(x))
    if noise <= 0:
        noise = 1e-12
    sigma = 1.4826 * noise

    # Prominence threshold tied to noise
    peaks_pos, props_pos = find_peaks(x, prominence=noise, distance=5)
    peaks_neg, props_neg = find_peaks(-x, prominence=noise, distance=5)

    n_peaks = int(peaks_pos.size + peaks_neg.size)
    prominences = np.concatenate(
        (props_pos.get("prominences", []), props_neg.get("prominences", []))
    ) if n_peaks > 0 else np.array([0.0])

    peak_ratio = float(np.mean(prominences) / noise) if noise > 0 else 0.0
    overshoot_z = float(np.max(np.abs(x)) / sigma) if sigma > 0 else 0.0

    ringing = (n_peaks >= min_peaks) and (peak_ratio >= peak_factor) and (overshoot_z >= overshoot_sigma)
    return {"ringing": bool(ringing), "n_peaks": n_peaks, "peak_ratio": peak_ratio, "overshoot_z": overshoot_z}
