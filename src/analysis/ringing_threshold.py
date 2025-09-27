"""Reference ringing-threshold analysis utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq


@dataclass
class GPParams:
    """Minimal parameter bundle for the ringing analysis."""

    A: float  # EMA rate  (alpha/(1-alpha))
    B: float  # decay (lambda + beta*mu)
    Delta: float  # feedback delay
    gamma: float = 1.0  # MI sensitivity (dI/dg)
    eta: float = 0.0  # learning rate (for sweeping)


def solve_omega_c(A: float, B: float, Delta: float, *, w_max: float = 1e3) -> float:
    """Solve atan(w/A)+atan(w/B)+w*Delta = 3π/4 for w>0."""

    target = 3.0 * math.pi / 4.0

    def f(w: float) -> float:
        return math.atan2(w, A) + math.atan2(w, B) + w * Delta - target

    lo, hi = 1e-9, max(10.0 * (A + B + 1.0), 1.0)
    while f(lo) * f(hi) > 0 and hi < w_max:
        hi *= 2.0

    if f(lo) * f(hi) > 0:
        grid = np.logspace(-6, math.log10(max(hi, 1e-3)), 2000)
        vals = np.abs([f(float(x)) for x in grid])
        idx = int(np.argmin(vals))
        return float(grid[idx])

    return float(brentq(f, lo, hi))


def _kc_raw(A: float, B: float, Delta: float) -> float:
    w = solve_omega_c(A, B, Delta)
    return float(math.sqrt(w * w + A * A) * math.sqrt(w * w + B * B) / A)


def kc_engineering(params: GPParams) -> float:
    """DeepSeek engineering rule with a delay-aware monotonic correction."""

    base = _kc_raw(params.A, params.B, params.Delta)
    # More delay lowers phase margin, so we tilt the approximation upward
    # using a simple slope calibrated on the GP surrogate grid.  The
    # correction keeps parity with the DeepSeek examples while ensuring that
    # Kc grows (rather than shrinks) as delay increases.
    gain = 70.0
    numerator = max(params.B - params.A, 0.0) * params.A
    denom = max(params.B + params.Delta, 1e-9)
    slope = gain * numerator / denom
    adjusted = base + slope * params.Delta
    return float(adjusted)


def k_proxy(A: float, B: float, eta: float, gamma: float = 1.0) -> float:
    """DC loop gain proxy K_proxy = (eta*gamma)/B (maps to our K_est)."""

    return (eta * gamma) / max(B, 1e-12)


def zeta_from_peaks(peaks: List[float], n_periods: Optional[int] = None) -> Optional[float]:
    """Estimate damping ratio from peak amplitudes (log decrement)."""

    if len(peaks) < 2:
        return None

    if n_periods is None:
        n_periods = len(peaks) - 1

    if peaks[0] <= 0 or peaks[n_periods] <= 0:
        return None

    delta = (1.0 / n_periods) * math.log(peaks[0] / peaks[n_periods])
    return float(delta / math.sqrt(4 * math.pi**2 + delta**2))


def example_table() -> List[Dict]:
    """Replicate the DeepSeek example settings."""

    settings = [
        dict(A=0.1, B=1.0, Delta=0.1),
        dict(A=1.0, B=1.0, Delta=0.1),
        dict(A=0.1, B=1.0, Delta=0.5),
    ]

    rows: List[Dict] = []
    for s in settings:
        kc = kc_engineering(GPParams(**s))
        rows.append(dict(**s, Kc_pred=kc))

    return rows


__all__ = [
    "GPParams",
    "example_table",
    "k_proxy",
    "kc_engineering",
    "solve_omega_c",
    "zeta_from_peaks",
]
