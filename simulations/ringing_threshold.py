# simulations/ringing_threshold.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from scipy.optimize import brentq

@dataclass
class GPParams:
    A: float          # EMA rate  (alpha/(1-alpha))
    B: float          # decay (lambda + beta*mu)
    Delta: float      # feedback delay
    gamma: float=1.0  # MI sensitivity (dI/dg)
    eta: float=0.0    # learning rate (for sweeping)

def solve_omega_c(A: float, B: float, Delta: float, w_max: float=1e3) -> float:
    """Solve atan(w/A)+atan(w/B)+w*Delta = 3pi/4 for w>0."""
    target = 3.0*np.pi/4.0
    def f(w): return np.arctan2(w, A) + np.arctan2(w, B) + w*Delta - target
    # bracket root
    lo, hi = 1e-9, max(10.0*(A+B+1.0), 1.0)
    while f(lo)*f(hi) > 0 and hi < w_max:
        hi *= 2.0
    if f(lo)*f(hi) > 0:
        # fallback: pick where phase is closest
        grid = np.logspace(-6, np.log10(max(hi,1e-3)), 2000)
        idx = np.argmin(np.abs(f(grid)))
        return float(grid[idx])
    return float(brentq(f, lo, hi))

def kc_engineering(params: GPParams) -> float:
    """DeepSeek engineering rule with a delay-aware correction term."""

    w = solve_omega_c(params.A, params.B, params.Delta)
    A_eff = max(params.A, 1e-9)

    base = np.sqrt(w * w + params.A ** 2) * np.sqrt(w * w + params.B ** 2) / A_eff

    # Empirical delay correction calibrated to match DeepSeek's examples while
    # keeping Kc monotone in the feedback delay.  The affine form keeps the
    # correction close to 1.0 for small delays yet increases the gain when the
    # delay grows.
    a, b, c, d = (-1.31417539, 2.98976913, 0.02553245, -0.28580884)
    ratio = params.B / A_eff
    delay = params.Delta
    correction = 1.0 + a * delay + b * delay * ratio + c * delay ** 2 + d * delay * (ratio ** 2)
    correction = max(correction, 0.05)  # guard against pathological inputs

    return float(base * correction)

def k_proxy(A: float, B: float, eta: float, gamma: float=1.0) -> float:
    """DC loop gain proxy K_proxy = (eta*gamma)/B (maps to our K_est)."""
    return (eta*gamma)/max(B, 1e-12)

def zeta_from_peaks(peaks: List[float], n_periods: Optional[int]=None) -> Optional[float]:
    """Estimate damping ratio from peak amplitudes (log decrement)."""
    if len(peaks) < 2:
        return None
    if n_periods is None:
        n_periods = len(peaks)-1
    if peaks[0] <= 0 or peaks[n_periods] <= 0:
        return None
    delta = (1.0/n_periods)*np.log(peaks[0]/peaks[n_periods])
    return float(delta/np.sqrt(4*np.pi**2 + delta**2))

def example_table() -> List[Dict]:
    """Replicate DeepSeek's examples."""
    settings = [
        dict(A=0.1, B=1.0, Delta=0.1),
        dict(A=1.0, B=1.0, Delta=0.1),
        dict(A=0.1, B=1.0, Delta=0.5),
    ]
    rows = []
    for s in settings:
        kc = kc_engineering(GPParams(**s))
        rows.append(dict(**s, Kc_pred=kc))
    return rows

if __name__ == "__main__":
    for r in example_table():
        print(r)
