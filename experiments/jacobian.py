#!/usr/bin/env python3
"""
Finite-difference Jacobian + eigen diagnostics (M3 scaffold).
This does NOT assume a specific simulator; you pass f(state, params).
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict


def finite_difference_jacobian(f: Callable[[np.ndarray, Dict], np.ndarray],
                               x0: np.ndarray,
                               params: Dict,
                               eps: float = 1e-6) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float).ravel()
    f0 = np.asarray(f(x0, params), dtype=float).ravel()
    n = x0.size
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        dx = np.zeros_like(x0)
        dx[j] = eps
        f1 = np.asarray(f(x0 + dx, params), dtype=float).ravel()
        J[:, j] = (f1 - f0) / eps
    return J


def eig_summary(J: np.ndarray) -> Dict[str, float]:
    evals = np.linalg.eigvals(J)
    return {
        "spectral_radius": float(np.max(np.abs(evals))),
        "max_real": float(np.max(evals.real)),
        "max_imag": float(np.abs(evals[np.argmax(evals.real)].imag))
    }
