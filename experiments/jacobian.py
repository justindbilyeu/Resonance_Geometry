from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def finite_diff_jacobian(
    f: Callable[[np.ndarray, Dict], np.ndarray],
    x0: np.ndarray,
    params: Dict,
    eps: float = 1e-5,
) -> np.ndarray:
    """Return numerical Jacobian ``J_ij = d f_i / d x_j`` for ``x' = f(x, params)``."""

    x0 = np.asarray(x0, dtype=float)
    f0 = np.asarray(f(x0, params), dtype=float)
    n = x0.size
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        x = x0.copy()
        x[j] += eps
        fx = np.asarray(f(x, params), dtype=float)
        J[:, j] = (fx - f0) / eps

    return J


def max_real_eig(J: np.ndarray) -> float:
    """Return maximum real part among the eigenvalues of ``J``."""

    vals = np.linalg.eigvals(J)
    return float(np.max(vals.real))
