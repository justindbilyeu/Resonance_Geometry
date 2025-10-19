from __future__ import annotations

import numpy as np

from experiments.jacobian import finite_diff_jacobian, max_real_eig


def test_finite_diff_jacobian_shapes() -> None:
    def f(x: np.ndarray, params: dict) -> np.ndarray:
        return np.array([x[1], -0.1 * x[1] - x[0]], dtype=float)

    x0 = np.array([1.0, 0.0], dtype=float)
    J = finite_diff_jacobian(f, x0, {})

    assert J.shape == (2, 2)
    assert isinstance(max_real_eig(J), float)
