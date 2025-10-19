from __future__ import annotations

import numpy as np

from experiments.fluency_velocity import fluency_velocity


def test_fluency_velocity_smoke() -> None:
    T, N = 400, 4
    t = np.linspace(0, 40, T)
    X = np.stack([np.sin(t + 0.2 * j) for j in range(N)], axis=1)
    out = fluency_velocity(X)

    assert "vf_mean" in out and "vf_std" in out
    assert isinstance(out["vf_mean"], float)
