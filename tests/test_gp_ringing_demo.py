from __future__ import annotations

import numpy as np

from experiments.gp_ringing_demo import simulate_coupled, windowed_mi


def test_simulate_coupled_stays_finite() -> None:
    lam, x, y = simulate_coupled(fs=64, dur_up=5, dur_dn=5, lam_max=0.9, seed=1)

    assert np.isfinite(lam).all()
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    starts, mi_vals = windowed_mi(x, y, win=128, hop=32, bins=16)
    assert starts.size > 0
    assert np.isfinite(mi_vals).all()
