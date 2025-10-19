import numpy as np

from experiments.fluency_velocity import (
    coherence_series,
    fluency_velocity,
    relaxation_velocity,
)


def test_oscillatory_series_has_nonzero_fluency():
    t = np.linspace(0, 20, 2000)
    s = np.sin(2 * np.pi * t / 3.0) * np.exp(-0.02 * t)  # damped oscillation
    phi = coherence_series(s)
    vf = fluency_velocity(phi)
    assert np.mean(np.abs(vf)) > 1e-4


def test_relaxation_velocity_positive_for_decay():
    t = np.linspace(0, 20, 2000)
    s = np.zeros_like(t)
    s[:1000] = 0.0
    s[1000:] = np.exp(-0.02 * (t[1000:] - t[1000]))  # decays to 0 after t0
    phi = coherence_series(s)
    vrel = relaxation_velocity(phi, t0=1000, eq_window=200)
    assert vrel > 0.0


def test_multinode_coherence_well_defined():
    T, N = 1000, 4
    t = np.linspace(0, 20, T)
    X = np.stack([np.sin(2 * np.pi * t / 5 + i * 0.2) for i in range(N)], axis=1)
    phi = coherence_series(X)
    assert phi.shape == (T,)
    vf = fluency_velocity(phi)
    assert vf.shape == (T,)
