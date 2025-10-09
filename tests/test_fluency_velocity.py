# tests/test_fluency_velocity.py
import numpy as np
from experiments.fluency_velocity import (
    _synthetic_phases, phase_coherence, coherence_velocity,
    relaxation_velocity, phase_space_velocity, compute_fluency_metrics
)

def test_basic_metrics_run():
    phases = _synthetic_phases(T=600, N=12, noise=0.03, drift=0.0, seed=1)
    phi = phase_coherence(phases)
    vcoh = coherence_velocity(phases)
    vrel = relaxation_velocity(phases, perturb_start_idx=300)
    vphase = phase_space_velocity(np.c_[np.sin(phi), np.cos(phi)])  # fake 2D state

    m = compute_fluency_metrics(phases, omega_history=np.c_[phi, -phi], perturb_start_idx=300)

    # sanity
    assert len(phi) == 600
    assert len(vcoh) == 600
    assert np.isfinite(vrel)
    assert np.isfinite(vphase)
    for k in ("phi_mean","v_coh_std","v_relax","v_phase"):
        assert k in m and np.isfinite(m[k])
