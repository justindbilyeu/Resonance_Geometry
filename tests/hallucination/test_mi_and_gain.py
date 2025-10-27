#!/usr/bin/env python3
"""
Unit and smoke tests for hallucination MI estimation and adaptive gain.

Tests:
1. gaussian_mi returns finite, non-negative
2. adaptive_gain_eta returns value >= base eta when use_adaptive=True
3. Smoke test: simulate for tiny grid, check no NaNs and regime differentiation
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from resonance_geometry.hallucination.phase_dynamics import (
    adaptive_gain_eta,
    compute_mi,
    simulate_trajectory,
    classify_regime,
)


class TestMutualInformation:
    """Tests for MI estimation."""

    def test_compute_mi_returns_finite(self):
        """MI should return finite, non-negative value."""
        # Create simple history
        history = [np.random.randn(6) * 0.1 for _ in range(40)]

        mi = compute_mi(history, window=30)

        assert np.isfinite(mi), "MI should be finite"
        assert mi >= 0.0, "MI should be non-negative"

    def test_compute_mi_empty_history(self):
        """MI should handle empty or short history gracefully."""
        history = []
        mi = compute_mi(history, window=30)
        assert mi == 0.0, "Empty history should return 0"

        history = [np.random.randn(6) for _ in range(2)]
        mi = compute_mi(history, window=30)
        assert mi == 0.0, "Short history should return 0"


class TestAdaptiveGain:
    """Tests for adaptive gain computation."""

    def test_adaptive_gain_disabled(self):
        """With use_adaptive=False, should return base eta unchanged."""
        eta_base = 2.0
        cov = np.eye(6) * 0.5

        eta_eff = adaptive_gain_eta(eta_base, cov, use_adaptive=False)

        assert eta_eff == eta_base, "Disabled adaptive gain should return base eta"

    def test_adaptive_gain_enabled(self):
        """With use_adaptive=True, should return eta >= base."""
        eta_base = 2.0
        # Create a covariance with some conditioning (not perfectly conditioned)
        cov = np.diag([1.0, 0.5, 0.3, 0.2, 0.15, 0.1])

        eta_eff = adaptive_gain_eta(eta_base, cov, use_adaptive=True)

        assert np.isfinite(eta_eff), "Adaptive eta should be finite"
        assert eta_eff >= eta_base, "Adaptive eta should be >= base eta"

    def test_adaptive_gain_identity_cov(self):
        """Identity covariance (cond=1) should give eta ~= base (log(1)/d = 0)."""
        eta_base = 2.0
        cov = np.eye(6)

        eta_eff = adaptive_gain_eta(eta_base, cov, use_adaptive=True)

        # log(1) = 0, so eta_eff = eta_base * (1 + 0/6) = eta_base
        assert np.isclose(eta_eff, eta_base, rtol=1e-6)

    def test_adaptive_gain_ill_conditioned(self):
        """Ill-conditioned covariance should boost eta."""
        eta_base = 2.0
        # Very ill-conditioned
        cov = np.diag([10.0, 1.0, 0.1, 0.01, 0.001, 1e-6])

        eta_eff = adaptive_gain_eta(eta_base, cov, use_adaptive=True)

        assert eta_eff > eta_base * 1.1, "Ill-conditioned cov should boost eta significantly"


class TestSimulationSmoke:
    """Smoke tests for simulation with adaptive gain."""

    def test_simulate_no_nans(self):
        """Simulation should not produce NaNs."""
        params = {
            'eta': 2.0,
            'lambda': 1.0,
            'gamma': 0.5,
            'k': 1.0,
            'alpha': 0.6,
            'beta': 0.02,
            'skew': 0.12,
            'mu': 0.0,
            'mi_window': 30,
            'mi_ema': 0.1,
            'omega_anchor': np.zeros(3),
            'use_adaptive_gain': True,
        }

        traj = simulate_trajectory(params, T=2.0, dt=0.01, seed=42)

        # Check no NaNs in key outputs
        assert not np.any(np.isnan(traj['E_dual'])), "E_dual should not contain NaNs"
        assert not np.any(np.isnan(traj['lambda_max'])), "lambda_max should not contain NaNs"
        assert not np.any(np.isnan(traj['MI_bar'])), "MI_bar should not contain NaNs"
        assert not np.any(np.isnan(traj['norm'])), "norm should not contain NaNs"
        assert np.isfinite(traj['final_x']).all(), "final_x should be finite"
        assert np.isfinite(traj['final_y']).all(), "final_y should be finite"

    def test_regime_differentiation(self):
        """Different (eta, lambda) should produce different regimes (weak check)."""
        params_base = {
            'gamma': 0.5,
            'k': 1.0,
            'alpha': 0.6,
            'beta': 0.02,
            'skew': 0.12,
            'mu': 0.0,
            'mi_window': 30,
            'mi_ema': 0.1,
            'omega_anchor': np.zeros(3),
            'use_adaptive_gain': True,
        }

        # Low eta, high lambda -> should be grounded
        params1 = params_base.copy()
        params1.update({'eta': 0.5, 'lambda': 3.0})
        traj1 = simulate_trajectory(params1, T=2.0, dt=0.01, seed=42)
        regime1 = classify_regime(traj1)

        # High eta, low lambda -> should be more unstable
        params2 = params_base.copy()
        params2.update({'eta': 4.0, 'lambda': 0.5})
        traj2 = simulate_trajectory(params2, T=2.0, dt=0.01, seed=42)
        regime2 = classify_regime(traj2)

        # Weak check: at least one should differ (not guaranteed, but likely)
        # or norm should be significantly different
        assert (regime1 != regime2) or (abs(traj1['norm'][-1] - traj2['norm'][-1]) > 0.5), \
            "Different (eta, lambda) should produce distinguishable dynamics"


class TestAdaptiveVsNonAdaptive:
    """Compare adaptive vs non-adaptive gain."""

    def test_adaptive_changes_trajectory(self):
        """Adaptive gain should produce measurably different trajectory than base."""
        params_nonadaptive = {
            'eta': 2.0,
            'lambda': 1.0,
            'gamma': 0.5,
            'k': 1.0,
            'alpha': 0.6,
            'beta': 0.02,
            'skew': 0.12,
            'mu': 0.0,
            'mi_window': 30,
            'mi_ema': 0.1,
            'omega_anchor': np.zeros(3),
            'use_adaptive_gain': False,
        }

        params_adaptive = params_nonadaptive.copy()
        params_adaptive['use_adaptive_gain'] = True

        traj_nonadaptive = simulate_trajectory(params_nonadaptive, T=3.0, dt=0.01, seed=42)
        traj_adaptive = simulate_trajectory(params_adaptive, T=3.0, dt=0.01, seed=42)

        # Final norms should differ (adaptive gain boosts dynamics)
        diff = abs(traj_adaptive['norm'][-1] - traj_nonadaptive['norm'][-1])
        assert diff > 0.01, "Adaptive gain should produce different final norm"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
