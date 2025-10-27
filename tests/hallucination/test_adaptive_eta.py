import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from resonance_geometry.hallucination.adaptive_gain import compute_effective_eta, EtaEffEMA


class TestComputeEffectiveEta:
    """Tests for compute_effective_eta function."""

    def test_eta_monotone_wrt_conditioning(self):
        """Effective eta should increase with conditioning."""
        eta = 1.0
        # identity -> kappa=1
        Sig_I = np.eye(4)
        # ill-conditioned
        Sig_bad = np.diag([1.0, 1e-6, 1.0, 1.0])
        eta_I, k_I, _ = compute_effective_eta(eta, Sig_I)
        eta_bad, k_bad, _ = compute_effective_eta(eta, Sig_bad)
        assert k_I == 1.0, "Identity matrix should have kappa=1"
        assert k_bad > k_I, "Ill-conditioned matrix should have kappa > 1"
        assert eta_bad > eta_I, "Effective eta should increase with conditioning"

    def test_no_nans_near_singular(self):
        """Should handle near-singular matrices without NaNs."""
        eta = 1.0
        Sig = np.array([[1.0, 0.999999], [0.999999, 1e-12]])
        eta_eff, kappa, gain = compute_effective_eta(eta, Sig)
        assert np.isfinite(eta_eff), "eta_eff should be finite"
        assert np.isfinite(kappa), "kappa should be finite"
        assert np.isfinite(gain), "gain should be finite"

    def test_identity_returns_base_eta(self):
        """Identity covariance (kappa=1) should return base eta."""
        eta = 2.0
        Sig = np.eye(6)
        eta_eff, kappa, gain = compute_effective_eta(eta, Sig, tanh_cap=False)
        assert np.isclose(kappa, 1.0), "Identity should have kappa=1"
        assert np.isclose(gain, 0.0, atol=1e-10), "log(1)=0, so gain should be 0"
        assert np.isclose(eta_eff, eta), "eta_eff should equal base eta for identity"

    def test_tanh_cap_limits_gain(self):
        """Tanh cap should limit gain for extremely ill-conditioned matrices."""
        eta = 1.0
        # Very ill-conditioned
        Sig = np.diag([1.0, 1e-10, 1.0, 1.0, 1.0, 1.0])
        eta_uncapped, _, gain_uncapped = compute_effective_eta(eta, Sig, tanh_cap=False)
        eta_capped, _, gain_capped = compute_effective_eta(eta, Sig, tanh_cap=True)

        assert gain_capped < gain_uncapped, "Capped gain should be less than uncapped"
        assert gain_capped <= 1.0, "Tanh should cap gain at most 1.0"
        assert eta_capped < eta_uncapped, "Capped eta should be less than uncapped"

    def test_epsilon_prevents_division_by_zero(self):
        """Epsilon regularization should prevent issues with zero eigenvalues."""
        eta = 1.0
        # Singular matrix (rank deficient)
        Sig = np.array([[1.0, 1.0], [1.0, 1.0]])
        eta_eff, kappa, gain = compute_effective_eta(eta, Sig, epsilon=1e-12)
        assert np.isfinite(eta_eff), "Should handle singular matrix"
        assert np.isfinite(kappa), "Kappa should be finite with epsilon"

    def test_d_scale_controls_normalization(self):
        """d_scale should control the normalization of log_kappa."""
        eta = 1.0
        Sig = np.diag([1.0, 0.1])  # kappa = 10

        _, _, gain_d2 = compute_effective_eta(eta, Sig, tanh_cap=False, d_scale=2)
        _, _, gain_d4 = compute_effective_eta(eta, Sig, tanh_cap=False, d_scale=4)

        # gain = log(kappa)/d_scale, so doubling d_scale halves gain
        assert np.isclose(gain_d4, gain_d2 / 2.0, rtol=0.01), "Doubling d_scale should halve gain"


class TestEtaEffEMA:
    """Tests for EMA smoother."""

    def test_initialization(self):
        """EMA should initialize to first value."""
        ema = EtaEffEMA(alpha=0.1)
        result = ema.update(2.0)
        assert result == 2.0, "First update should return the value itself"

    def test_smoothing_effect(self):
        """EMA should smooth out jumps."""
        ema = EtaEffEMA(alpha=0.1)
        ema.update(1.0)
        result = ema.update(10.0)
        # With alpha=0.1: new_state = 0.1*10 + 0.9*1.0 = 1.9
        expected = 0.1 * 10.0 + 0.9 * 1.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_convergence_to_constant(self):
        """EMA should converge to constant input."""
        ema = EtaEffEMA(alpha=0.5)
        target = 5.0
        result = target
        for _ in range(20):
            result = ema.update(target)
        assert np.isclose(result, target, rtol=0.01), "Should converge to constant"

    def test_alpha_zero_no_update(self):
        """Alpha=0 should freeze the state."""
        ema = EtaEffEMA(alpha=0.0)
        ema.update(1.0)
        result = ema.update(100.0)
        assert result == 1.0, "Alpha=0 should keep initial value"

    def test_alpha_one_instant_update(self):
        """Alpha=1 should immediately adopt new value."""
        ema = EtaEffEMA(alpha=1.0)
        ema.update(1.0)
        result = ema.update(10.0)
        assert result == 10.0, "Alpha=1 should immediately update to new value"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_eta_eff_increases_with_mi_window(self):
        """As covariance becomes more ill-conditioned over time, eta_eff should increase."""
        eta = 1.0
        # Simulate degrading covariance
        Sig1 = np.eye(6)
        Sig2 = np.diag([1.0, 0.5, 0.3, 0.2, 0.1, 0.05])
        Sig3 = np.diag([1.0, 0.1, 0.01, 0.01, 0.01, 0.001])

        eta1, _, _ = compute_effective_eta(eta, Sig1)
        eta2, _, _ = compute_effective_eta(eta, Sig2)
        eta3, _, _ = compute_effective_eta(eta, Sig3)

        assert eta1 <= eta2 <= eta3, "eta_eff should increase as conditioning worsens"

    def test_ema_smooths_eta_eff_spikes(self):
        """EMA should smooth out sudden spikes in eta_eff."""
        eta = 1.0
        ema = EtaEffEMA(alpha=0.1)

        # Good conditioning
        Sig_good = np.eye(6)
        eta_good, _, _ = compute_effective_eta(eta, Sig_good)
        smooth1 = ema.update(eta_good)

        # Sudden spike
        Sig_bad = np.diag([1.0, 1e-6, 1.0, 1.0, 1.0, 1.0])
        eta_bad, _, _ = compute_effective_eta(eta, Sig_bad)
        smooth2 = ema.update(eta_bad)

        # Smoothed value should be between good and bad
        assert eta_good < smooth2 < eta_bad, "EMA should smooth the spike"
        assert smooth2 - smooth1 < eta_bad - eta_good, "Change should be damped"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
