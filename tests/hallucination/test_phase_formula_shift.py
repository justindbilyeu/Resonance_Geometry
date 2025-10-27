"""
Tests for phase boundary formula shift under adaptive eta.

The phase boundary is predicted by:
    η * Ī * (1 + log(kappa)/d) ≈ λ + γ

Solving for critical η:
    η_crit = (λ + γ) / (Ī * (1 + log(kappa)/d))

As kappa increases (worse conditioning), the denominator increases,
so η_crit decreases (boundary shifts left).
"""
import numpy as np
import pytest


def eta_crit(lam, gam, I_bar, kappa, d):
    """
    Critical eta for phase boundary with adaptive gain.

    Args:
        lam: Grounding strength
        gam: Damping
        I_bar: Average mutual information
        kappa: Condition number
        d: Dimensionality

    Returns:
        Critical eta value for phase transition
    """
    if kappa < 1.0:
        kappa = 1.0  # Minimum conditioning
    whitening_factor = 1.0 + np.log(kappa) / d
    return (lam + gam) / (I_bar * whitening_factor)


class TestPhaseBoundaryShift:
    """Tests for phase boundary shift under adaptive whitening gain."""

    def test_eta_crit_decreases_with_kappa(self):
        """Critical eta should decrease as conditioning worsens."""
        lam, gam, I_bar, d = 0.5, 0.3, 0.8, 6
        e1 = eta_crit(lam, gam, I_bar, 1.0, d)        # kappa=1 (identity)
        e2 = eta_crit(lam, gam, I_bar, 10.0, d)       # moderate conditioning
        e3 = eta_crit(lam, gam, I_bar, 1000.0, d)     # ill-conditioned
        assert e1 > e2 > e3, "Critical eta should decrease as kappa increases"

    def test_well_conditioned_matches_baseline(self):
        """For kappa=1 (identity), should match non-adaptive formula."""
        lam, gam, I_bar, d = 1.0, 0.5, 1.0, 6
        eta_adaptive = eta_crit(lam, gam, I_bar, 1.0, d)
        eta_baseline = (lam + gam) / I_bar  # No whitening factor
        assert np.isclose(eta_adaptive, eta_baseline), "kappa=1 should match baseline"

    def test_boundary_shift_quantitative(self):
        """Test specific quantitative shift."""
        lam, gam, I_bar, d = 1.0, 0.5, 1.0, 6
        eta_baseline = eta_crit(lam, gam, I_bar, 1.0, d)  # kappa=1
        eta_shifted = eta_crit(lam, gam, I_bar, 100.0, d)  # kappa=100

        # log(100) ≈ 4.605, so whitening factor ≈ 1 + 4.605/6 ≈ 1.768
        expected_shift_factor = 1.0 / (1.0 + np.log(100) / 6)
        expected_eta = eta_baseline * expected_shift_factor

        assert np.isclose(eta_shifted, expected_eta, rtol=0.01), \
            f"Expected shift to {expected_eta:.3f}, got {eta_shifted:.3f}"

    def test_higher_dimension_reduces_shift(self):
        """Higher dimensionality should reduce the boundary shift."""
        lam, gam, I_bar, kappa = 1.0, 0.5, 1.0, 100.0

        eta_d6 = eta_crit(lam, gam, I_bar, kappa, d=6)
        eta_d60 = eta_crit(lam, gam, I_bar, kappa, d=60)

        # With higher d, log(kappa)/d is smaller, so eta_crit closer to baseline
        eta_baseline = (lam + gam) / I_bar
        assert abs(eta_d60 - eta_baseline) < abs(eta_d6 - eta_baseline), \
            "Higher dimension should reduce shift magnitude"

    def test_linear_scaling_with_grounding(self):
        """Boundary should scale linearly with λ + γ."""
        I_bar, kappa, d = 1.0, 10.0, 6
        gam = 0.5

        lam1, lam2 = 1.0, 2.0
        eta1 = eta_crit(lam1, gam, I_bar, kappa, d)
        eta2 = eta_crit(lam2, gam, I_bar, kappa, d)

        # Ratio should match (λ₁ + γ) / (λ₂ + γ)
        expected_ratio = (lam1 + gam) / (lam2 + gam)
        actual_ratio = eta1 / eta2

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01), \
            f"Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}"

    def test_inverse_scaling_with_mi(self):
        """Boundary should scale inversely with Ī."""
        lam, gam, kappa, d = 1.0, 0.5, 10.0, 6

        I1, I2 = 0.5, 1.0
        eta1 = eta_crit(lam, gam, I1, kappa, d)
        eta2 = eta_crit(lam, gam, I2, kappa, d)

        # Ratio should match I₂ / I₁
        expected_ratio = I2 / I1
        actual_ratio = eta1 / eta2

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01), \
            f"Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}"

    def test_extreme_conditioning_asymptote(self):
        """Very large kappa should approach an asymptotic limit."""
        lam, gam, I_bar, d = 1.0, 0.5, 1.0, 6

        eta_1e6 = eta_crit(lam, gam, I_bar, 1e6, d)
        eta_1e12 = eta_crit(lam, gam, I_bar, 1e12, d)

        # Should still be decreasing but at slower rate
        assert eta_1e6 > eta_1e12, "Should continue decreasing"
        # Relative change should be small
        rel_change = abs(eta_1e12 - eta_1e6) / eta_1e6
        assert rel_change < 0.5, "Change should slow down at extreme conditioning"


class TestWhiteningFactor:
    """Tests for the whitening factor computation."""

    def test_whitening_factor_identity(self):
        """Identity (kappa=1) should give factor=1."""
        kappa, d = 1.0, 6
        factor = 1.0 + np.log(kappa) / d
        assert np.isclose(factor, 1.0), "log(1) = 0, so factor should be 1"

    def test_whitening_factor_monotonic(self):
        """Factor should increase monotonically with kappa."""
        d = 6
        kappas = [1.0, 10.0, 100.0, 1000.0]
        factors = [1.0 + np.log(k) / d for k in kappas]
        assert factors == sorted(factors), "Factor should increase with kappa"

    def test_whitening_factor_dimension_scaling(self):
        """Higher dimension should reduce factor for same kappa."""
        kappa = 100.0
        factor_d6 = 1.0 + np.log(kappa) / 6
        factor_d60 = 1.0 + np.log(kappa) / 60
        assert factor_d6 > factor_d60, "Higher dimension should reduce factor"

    def test_whitening_factor_log_property(self):
        """log(kappa₁ * kappa₂) = log(kappa₁) + log(kappa₂)."""
        d = 6
        k1, k2 = 10.0, 100.0
        factor_product = 1.0 + np.log(k1 * k2) / d
        # Not exactly additive for factors, but log parts are
        log_sum = (np.log(k1) + np.log(k2)) / d
        log_product = np.log(k1 * k2) / d
        assert np.isclose(log_sum, log_product), "Log property should hold"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
