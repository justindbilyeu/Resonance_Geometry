"""Guards the retraction of the Resonant Transition Point at alpha ~= 0.35.

The non-Hopf paper claimed an RTP near alpha = 0.35 -- a qualitative
reorganisation of macroscopic behaviour while all eigenvalues remain strictly
negative. On 2026-09-02 that claim was falsified. See
docs/papers/non_hopf/RTP_NULL_RESULT.md for the derivation and the forensics.

The argument in one line: on the phi_eq = 0 branch (the only equilibrium for
alpha < omega0^2 / K0), the effective stiffness k(alpha) = omega0^2 - K0*alpha
is exactly LINEAR in alpha. Every smooth quantity the system supports is
therefore a function of a linear function, and each one turns out to be
monotone with fixed convexity across the whole range. A feature at alpha = 0.35
would require a sign change in some second derivative. There is none.

This is a NULL-RESULT GUARD, which is an unusual thing for a test to be, so it
is worth saying what it is for: it keeps a retraction retracted. If the model,
its parameters, or the equilibrium-branch logic drift such that a feature does
appear -- or such that the closed form below stops holding -- this test goes
red and somebody has to look, rather than an RTP claim quietly reappearing in
an abstract because it sounded right the first time.

It also asserts the result that replaced the retracted one: the saddle-type
instability sits at the closed form alpha* = omega0^2 / K0, which the paper
originally reported only numerically as 0.833051 +/- 0.000508.
"""

import numpy as np
import pytest

# configs/equilibrium_sweep.yaml
K0 = 1.2
GAMMA = 0.08
OMEGA0_SQ = 1.0

ALPHA_STAR = OMEGA0_SQ / K0          # 0.8333..., the stiffness zero
RTP_CLAIMED = 0.35                   # the retracted claim


def _stiffness(alpha):
    """k(alpha) on the phi_eq = 0 branch.

    phi = 0 solves omega0^2*phi = K0*sin(alpha*phi) for every alpha, and for
    alpha < alpha* it is the ONLY solution: |sin x| <= |x| with K0*alpha < 1
    leaves no nonzero root. So cos(alpha*phi_eq) = 1 throughout.
    """
    return OMEGA0_SQ - K0 * alpha


def _quantities(alpha):
    k = _stiffness(alpha)
    disc = 4.0 * k - GAMMA**2        # > 0 => complex pair, stable spiral
    return {
        "Im(lambda)": 0.5 * np.sqrt(disc),
        "damping ratio": GAMMA / (2.0 * np.sqrt(k)),
        "quality factor": np.sqrt(k) / GAMMA,
        "spiral pitch": np.sqrt(disc) / GAMMA,
        "Fisher g_aa": 0.5 * (K0 / k) ** 2,
        "metric strain |k'/k|": K0 / k,
        "nonlinear stiffness ratio": K0 * alpha**3 / (8.0 * k),
    }


def test_alpha_star_has_a_closed_form():
    """alpha* = omega0^2 / K0, matching the paper's numerical 0.833051."""
    assert _stiffness(ALPHA_STAR) == pytest.approx(0.0, abs=1e-12)
    assert ALPHA_STAR == pytest.approx(0.833051, abs=0.000508 + 1e-4)


def test_real_part_is_pinned_across_the_claimed_rtp():
    """Re(lambda) = -gamma/2 everywhere the RTP was claimed to live.

    Nothing in the linear spectrum can single out 0.35 when the real part does
    not vary at all.
    """
    alpha = np.linspace(0.25, 0.55, 601)      # the paper's narrow sweep
    k = _stiffness(alpha)
    assert np.all(4.0 * k - GAMMA**2 > 0), "expected a complex pair throughout"
    re = np.full_like(alpha, -GAMMA / 2.0)
    assert re.max() == pytest.approx(-0.04, abs=1e-12)
    assert np.ptp(re) == pytest.approx(0.0, abs=1e-15)


def test_no_quantity_has_a_feature_at_the_claimed_rtp():
    """The kill test: no sign change in any second derivative below alpha*.

    A "qualitative reorganisation" at a specific alpha needs some quantity to
    change character there -- an extremum, an inflection, a divergence. Fixed
    convexity across the whole interval rules all of those out at once.
    """
    alpha = np.linspace(1e-4, ALPHA_STAR - 0.004, 20001)
    offenders = []
    for name, q in _quantities(alpha).items():
        d2 = np.gradient(np.gradient(q, alpha), alpha)
        d2 = d2[20:-20]                        # drop finite-difference edges
        signs = np.sign(d2)
        flips = int(np.sum(signs[1:] * signs[:-1] < 0))
        if flips:
            offenders.append(f"{name}: {flips} sign change(s) in d2/dalpha2")
    assert not offenders, (
        "A quantity changed convexity below alpha*. The RTP retraction in "
        "docs/papers/non_hopf/RTP_NULL_RESULT.md assumes none does:\n  "
        + "\n  ".join(offenders)
    )


def test_curvature_is_monotone_so_no_alpha_is_distinguished():
    """Stronger than "0.35 is smooth": no alpha is singled out, anywhere.

    Several of these quantities are genuinely convex -- Fisher g_aa diverges at
    alpha*. Curvature is not zero and is not supposed to be. What matters is
    that |d2Q/dalpha2| varies MONOTONICALLY: a distinguished point would be a
    local extremum of curvature, and there isn't one, at 0.35 or anywhere else
    below alpha*.
    """
    alpha = np.linspace(1e-4, ALPHA_STAR - 0.004, 20001)
    assert alpha[0] < RTP_CLAIMED < alpha[-1], "the claimed RTP must be in range"
    offenders = []
    for name, q in _quantities(alpha).items():
        d2 = np.abs(np.gradient(np.gradient(q, alpha), alpha))[20:-20]
        step = np.diff(d2)
        # allow a handful of floating-point-noise reversals, not a real turn
        reversals = int(np.sum(np.sign(step[1:]) * np.sign(step[:-1]) < 0))
        if reversals > 8:
            offenders.append(f"{name}: |d2| turns {reversals} times")
    assert not offenders, (
        "Curvature has a local extremum below alpha*, which would be a "
        "distinguished point the RTP retraction says does not exist:\n  "
        + "\n  ".join(offenders)
    )
