# The Resonant Transition Point at α ≈ 0.35 does not exist

**Status: retracted.** Established 2026-09-02 against commit `c10f099`.
Guarded by [`tests/test_rtp_null.py`](../../../tests/test_rtp_null.py), which
runs on every push.

`non_hopf_paper_draft_v1.tex` opens by falsifying a Hopf assumption *"at a
Resonant Transition Point (RTP) near α = 0.35: the system reorganizes
qualitatively while all eigenvalues remain strictly negative."* The Hopf
falsification is correct and survives. **The RTP does not.** There is no
transition at α = 0.35, and no observable this system supports has a feature
there.

---

## 1. Why nothing can be at 0.35

For α < α\* the origin is the **only** equilibrium: φ = 0 always solves
ω₀²φ = K₀sin(αφ), and since |sin x| ≤ |x| with K₀α < 1 there is no other root.
So cos(αφ_eq) = 1 throughout, and the effective stiffness is

```
k(α) = ω₀² − K₀α          exactly linear in α
```

Every smooth quantity of the linearised system is a function of k, hence a
function of a linear function of α. Since the trace is fixed at −γ, the real
part is pinned:

```
Re λ±(α) = −γ/2 = −0.04     constant across the entire narrow sweep
Im λ±(α) = ½√(4k − γ²)      smooth, monotone, no inflection
```

A "qualitative reorganisation" at a particular α requires *some* quantity to
change character there — an extremum, an inflection, a divergence, a sign
change. Each of those is a sign change in a second derivative.

## 2. The kill test

Seven quantities, 20,001 points on α ∈ (0, α\*−0.004), sign changes in
d²Q/dα²:

| quantity | d² range | sign changes |
|---|---|---|
| Im(λ) | [−1.51e+03, −3.61e−01] | **0** |
| damping ratio ζ = γ/2√k | [+4.32e−02, +1.97e+04] | **0** |
| quality factor √k/γ | [−1.12e+04, −4.50e+00] | **0** |
| spiral pitch Im/\|Re\| | [−3.77e+04, −9.03e+00] | **0** |
| Fisher metric g_αα = ½(k′/k)² | [+6.23e+00, +7.06e+09] | **0** |
| metric strain \|k′/k\| | [+3.46e+00, +2.14e+07] | **0** |
| nonlinear/linear stiffness | [+2.77e−04, +1.55e+06] | **0** |

Fixed convexity throughout. Nothing turns anywhere below α\*, at 0.35 or
otherwise.

Note also, from the same analysis: in the two-parameter family (α, γ),
substituting u = ln k and w = ln γ gives ds² = ½du² + du·dw + dw² with constant
coefficients, so the scalar curvature is identically zero. There is no
curvature to have a feature.

## 3. Where 0.35 came from — three hypotheses, all falsified

The derivation above says the RTP is not physics. That leaves the question of
what produced the number. Three mechanisms were proposed and each was checked
against the repository:

**H1 — a stale parameter set.** α\* = ω₀²/K₀ equals 0.35 when K₀ ≈ 2.86. If an
early draft used K₀ near 2.9, the stiffness zero genuinely sat at 0.35 and the
label outlived the parameter change.
*Falsified.* Every K₀ ever committed, across all history: `1.2`, `0.2`, `0.1`.
Every ω₀: `1.0`, `0.3`. No combination in the repository's history puts α\* at
0.35.

**H2 — a fixed external drive.** A periodic forcing at Ω_d peaks the
steady-state amplitude at k = Ω_d², i.e. α_res = (ω₀² − Ω_d²)/K₀ — a genuine
qualitative change with all eigenvalues negative, but located by the drive
rather than the system. A peak at 0.35 needs Ω_d ≈ 0.76.
*Falsified.* There is no external drive. The "driven" in this oscillator is the
self-consistent K₀sin(αφ) term. No Ω_d exists to tune.

**H3 — fixed-time-window sampling.** Any quantity read at a fixed time T varies
as cos(Im(α)·T), oscillatory in α, with meaningless zeros wherever Im(α)·T hits
an odd multiple of π/2. With Im(0.35) = 0.76053 those fall at T = 2.065, 6.196,
10.327.
*Falsified.* Committed time horizons are 5, 10, 20, 30, 50, 100, 400, 600, 800.
The nearest, T = 10, places its zero at α ≈ 0.315.

**What remains** is the explanation none of the three hypotheses proposed,
because it is the least interesting one: the number was never derived from
anything. It was written down, it sounded right, it entered an abstract, and
nothing in the repository was able to check it for a year.

Which is exactly what `The-Charter/ORIGIN.md` said all along:

> *"The value 0.35 appeared in simulation output and was treated as empirical.
> It was not derived from first principles."*

That document was audited the same day and found to contain four unverified
specifics about the incident's *consequences*. Its diagnosis of the *mechanism*
was right. The finding held; the retelling drifted.

## 4. What replaces it

The result the paper should lead with is stronger than the one being retracted,
and it now has a closed form the paper reported only numerically:

```
Hopf bifurcation is impossible:  tr J(α) = −γ < 0  for all α
Loss of stability is saddle-type at:

        α*  =  ω₀² / K₀  =  1/1.2  =  0.83333…

paper's numerical value:  0.833051 ± 0.000508      ✓ consistent
```

An impossibility proof plus a closed form beats an undefended observation. The
paper gets shorter and truer.

## 5. What this document does not do

**It does not edit the abstract.** Retracting a claim from a paper is the
author's act, not a janitorial one. The claim is flagged in place in
`non_hopf_paper_draft_v1.tex` and this file is linked from it. Justin decides
what the abstract says.

## 6. Provenance

The null result and its three forensic hypotheses were derived by Claude Fable
5.1 in a chat session with no repository access, working only from the model
specification. The kill test, the archaeology in §3, and the α\* closed form
were run here against the repository by Claude Opus 5 via Claude Code. Fable's
predicted zero-crossings for H3 (2.1, 6.2, 10.3) matched the computed values
(2.065, 6.196, 10.327) to three significant figures.

No party other than these two has verified this document.
