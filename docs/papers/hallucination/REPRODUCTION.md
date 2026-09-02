# Reproducing "A Geometric Theory of AI Hallucination"

What in the paper can be regenerated from this repository, what cannot, and
the exact commands. Written 2026-09-02 against commit `0832dde`.

Nothing in `src/` or `experiments/` was modified to produce these numbers.
The runs below use the paper's own parameters, from its section 3.3.

---

## 1. What was broken

The paper could not be reproduced as committed, for three reasons, none of
them to do with the science:

**Cited paths were wrong.** All five files the paper names by path were
somewhere else — the code under `src/`, the figures under
`docs/papers/neurips/figures/`. Fixed.

**No config matched the scripts.** `run_phase_map.py` and
`run_hysteresis.py` read `alpha`, `beta`, `kappa`, `ema_alpha`, and
`use_adaptive_gain`. The only config in the repo holding these values,
`hallucination_research/configs/hallu_su2_v2.yaml`, spells them `alpha_sat`,
`beta_sat`, `kappa_couple`, `ema_I`, and `adaptive_eta.enabled`. Both
scripts therefore died with `KeyError: 'alpha'` against the one file meant
to drive them.

The values are not in doubt — section 3.3 states them in prose ("$\gamma=0.5,
\alpha=0.6, \beta=0.02, \kappa=0.12$, MI window 30, EMA 0.1") and they match
`hallu_su2_v2.yaml` value for value. Only the key names had drifted. The
mapping is therefore read off the paper, not guessed.
`configs/hallucination/paper_section_3_3.yaml` is that config in the names
the scripts read. `hallu_su2_v2.yaml` was left alone.

**A file blocked the output directory.** Both scripts write figures into
`docs/papers/neurips/figures/Geometric Theory of AI Hallucination/`. A
15 KB extensionless text file — a plain-text copy of the paper — sat at
exactly that path, so `Path.mkdir()` raised `FileExistsError` and every run
crashed after computing its results. Renamed to `... .md`.

---

## 2. What reproduces

### 2.1 Hysteresis loop gap — reproduces exactly

    python experiments/hallucination/run_hysteresis.py \
      --config configs/hallucination/paper_section_3_3.yaml --lam 1.0

| | value |
|---|---|
| Paper, section 4.2 | max loop gap ≈ **11.52** |
| Figure title in `hysteresis_v2.png` | gap = **11.516** |
| This run | **11.5158** |

Confirmed. The paper's number is real and the figure is the figure it
claims to be.

### 2.2 Phase boundary — reproduces, and comes out BETTER than the paper reports

The paper boxes a prediction:

> η · Ī ≈ λ + γ

i.e. slope 1.0, intercept γ = 0.5. `compute_boundary_fit()` fits exactly
that quantity (η·Ī against λ), so the fitted coefficients are directly
comparable to the prediction.

| configuration | grid | n | fit | R² |
|---|---|---|---|---|
| `paper_section_3_3.yaml` | 25 × 11 | 16 | η·Ī = **0.996**λ + **0.502** | 0.998 |
| `ci_boundary_check.yaml` | 6 × 5 | — | η·Ī = **1.025**λ + **0.453** | 0.997 |
| `paper_section_3_3_adaptive.yaml` | 25 × 11 | 5 | η·Ī = **1.000**λ + **0.454** | 0.999 |
| **predicted** | | | η·Ī = **1.000**λ + **0.500** | |

Three independent grids, slope within 2.5% of 1.0 and intercept within
0.05 of γ. The boxed prediction holds.

---

## 3. What does NOT reproduce, and why it matters

**Section 4.1 reports η_c ≈ 0.346λ + 0.506 with R² ≈ 0.94, and describes
this as the boundary "aligning with" η·Ī ≈ λ + γ.**

It does not align. A slope of 0.346 against a predicted 1.0 is off by a
factor of ~2.9. The intercept, 0.506, matches γ almost exactly — so the
reported fit agrees with the theory on one coefficient and disagrees
sharply on the other, and the paper reports both without noting it. Figure
3 draws the two lines visibly diverging: at λ = 5 the theory line sits near
5.45 and the empirical points near 1.98.

Note also that R² ≈ 0.94 measures how *straight* the boundary points are.
It says nothing about whether the line matches the prediction. Placing it
next to a claim of alignment reads as support that it does not provide.

**The 0.346 figure could not be reproduced under any configuration tried
here** — paper grid, coarse grid, adaptive gain on, adaptive gain off. Every
run produced slope ≈ 1.0. Whatever generated 0.346 is not recoverable from
this repository.

The practical consequence is favourable and should be stated plainly: the
paper under-reports its own result. The current code supports the boxed
prediction considerably better than section 4.1 claims.

**Section 4.1 needs rewriting by its author.** It is not corrected here —
changing a paper's reported results is not a janitorial edit, and this file
records the discrepancy rather than resolving it.

---

## 4. Two figures are not what their names say

    hysteresis_v2.png       md5 85fd3fdb...  } identical
    hysteresis.png          md5 85fd3fdb...  }

    phase_diagram_v2.png    md5 e7a59baf...  } identical
    phase_diagram.png       md5 e7a59baf...  }

The `_v2` files are byte-for-byte copies of the v1 files. Independently:
`run_hysteresis.py` appends `[v2: Adaptive Gain]` to the plot title when
adaptive gain is enabled, and the title inside `hysteresis_v2.png` reads
`Hysteresis @ λ=1.0, γ=0.5 (gap=11.516)` with no such suffix. So that figure
was produced with adaptive gain **off**, whatever its filename implies.

`phase_diagram_boundary_overlay_v2.png` is the one genuinely distinct v2
figure.

The files are left in place. Renaming or regenerating them is the author's
call.

---

## 5. Standing check

`tests/test_paper_reproduction.py` re-runs the boundary sweep on every push
(≈8 s) and asserts the fitted slope stays in [0.85, 1.20] and the intercept
within 0.15 of γ. It brackets the *theory*, not the paper's printed
coefficients, because the theory is what the paper actually claims and the
printed coefficients are what could not be reproduced.

If someone changes the dynamics in `phase_dynamics.py` and the paper's
central result stops holding, CI goes red on that push instead of a reader
finding out later.

---

## 6. What this does and does not establish

It establishes that the paper's SU(2) simulation behaves as the paper's
theory says it should, and that its two headline numbers are honest outputs
of committed code.

It establishes nothing about language models. The system studied here is a
low-dimensional dynamical system with a resonance-gain term; the leap from
its phase structure to hallucination in a transformer is an interpretive
claim the simulation does not test. That gap is the paper's real exposure —
not its arithmetic, which holds up.
