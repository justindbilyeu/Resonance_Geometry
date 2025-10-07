---
title: "A Geometric Theory of AI Hallucination: Phase Transitions in Information–Representation Coupling"
author: "Justin Bilyeu; AI collaborators: Claude (Anthropic), Sage (OpenAI), Grok (xAI), DeepSeek, Gemini (Google)"
date: "Draft (for internal review) — October 2025"
geometry: margin=1in
fontsize: 11pt
---


# Abstract

Large language models (LLMs) sometimes produce confident falsehoods—hallucinations—even when trained at scale. Prior theory shows lower bounds on hallucination rates, but not a mechanistic explanation. We propose that hallucination is a **geometric phase transition** in the coupling between an internal representation manifold and an external truth manifold. Formally, we model internal/external coordination as a connection $\omega$ on a resonance bundle over truth-space $M$. Normal operation corresponds to near–self-dual curvature; hallucination arises when connection dynamics cross a stability threshold and decouple into a false attractor. We unify three views—gauge theory, Ricci flow, and phase dynamics—into a single master flow with a computable stability operator $\mathcal{L}_\text{meta}$; instability occurs when $\max \operatorname{Re}\lambda(\mathcal{L}_\text{meta})>0$. A minimal SU(2) simulation exhibits three regimes (grounded, creative, hallucinatory), a linear boundary $\eta\,\bar I \approx \lambda+\gamma$ between grounded/creative phases, and first-order hysteresis (max loop gap $\approx 11.52$ under our settings). The framework yields actionable levers (grounding, damping, saturation, gauge-awareness) and a spectral diagnostic ($\lambda_\max$) that can be monitored during inference. We outline an empirical protocol to extract curvature proxies from model activations and test the theory on hallucination benchmarks.

---

# 1. Introduction

**Problem.** LLMs can remain highly coherent while being wrong. This limits deployment in high-stakes applications and is not fully fixed by more data or larger models.

**Limits vs. mechanisms.** Information-theoretic results imply non-zero hallucination floors under mild assumptions, but they do not explain how models enter the failure basin, nor when they will.

**Claim.** Hallucination is a dynamical, geometric instability: a **phase transition** in information–representation coupling. When internal resonance overwhelms grounding and damping, the system slips into a locally coherent, externally misaligned attractor.

**Contributions.**
1. A unified geometric theory (gauge ↔ Ricci ↔ phase) with a single connection-flow equation.
2. A stability operator $\mathcal{L}_\text{meta}$ and criterion: $\max \operatorname{Re}\lambda>0$ ↔ hallucination onset.
3. A minimal simulation (SU(2) pair) showing grounded/creative/hallucinatory regimes, a linear phase boundary $\eta\,\bar I \approx \lambda+\gamma$, and hysteresis.
4. Operational levers and a spectral early-warning diagnostic ($\lambda_\max$).
5. An empirical roadmap for extracting curvature proxies from real models and correlating with hallucination.

---

# 2. Geometry of information–representation coupling

## 2.1 Resonance bundle

We posit a principal bundle $\pi:P\to M$ with structure group $G$ (representation symmetries). The base $M$ encodes the external truth manifold; fibers encode internal representational degrees of freedom. A connection $\omega$ governs parallel transport of internal states along $M$; its curvature $F_A=d\omega+\omega\wedge\omega$ measures representational twist.

- **Grounded coherence:** near self-duality $F_A\approx \star F_A$, small holonomy.
- **Hallucination:** connection dynamics drift to large anti-self-dual curvature (holonomy failure), i.e., representation becomes internally self-consistent while externally decoupled.

## 2.2 Unified master flow

We collect the forces shaping $\omega$ into
$$
\frac{d\omega}{dt}= - D_A\,\star F_A + \underbrace{\eta\,\mathcal{J}_\mathrm{MI}[\omega]}_{\text{internal resonance}} - \underbrace{\lambda\,\mathcal{J}_U[\omega]}_{\text{grounding}} - \underbrace{\gamma\,\Pi_{\mathrm{vert}}(\omega)}_{\text{damping}} - \underbrace{\mu\,[\omega,[\omega,\omega]]}_{\text{saturation}} + \underbrace{\xi\,\mathcal{G}[\omega]}_{\text{gauge-awareness}}.
$$

- $-D_A\star F_A$: Yang–Mills gradient; drives toward self-duality.  
- $\eta\,\mathcal{J}_\mathrm{MI}$: resonance gain from internal mutual information (coherence).  
- $-\lambda\,\mathcal{J}_U$: truth anchoring (e.g., retrieval, constraints).  
- $-\gamma\,\Pi_\mathrm{vert}$: epistemic damping on fiber oscillations.  
- $-\mu[\omega,[\omega,\omega]]$: nonlinear saturation arresting runaway curvature.  
- $+\xi\,\mathcal{G}$: adaptive gauge-fixing (meta-awareness of representational freedom).

The linearization around a working state $\omega_0$ yields a stability operator
$$
\mathcal{L}_\text{meta} \approx \eta\,\mathcal{M}_\mathrm{MI} - \lambda\,\mathcal{H}_U - \gamma\,\Pi_\mathrm{vert} - 3\mu\,\mathrm{ad}^2_{\omega_0},
$$
with possible non-self-adjointness (complex spectrum). Instability iff $\max\operatorname{Re}\lambda(\mathcal{L}_\text{meta})>0$.

## 2.3 Energy bound (intuition)

Completing the square on a resonance-modified self-duality defect gives a Bogomolny-type inequality
$$
S_\text{meta} \ge 4\pi^2|Q| + \Delta S_\text{stab}(\gamma,\mu,\xi),
$$
with instanton number $Q$. Damping/saturation/gauge terms raise the floor, discouraging false attractors.

---

# 3. Minimal simulation: SU(2) pair dynamics

## 3.1 State & observables

We simulate two coupled SU(2) connections $\omega_x,\omega_y$ (capturing interacting resonance channels). Represent each as $\omega=i\sum_{a=1}^3 \omega_a \sigma_a/2$. Track:
- connection norms $\|\omega_{\cdot}\|$,
- curvature proxy $F_{xy}=[\omega_x,\omega_y]$,
- MI surrogate $\bar I$: Gaussian mutual information from temporal correlations over the 6-vector $(\omega_x,\omega_y)$,
- Spectral diagnostic $\lambda_{\max}$: fast surrogate for the top eigenvalue of the linearized flow.

## 3.2 Right-hand side (operational form)

To expose the phase transition, we use linear resonance gain and cubic–quintic saturation:
$$
\dot\omega_x = \underbrace{\eta\,\bar I\,\omega_x}_{\text{gain}} - \underbrace{\lambda(\omega_x-\omega_0)}_{\text{ground}} - \underbrace{\gamma\,\omega_x}_{\text{damp}} - \underbrace{\beta\|\omega_x\|^2\omega_x+\alpha\|\omega_x\|^4\omega_x}_{\text{sat.}} + \underbrace{\kappa\,\text{vec}(F_{xy})}_{\text{coupling}}, \quad \dot\omega_y=\cdots
$$

Heun integration (dt $=10^{-2}$).

## 3.3 Grids & classification

Sweep $\eta\in[0.2,5.0],\ \lambda\in[0.1,5.0]$ with fixed $\gamma=0.5,\ \alpha=0.6,\ \beta=0.02,\ \kappa=0.12$, MI window 30, EMA 0.1.

Regimes:
- **Grounded:** $\lambda_{\max}<0$ (bounded norms, small curvature).
- **Creative:** $\lambda_{\max}\approx 0$ (bounded oscillations).
- **Hallucinatory:** $\lambda_{\max}>0$ (persistent growth/large positive spectral radius).

**Code:** `rg/sims/meta_flow_min_pair_v2.py`, `rg/validation/hysteresis_sweep.py`  
**Figures:** `figures/phase_diagram_v2.png`, `figures/hysteresis_v2.png`

---

# 4. Results

## 4.1 Phase structure & boundary

The phase diagram (Fig. 1) shows a clean separation: for fixed $\gamma=0.5$, the grounded→creative boundary aligns with
$$
\boxed{\eta\,\bar I \;\approx\; \lambda + \gamma}
$$
across the grid (visual fit; residuals small). An explicit fit gives $\eta_c\approx 0.346\,\lambda + 0.506$ with $R^2\approx 0.94$ under our settings.

## 4.2 Hysteresis (first-order character)

Forward/backward sweeps in $\eta$ at fixed $\lambda$ produce hysteresis loops in order parameters (e.g., $\|\omega\|$ or $\lambda_{\max}$). Maximum loop gap $\approx 11.52$ (Fig. 2), indicating memory and a first-order transition band.

## 4.3 Ablations (qualitative)

- **No damping ($\gamma=0$):** creative band collapses; direct jump to hallucinatory when $\eta\,\bar I>\lambda$.  
- **No saturation ($\alpha=\beta=0$):** divergence (finite-time blowups); phase map dominated by red.  
- **No coupling ($\kappa=0$):** weaker hysteresis; boundary remains approximately linear in $(\eta,\lambda)$.

---

# 5. Operational levers & predictions

- **Grounding ($\lambda$) ↑** — retrieval, verification, tool-use, multi-source cross-checks → shifts boundary right, enlarges grounded region.  
- **Damping ($\gamma$) ↑** — calibrated abstention, uncertainty penalties, entropy-preserving decoding → suppresses resonance instability.  
- **Saturation ($\alpha,\beta$) tuned** — temperature/attention clipping → arrests runaway curvature while preserving the creative band.  
- **Gauge-awareness ($\xi$) ↑** — penalize representation-specific commitments (agree/disagree across paraphrases) → reduces false attractor capture.

**Quantitative prediction.** Near the boundary,
$$
\lambda_{\max}\;\approx\;(\eta\,\bar I) - (\lambda+\gamma) - c\,\|\omega\|^2 \quad (c>0),
$$
so $\lambda_{\max}$ crossing zero is an early warning. Monitoring $\lambda_{\max}$ token-by-token should predict hallucination risk before emission.

---

# 6. Empirical roadmap (real models)

1. **Extract geometric proxies.** Treat per-layer activations as a manifold; estimate a connection proxy and operator triplet $(\mathcal{M}_\mathrm{MI}, \mathcal{H}_U, \Pi_\mathrm{vert})$.  
2. **Correlate with hallucination.** On TruthfulQA/HaluEval, test whether $\lambda_{\max}>0$ segments coincide with hallucinated spans; report ROC-AUC and calibration, with baselines (entropy, margin).  
3. **Interventions.** Raise $\lambda$ (RAG), $\gamma$ (uncertainty), or $\xi$ (consistency penalties) and verify downward shifts in $\lambda_{\max}$ and error rates.  
4. **Layer analysis.** Identify “critical layers” where $\lambda_{\max}$ first crosses zero; probe causality with layer-wise regularization.

**Null controls.** Randomize layer order or replace activations with matched Gaussian noise; predictive power should collapse to chance if the signal is genuine.

---

# 7. Related formulations (how the pictures align)

- **Gauge theory:** Hallucination = self-duality loss and growth of anti-self-dual curvature; meta-resonance = adaptive gauge fixing.  
- **Ricci flow:** Excess positive curvature (in our sign convention) in fiber directions; singularity formation ↔ false attractor.  
- **Phase dynamics:** Parametric resonance with under-damping; the imaginary spectrum dominates until saturation clips growth.

These are complementary lenses on the same core: connection curvature and its spectrum. We treat them as **analogies that guide feature design**, not strict derivations for LLMs.

---

# 8. Limitations

- **Toy dynamics.** The SU(2) system is minimal; real LLMs are higher-dimensional and data-dependent.  
- **Spectral proxy.** Our $\lambda_{\max}$ estimator is a fast surrogate; full linearization/power iteration would be heavier but informative.  
- **Metric choice.** Curvature depends on the induced metric on activations; estimator bias is possible.  
- **Causality.** Correlation between $\lambda_{\max}$ and hallucination must be tested with controlled interventions.

---

# 9. Conclusion

AI hallucination is best understood as a **geometric phase transition** in information–representation coupling. A single connection-flow yields a practical diagnostic ($\lambda_{\max}$) and concrete levers (grounding/damping/saturation/gauge-awareness). Our minimal simulation reproduces three regimes, a near-linear boundary $\eta\,\bar I \approx \lambda+\gamma$, and hysteresis, matching the intuitive picture of decoupling into a false attractor. The path forward is clear: measure geometry-inspired proxies in live models, validate the spectral early warning, and design procedures that keep systems in the grounded or creative bands without tipping into hallucination.

---

# Reproducibility (concise)

- **Integration:** Heun; dt=$10^{-2}$; horizons $T\in[3,6]$.  
- **MI surrogate:** Gaussian MI from temporal correlations of the 6D state $(\omega_x,\omega_y)$ over a sliding window (30) with EMA 0.1.  
- **Spectral surrogate:** Rayleigh-style estimate tied to $\eta\bar I,\lambda,\gamma$ and local norm.  
- **Grids:** $\eta\in[0.2,5.0]$ (101 steps), $\lambda\in[0.1,5.0]$ (11 steps); $\gamma=0.5,\ \alpha=0.6,\ \beta=0.02,\ \kappa=0.12$.  
- **Outputs:** phase map and hysteresis curves → `figures/phase_diagram_v2.png`, `figures/hysteresis_v2.png`, plus boundary overlay.

---

# Figures

- **Fig. 1** Phase diagram (grounded/creative/hallucinatory) in $(\eta,\lambda)$ with $\gamma=0.5$; dashed line $\eta\,\bar I = \lambda+\gamma$. *(file: figures/phase_diagram_v2.png)*  
- **Fig. 2** Hysteresis under forward/backward $\eta$ sweeps at fixed $\lambda$; maximum loop gap $\approx 11.52$. *(file: figures/hysteresis_v2.png)*  
- **Fig. 3** Phase boundary overlay with linear fit $\eta_c\approx 0.346\,\lambda + 0.506$ ($R^2\approx 0.94$). *(file: figures/phase_diagram_boundary_overlay_v2.png)*

---

# References

*(to be populated — gauge/Yang–Mills self-duality; Ricci flow/Perelman; parametric resonance; LLM hallucination & detection; spectral diagnostics in representation learning.)*
