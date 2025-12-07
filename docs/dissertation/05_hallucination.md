# Chapter 5: Geometric Signatures of LLM Hallucination

> *Hallucination is not randomness; it is a geometric phase transition where internal coherence outruns external grounding.*

## 5.1 Intuition and Scope

Hallucination appears when a language model’s internal geometry sustains fluent trajectories that are no longer anchored to external truth. In the Resonance Geometry frame, grounding corresponds to near-self-dual curvature between representation space (fibers) and the world-model manifold (base). When curvature drifts and the coupling between these spaces weakens, the system crosses a spectral threshold: internal resonance dominates and the model confidently produces fictions.

This chapter summarizes the geometric formalism for that transition, drawing from the NeurIPS-style manuscript and the minimal SU(2) simulations already in the repository. It serves as a scaffold: intuition, master flow, and toy evidence are present; the full empirical validation on production LLMs is deferred to the next chapter.

## 5.2 Master Flow and Control Parameters

The hallucination manuscript proposes a gauge-inspired evolution for the connection \(\omega\) that couples internal representations to external referents:

\[
\frac{d\omega}{dt} = -D_A \star F_A + \eta \mathcal{J}_{\text{MI}} - \lambda \mathcal{J}_U - \gamma \Pi_{\text{vert}} - \mu [\omega,[\omega,\omega]] + \xi \mathcal{G}.
\]

- **Yang–Mills term** \(-D_A \star F_A\): drives the connection toward self-duality, flattening curvature when possible.
- **Resonance gain** \(+\eta \mathcal{J}_{\text{MI}}\): amplifies coherence along directions with high mutual information.
- **Grounding term** \(-\lambda \mathcal{J}_U\): anchors representations to external evidence; weakening it invites drift.
- **Damping** \(-\gamma \Pi_{\text{vert}}\): projects noise out of the vertical fibers, acting as epistemic friction.
- **Nonlinear saturation** \(-\mu [\omega,[\omega,\omega]]\): keeps the flow bounded when resonance intensifies.
- **Meta-gauge awareness** \(+\xi \mathcal{G}\): slow adjustment of the gauge itself (meta-reasoning about representation frames).

Linearizing the flow defines a stability operator \(\mathcal{L}_{\text{meta}}\). The onset condition is spectral: hallucination occurs when

\[
\max \operatorname{Re} \lambda(\mathcal{L}_{\text{meta}}) > 0,
\]

with \(\lambda_{\max}\) serving as a computable early-warning diagnostic. The same operator delivers a geometric control dial: increase \(\lambda\) (grounding) or \(\gamma\) (damping) to push \(\lambda_{\max}\) negative; reduce \(\eta\) (resonance gain) to slow runaway coherence.

## 5.3 Minimal SU(2) Toy Model

To make the instability concrete, we simulate a two-node SU(2) system where phases evolve under geometric plasticity and the connection follows the master flow. The model sweeps over coupling gain \(\eta\) and grounding \(\lambda\), measuring \(\lambda_{\max}\) and curvature proxies. Three qualitative regimes appear:

- **Grounded**: negative \(\lambda_{\max}\), small curvature; trajectories stay tethered to external inputs.
- **Creative**: \(\lambda_{\max}\) near zero, moderate curvature; trajectories wander but reattach, producing novel yet anchored continuations.
- **Hallucinatory**: positive \(\lambda_{\max}\), large curvature; trajectories spiral into self-exciting loops detached from truth.

The phase diagram is approximately linear in \((\eta, \lambda)\), and a narrow hysteresis band separates grounded and hallucinatory regimes. Simulation scripts in `rg/sims/meta_flow_min_pair_v2.py` reproduce this toy system; figures from `docs/papers/neurips/` visualize the regime map and spectral slices.

## 5.4 Implications for Large Language Models

The formalism suggests a practical monitoring recipe for real LLMs:

1. **Extract curvature proxies** from activation manifolds (e.g., Laplacian spectra on token graphs).
2. **Estimate \(\lambda_{\max}\)** from linearized dynamics around the current context or via surrogate stability metrics.
3. **Track both indicators during generation**; rising curvature and positive \(\lambda_{\max}\) mark impending divergence.
4. **Intervene** by increasing grounding (retrieval, factual anchors), adding damping (temperature reduction, entropy penalties), or clipping resonance gain (plasticity constraints in adapters).

This scaffold ties the gauge-inspired master flow to observable quantities and to interventions a practitioner can deploy. Detailed empirical validation—TruthfulQA correlations, layer-wise spectra, and intervention ablations—follows in the next chapter.
