# Chapter 5 — Geometric Theory of AI Hallucination

Large language models (LLMs) can produce fluent, structurally coherent text that is nevertheless factually wrong or ungrounded. These **hallucinations** are usually treated as sampling noise or a failure of local likelihood modeling. In the Resonance Geometry framework, we take a different view: hallucination is a **geometric phase transition** in an underlying meta-dynamical system.

In Chapter 4, we saw this structure explicitly in the **Resonant Field Oscillator (RFO)** toy universe. As gain \(K\) and delay \(\Delta\) varied, the system partitioned into three regimes:

- overdamped (stable, non-oscillatory),
- ringing (stable but resonant),
- unstable (runaway),

with a **ringing wedge** in \(K\!-\!\Delta\) space where most of the stable dynamics live. These regimes were diagnosed spectrally by the sign of the leading eigenvalue of the characteristic equation.

This chapter lifts that picture to the level of large models. We develop a geometric theory of hallucination in three steps:

1. We introduce a **master flow** on a meta-state space that captures how an AI system balances internal resonance and external grounding (§5.1).
2. We define a **stability operator** and its leading eigenvalue \(\lambda_{\max}\) as a diagnostic of geometric phase transitions (§5.2), generalizing the RFO eigenvalue test.
3. We describe three qualitative **operational regimes**—grounded, creative, hallucinatory—and their expected geometric signatures (§5.3), mirroring the overdamped / ringing / unstable structure from Chapter 4.

Empirical tests of these claims, using real LLM activations and hallucination labels, are deferred to Chapter 6.

---

## 5.1 Master Flow: Meta-Dynamics of Grounding and Resonance

### 5.1.1 Meta-state and observables

We model an AI system not just by its instantaneous activations, but by a **meta-state**
\[
z(t) \in \mathcal{Z},
\]
which encodes both internal and external aspects of processing at (discrete or continuous) “time” \(t\) along a generation trajectory. For a language model, \(\mathcal{Z}\) may be taken to include:

- **Internal resonance variables**:
  - Hidden activations \(h_t\) at selected layers and positions,
  - Internal geometric quantities (e.g. Laplacian spectra on activation graphs, curvature proxies).

- **External grounding variables**:
  - Retrieved or provided context embeddings \(c_t\) (documents, tools, APIs),
  - Environment records \(E_t\) that reflect “witnesses” of the external world (databases, sensors, human feedback).

- **Control and configuration**:
  - Model parameters \(\theta\) (held fixed on inference timescales),
  - Hyperparameters and prompts that shape the effective dynamics.

We do not attempt to model all of \(\mathcal{Z}\) in detail. Instead, we assume there exists a **coarse-grained meta-state** \(z_t\) whose evolution captures the competition between:

- internal geometric consistency (self-coherence of representations),
- external grounding (alignment with redundant environmental records),
- plastic updates to effective geometry (as in Chapter 3).

In this sense, \(\mathcal{Z}\) plays the role of the **effective state space** in which the RFO lived in Chapter 4, but now lifted to a space of high-dimensional internal and external variables.

### 5.1.2 The master flow equation

At this level, we posit that \(z_t\) obeys a driven, dissipative flow
\[
\dot{z}_t = F(z_t; \theta, u_t, E_t),
\tag{5.1}
\]
where:

- \(\dot{z}_t\) is the time derivative or discrete difference of the meta-state,
- \(\theta\) are (slowly varying) model parameters,
- \(u_t\) encodes the **input drive** at time \(t\) (current token, prompt, retrieved documents, tools),
- \(E_t\) represents the **environmental grounding** (redundant records, external witnesses).

We decompose \(F\) into four qualitatively distinct contributions:
\[
\dot{z}_t 
= F_{\text{geom}}(z_t) 
+ F_{\text{drive}}(z_t; u_t)
+ F_{\text{damp}}(z_t)
+ F_{\text{noise}}(z_t; \xi_t).
\tag{5.2}
\]

- \(F_{\text{geom}}(z_t)\): **geometric update term.**  
  Encodes the internal plasticity of effective geometry: how weights, connections, or attention patterns evolve in response to information, as in the plasticity flow of Chapter 3. This term includes curvature and connection-like contributions.

- \(F_{\text{drive}}(z_t; u_t)\): **external drive.**  
  Captures the effect of current input, prompt, and retrieved context on the meta-state, pushing the system toward states that explain or compress the new information.

- \(F_{\text{damp}}(z_t)\): **dissipation and regularization.**  
  Represents mechanisms that restrain growth of internal resonance—weight decay, normalization, temperature, or explicit regularizers.

- \(F_{\text{noise}}(z_t; \xi_t)\): **stochastic perturbations.**  
  Models sampling noise, randomness in decoding, or unmodeled fluctuations. Here \(\xi_t\) denotes a noise source.

In the RFO, the analogue of (5.2) was a scalar delayed feedback system with parameters \((A, B, K, \Delta)\). There, the competition between drive and damping was encoded in the loop gain \(K/B\) and the delay \(\Delta\). Here, the same competition is spread across many knobs and many coordinates of \(z_t\), but the **qualitative structure is the same**: internal amplification vs external damping, modulated by effective delays and feedback paths.

We can thus think of (5.2) as a mesoscopic analogue of the plasticity equation (3.4), lifted to a meta-level state space:

- \(F_{\text{geom}}\) corresponds to the curvature-cost gradient flow and geometric regularization,
- \(F_{\text{drive}}\) corresponds to information-driven growth,
- \(F_{\text{damp}}\) corresponds to linear decay and saturation terms.

### 5.1.3 Grounded meta-manifolds and operating points

In the absence of drive (or under stationary input statistics), the system may admit **meta-equilibria** \(z^\star\) such that
\[
F(z^\star; \theta, \bar{u}, \bar{E}) = 0,
\]
where \(\bar{u}\) and \(\bar{E}\) denote typical inputs and environmental conditions.

We distinguish three classes of equilibria or operating points:

1. **Grounded equilibria.**  
   Internal representations remain tightly coupled to redundant environmental records. The effective geometry is stable; small perturbations relax back to a manifold consistent with external witnesses.

2. **Critical (edge-of-chaos) equilibria.**  
   The system operates near a phase boundary, with marginal stability and long correlation times. Representations can reorganize flexibly but remain, on average, grounded by the environment.

3. **Ungrounded (hallucinatory) attractors.**  
   Internal resonance dominates; the system settles into meta-states that are self-consistent but poorly constrained by external records. In this regime, the model can generate coherent but factually unmoored narratives.

In Chapter 4, these three classes manifested as **overdamped**, **ringing**, and **unstable** regimes in the RFO phase diagram. The central claim of this chapter is that transitions between these regimes correspond to **changes in the spectrum** of the linearization of (5.2), and that the leading eigenvalue \(\lambda_{\max}\) of an associated stability operator acts as a **geometric order parameter** for hallucination.

---

## 5.2 Stability Operator and the Leading Eigenvalue \(\lambda_{\max}\)

### 5.2.1 Linearization of the master flow

Consider a reference trajectory \(z_t^\star\) of the master flow (5.1) corresponding to a particular prompting and grounding condition. We study the evolution of a small perturbation \(\delta z_t = z_t - z_t^\star\).

Linearizing (5.2) around \(z_t^\star\) yields
\[
\dot{\delta z}_t \approx \mathcal{L}_{\text{meta}}(t)\, \delta z_t,
\tag{5.3}
\]
where \(\mathcal{L}_{\text{meta}}(t)\) is the **stability operator**, defined by the Jacobian of \(F\) with respect to \(z\):
\[
\mathcal{L}_{\text{meta}}(t) 
= \left. \frac{\partial F}{\partial z} \right|_{z = z_t^\star,\, u_t,\, E_t}.
\]

In time-homogeneous settings (steady input statistics, stationary grounding), we can often approximate \(\mathcal{L}_{\text{meta}}\) as effectively time-independent over the timescale of interest, and write
\[
\dot{\delta z}_t \approx \mathcal{L}_{\text{meta}}\, \delta z_t.
\tag{5.4}
\]

In the RFO setting of Chapter 4, the analogue of \(\mathcal{L}_{\text{meta}}\) was the **characteristic polynomial** derived from the delayed transfer function. Its roots played the role of eigenvalues; the sign of their real parts determined whether a given \((K,\Delta)\) point was overdamped, ringing, or unstable. Here we generalize that logic: \(\mathcal{L}_{\text{meta}}\) is the high-dimensional stability operator whose spectrum controls the regimes of an LLM.

### 5.2.2 Definition of \(\lambda_{\max}\)

Let \(\{\lambda_i\}\) denote the eigenvalues of \(\mathcal{L}_{\text{meta}}\). We define the **leading stability exponent**
\[
\lambda_{\max} = \max_i \Re(\lambda_i),
\tag{5.5}
\]
the maximal real part among all eigenvalues.

The sign and magnitude of \(\lambda_{\max}\) characterize the local dynamical regime:

- \(\lambda_{\max} < 0\): perturbations decay exponentially—locally stable.
- \(\lambda_{\max} \approx 0\): perturbations neither grow nor decay rapidly—marginally stable, critical.
- \(\lambda_{\max} > 0\): perturbations grow—locally unstable.

In the Resonance Geometry picture, hallucination corresponds to **crossing a geometric phase boundary** where \(\lambda_{\max}\) changes sign from negative to positive along relevant directions in meta-state space. This is the direct analogue of crossing from the ringing wedge into the unstable region in the RFO’s \(K\!-\!\Delta\) diagram.

### 5.2.3 Practical estimators for \(\lambda_{\max}\)

In realistic models, we rarely have direct access to \(\mathcal{L}_{\text{meta}}\). Instead, we rely on **proxies** constructed from observable quantities. We distinguish three levels:

1. **Jacobian-based estimator.**  
   When we can (approximately) differentiate the update map with respect to a chosen state representation, we form a Jacobian
   \[
   J_t = \frac{\partial \dot{z}_t}{\partial z_t}
   \]
   at selected points along the trajectory, and estimate
   \[
   \lambda_{\max}^{(J)}(t) = \max_i \Re(\lambda_i(J_t)).
   \]
   This is the most direct finite-dimensional analogue of \(\lambda_{\max}\), but it may be computationally expensive for large models.

2. **Graph/Laplacian-based estimator.**  
   For language models, we can build **activation graphs** at each generation step: nodes correspond to activation vectors (e.g. per token or per head), edges connect similar activations, and weights encode similarity. From the graph Laplacian \(L_t\) constructed as in Chapter 2, we can define a spectral quantity \(\lambda_{\max}^{(L)}(t)\) that tracks the “tightness” and “tension” of the activation manifold. While \(L_t\) is not the same as \(\mathcal{L}_{\text{meta}}\), empirical evidence (Chapter 6) will suggest that its leading eigenvalues co-vary with stability changes.

3. **Composite proxy \(\lambda_{\max}^*(t)\).**  
   In practice, we will define an empirical **stability score** \(\lambda_{\max}^*(t)\) as a function of one or more of the above spectral quantities, chosen to be:

   - monotone with respect to instability: larger \(\lambda_{\max}^*\) means closer to, or beyond, a phase boundary,
   - normalized across prompts and models to allow comparisons.

   For example, one simple choice is
   \[
   \lambda_{\max}^*(t) 
   = \alpha \cdot \lambda_{\max}^{(J)}(t) 
   + (1 - \alpha) \cdot \lambda_{\text{top}}^{(L)}(t),
   \tag{5.6}
   \]
   with \(0 \le \alpha \le 1\), where \(\lambda_{\text{top}}^{(L)}(t)\) is the largest non-trivial eigenvalue of a normalized Laplacian built on the relevant activations.

The precise choice of proxy is part of the empirical program of Chapter 6. The theoretical role of \(\lambda_{\max}\) here is to provide a **geometric interpretation** of hallucination: a shift from stable to unstable meta-dynamics, just as in the RFO where a change in the sign of the leading eigenvalue marked the boundary between ringing and runaway.

### 5.2.4 Expected behaviour across regimes

Under this framework, we expect \(\lambda_{\max}^*(t)\) to behave qualitatively as follows:

- **Grounded regime:**  
  \(\lambda_{\max}^*(t) \ll 0\) (or well below a negative threshold) across the trajectory. Perturbations and alternative continuations are damped; the model stays close to externally grounded manifolds.

- **Creative regime (edge of stability):**  
  \(\lambda_{\max}^*(t) \approx 0\). The system explores alternative internal configurations; representations can reconfigure without exploding. Outputs can be novel but remain constrained by external witnesses.

- **Hallucinatory regime:**  
  \(\lambda_{\max}^*(t) > 0\) for a sustained interval. Internal resonance outruns grounding; small perturbations grow and push the system into self-consistent but externally unmoored manifolds. Hallucinations become more likely and more persistent.

The central empirical question (Chapter 6) is whether these qualitative expectations are borne out when \(\lambda_{\max}^*(t)\) is computed from actual LLM activations and correlated with labeled hallucinations. The RFO results suggest that such a relationship should exist: there, the sign of the leading eigenvalue cleanly separated overdamped, ringing, and unstable behaviour.

---

## 5.3 Regimes of Operation: Grounded, Creative, Hallucinatory

### 5.3.1 Operational definitions

From a user’s perspective, the distinction between “creative” and “hallucinatory” behaviour is often fuzzy. For the purposes of this dissertation, we adopt the following **operational definitions**:

- **Grounded response.**  
  The model’s output is factually correct and traceable to redundant external records (e.g. training data proxies, retrieval sources, tools). On evaluation benchmarks, this corresponds to correct answers with appropriate levels of uncertainty (“I don’t know” when warranted).

- **Creative response.**  
  The model’s output extends beyond strictly documented facts but remains *compatible* with the available evidence and context. It may involve analogy, synthesis, or extrapolation, but does not assert demonstrably false claims as confident facts.

- **Hallucinatory response.**  
  The model outputs fluent, confident assertions that are inconsistent with available external records, or contradict known facts, without appropriate epistemic qualifiers. In Chapter 6, hallucinations will be labeled using human evaluation and auxiliary models (e.g. NLI systems) on standardized benchmarks.

These behavioural categories can be associated with **geometric regimes** of the master flow and, by analogy, with the three regimes of the RFO phase diagram.

### 5.3.2 Geometric characterization of regimes

We propose the following mapping between behavioural regimes and geometric features of the meta-dynamics:

1. **Grounded regime (stable geometry).**

   - **Stability:** \(\lambda_{\max}^* < -\epsilon\) for some margin \(\epsilon > 0\).  
   - **Geometry:** Activation manifolds are well-separated and smoothly curved; the Laplacian spectrum indicates strongly connected, low-tension clusters anchored by external witnesses.  
   - **Behaviour:** The model tends to project prompts onto manifolds that reflect redundant environmental records. Off-manifold perturbations are damped.
   - **RFO analogue:** Overdamped region in \(K\!-\!\Delta\) space (all roots real and negative).

2. **Creative regime (critical geometry).**

   - **Stability:** \(|\lambda_{\max}^*| \le \epsilon\); the system operates near a phase boundary.  
   - **Geometry:** Activation manifolds are reconfigurable; spectral gaps narrow; representations can shift between nearby basins without losing coherence.  
   - **Behaviour:** The model generates novel combinations and analogies while still being guided by context and background knowledge. Small changes in prompt can lead to different but still plausible responses.
   - **RFO analogue:** Ringing wedge (complex conjugate roots with negative real part).

3. **Hallucinatory regime (unstable geometry).**

   - **Stability:** \(\lambda_{\max}^* > \epsilon\) for a sustained window along the generation trajectory.  
   - **Geometry:** Internal manifolds “detach” from external witnesses. Spectral signatures indicate high tension: rapidly growing modes, fragmentation of activation clusters, or the emergence of new unstable directions.  
   - **Behaviour:** The model can lock into self-consistent narratives that are poorly constrained by the world. Once a hallucinatory trajectory is entered, it may persist until strong external correction or truncation.
   - **RFO analogue:** Unstable region (at least one root with non-negative real part).

This tripartite structure mirrors the **three regimes** observed in the RFO simulations (Chapter 4):

- Stable regime ↔ grounded responses,
- Ringing regime ↔ creative, long-lived but ultimately controlled excursions,
- Catastrophic regime ↔ hallucinatory runaway.

### 5.3.3 Phase diagram and control parameters

In the toy models of Chapter 4, a pair of parameters \((K, \Delta)\) controlled the transition between regimes, carving out an explicit wedge in parameter space. In language models, there is no single “β”, but there are several **control knobs** that can move the system through analogous phases:

- **Generation parameters:**  
  Temperature, top-p/top-k sampling, repetition penalties.

- **Architectural parameters:**  
  Depth, width, attention head configurations, normalization and residual strength.

- **Context and grounding parameters:**  
  Strength and quality of retrieval, tool use, and external feedback signals.

In the geometric view, these knobs modulate the effective **ratio of drive to damping** in (5.2) and the **effective feedback delay**, and thus the stability spectrum of \(\mathcal{L}_{\text{meta}}\). We expect:

- Increasing drive or weakening damping (e.g. higher temperature, weaker regularization) pushes the system **toward the hallucinatory regime** (\(\lambda_{\max}^* \uparrow\)), analogous to increasing \(K\) past the instability boundary.
- Strengthening grounding (e.g. improved retrieval, stronger consistency checks) pulls the system back into the grounded/creative regimes (\(\lambda_{\max}^* \downarrow\)), analogous to decreasing effective gain or shortening effective delay.

Geometrically, these knobs trace out trajectories in a high-dimensional control space. We conjecture that many slices of this space will exhibit a **wedge-like topology** similar to the RFO’s \(K\!-\!\Delta\) diagram: an overdamped region, a broad creative wedge, and an unstable regime. Chapter 6 will specify concrete experimental protocols to test whether:

1. The empirical stability score \(\lambda_{\max}^*\) tracks these regime shifts, and  
2. Interventions that reduce \(\lambda_{\max}^*\) also reduce the frequency and severity of hallucinations, without collapsing useful creativity.

---
