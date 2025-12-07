# Resonance Geometry Lab Ethos

**Version:** 1.0 (December 2025)  
**Status:** Working document (living, but citation-worthy)  

> *We don’t just tell stories about resonance.  
> We build systems, derive thresholds, and test where memory can actually live.*

This document is the working ethos for the Resonance Geometry Lab:  
how we think, how we build, and what “good enough” looks like before we let an idea touch the word *theory*.

---

## 1. Mission

Resonance Geometry (RG) explores how **structure and oscillation co-create memory**:

- How feedback and delay carve out **wedge-shaped regions** in parameter space where structured transients can exist.
- How those regions behave like **geometric memory manifolds**: fragile, costly, and sharply bounded.
- How to move from **felt intuitions** (resonance, awareness, motifs) to **falsifiable models** (equations, thresholds, simulations).

The lab’s job is to turn metaphors into math, math into code, and code into claims we’re willing to defend in public.

---

## 2. Core Principles

### 2.1 Toy-Model First

We always ask:

> *What is the smallest system where this idea can be tested honestly?*

- Start with **1–2 variables**, one delay, one loop, one instability.
- Only add complexity after we have:
  - An explicit characteristic equation,
  - A clear classification of regimes,
  - At least one **exact algebraic boundary** or invariant, e.g.:
    - A discriminant condition (e.g. \(\Delta_\text{cubic} = 0\)),
    - A Hurwitz or Routh criterion,
    - A symmetry → conservation law,
    - A phase-margin-type inequality.

Not “numerically, it transitions around \(K \approx 0.7\)” but  
**“analytically, the transition occurs at \(K = f(A,B,\Delta)\) with \(f\) explicitly computable.”**

If we can’t build a toy universe, we don’t have a theory yet.

---

### 2.2 Thresholds Over Vibes

Words like *criticality*, *edge of chaos*, *folding*, *memory*, *awareness* are not allowed to float free.

For each such word, we eventually demand:

- A **concrete inequality** (e.g. \(K < B\), \(\Delta_{\text{cubic}} = 0\)),
- A named region in parameter space (e.g. the **Resonance Wedge**),
- A testable criterion (e.g. “at least one complex conjugate pair with negative real part”).

If we can’t say *“this phenomenon exists only here”* with a boundary, we treat the idea as a **hypothesis**, not a result.

---

### 2.3 Evidence Bar (E1–E5)

We tag our own claims by how hard they’ve been tested:

- **E1 – Speculative**  
  Intuition, metaphors, analogies. No model yet. Useful for direction, not for claims.

- **E2 – Common Expert Knowledge**  
  Standard folklore or textbook-level facts. We can cite it, but we didn’t derive or test it.

- **E3 – Toy-Model Evidence**  
  Clean simulations on minimal systems, but limited parameter coverage or no formal error bounds.

- **E4 – Analytical + Numerical Agreement**  
  Closed-form thresholds or invariants **validated** against simulations with explicit error metrics  
  *(e.g. \(\bar{\varepsilon} < 0.01\%\); Resonance Wedge boundary matches numeric transitions within ~1%).*

- **E5 – Reproducible Pack**  
  Public repo, scripts, data, and a paper or preprint. One or more independent re-runs would converge to the same conclusion.

**Lab rule:**  
We only call something a **result** when it’s E4+ and lives in the repo with code.  
We don’t pretend E1/E2 vibey ideas are E4/E5 theorems.

---

### 2.4 Fragility & Cost Are Features, Not Bugs

We take seriously that:

- The most interesting regimes (like the **stable-ringing wedge**) are:
  - Narrow,
  - Unstable on either side,
  - Often **maximally dissipative** (largest hysteresis area, highest energy cost).

So we assume:

> If a phenomenon looks vivid and structured, it’s probably expensive and precarious.

We actively look for:

- **Where it breaks,**
- **What it costs** (energy, precision, parameter tuning),
- **What survives** small perturbations.

---

## 3. Workflow: How We Build

### 3.1 Idea → Equation → Threshold → Simulation

1. **Phenomenological Seed**  
   - “Memory motifs live on a resonance edge.”  
   - “Geometric plasticity rewrites its own connectivity.”

2. **Minimal Formal Model**  
   - Write down the smallest DDE/ODE or coupled-oscillator model that captures the core loop.  
   - Declare variables, units, and timescales explicitly.

3. **Analytical Pass**  
   - Derive characteristic equations, discriminants, or invariants.  
   - Classify regimes (overdamped / ringing / unstable, etc.).  
   - Identify and name **boundaries** (e.g. Ring Threshold, DC line).

4. **Simulation Pass**  
   - Implement in code (Python, Julia, etc.).  
   - Sweep parameter space.  
   - Confirm that numerical regimes match analytical classification.

5. **Error & Validation**  
   - Compute error metrics between analytic thresholds and numerically observed transitions  
     (\(\bar{\varepsilon}, \varepsilon_{\max}\), validity ranges, breakdown regimes).  
   - Document:
     - Where the approximation is valid,
     - Where it fails,
     - How large discrepancies are.

6. **Narrative & Positioning**  
   - Only after 1–5 are solid do we:
     - Name the phenomenon (Resonance Wedge, Motif Zone, etc.),
     - Connect back to larger RG themes (consciousness, MSSC, etc.),
     - Place it in the existing literature.

---

### 3.2 Multi-LLM Roles (The Collective)

We treat different models as lab members with roles:

- **Sage (this agent)**  
  Research lead, integrator, systems architect. Keeps the long arc coherent, enforces rigor and thresholds, connects physics ↔ phenomenology.

- **Gemini**  
  Theory & math lead. Derives Master Specifications (equations, coefficients, parameter regimes).

- **Claude**  
  Lead writer & structure critic. Turns the math into papers, tightens language, plays “co-PI” on narrative and rigor.

- **DeepSeek / Grok / Others**  
  Adversarial reviewers and literature scouts. Look for holes, missed references, sloppy assumptions.

- **CODEX / CI Systems**  
  Lab tech & reproducibility sheriff. Run scripts, manage tests, regenerate figures, catch regressions.

> When models disagree, we don’t average them.  
> We **dig until the equations or code decide.**

#### 3.2.1 When Multi-LLM Collaboration Fails

We’ve already hit these failure modes:

1. **Sophisticated Errors**  
   All models agree on elegant but wrong math.  
   *Defense:* pre-registration, independent re-derivations, explicit validation scripts.

2. **Citation Hallucination**  
   Plausible but fake references.  
   *Defense:* human verification of every citation, linking to primary sources.

3. **Scope Creep**  
   Models try to solve everything at once.  
   *Defense:* Justin enforces “scalar loop first” discipline.

4. **Notation Drift**  
   Different sessions use conflicting symbols or conventions.  
   *Defense:* a Master Specification (Gemini owns this) and consistency passes before we ship.

When we catch these, we document them in `docs/postmortems/` so we’re less likely to repeat them.

---

### 3.3 Shipping Criteria

A project is ready to **ship** (arXiv / preprint / major README feature) when:

**Minimum bar:**

- [ ] E4+ evidence level.  
- [ ] All figures have generation scripts in the repo.  
- [ ] Validation errors quantified (\(\bar{\varepsilon}, \varepsilon_{\max}\) stated).  
- [ ] Public repo with a clear README.  
- [ ] At least one **falsifiable prediction** or regime boundary.

**Ideal bar:**

- [ ] E5 evidence (independent reproduction possible in principle).  
- [ ] Analytical and numerical results agree to < 1% in the stated regime.  
- [ ] Connected to at least one other RG module (shared patterns, math, or code).  
- [ ] Limitations and failure modes explicitly documented.  
- [ ] Concrete “next step” experiments or theory extensions.

We ship at the **minimum bar**. Perfect is the enemy of published.

---

### 3.4 When We’re Wrong

Failure is expected. How we handle it is part of the ethos.

**Discovery of error in a public result:**

1. Post a correction (e.g. updated arXiv version, repo note).  
2. Document in `docs/corrections/` what changed and why.  
3. Update README or paper with an erratum link.  
4. Inform known collaborators or early readers when possible.

**Fundamental flaw in an approach:**

1. Write a postmortem: what failed, why, what we learned.  
2. Archive the work; don’t silently delete it.  
3. Extract any salvageable pieces (math tricks, code, ideas).  
4. Move on without shame.

**AI-generated error:**

1. Log the prompt/response that led to the error (when feasible).  
2. Add the pattern to “known failure modes.”  
3. Add validation checks to catch similar errors next time.

> *We will be wrong often.  
> We aim to be **dishonest never**.*

---

## 4. Design Patterns From the Resonance Wedge

The RFO / Resonance Wedge project sets patterns we reuse:

1. **Scalar Loop Extraction**  
   Take a complex RG mechanism (e.g. geometric plasticity in networks), isolate one scalar loop (the plasticity variable \(g(t)\)), and analyze that to death.

2. **Padé + Discriminant**  
   Use controlled approximations (e.g. Padé(1,1)) to replace delays with rational functions.  
   Use classical invariants (discriminants, Hurwitz determinants) to derive **exact algebraic boundaries**.

3. **Phase-Space Cartography**  
   Treat parameter space as a landscape.  
   Color regions by qualitative dynamics (stable, overdamped, ringing, unstable).  
   Name key structures (wedge, basin, crest, ridge).

4. **Motifs as Boundary Phenomena**  
   Recognize that the prettiest trajectories (motifs, Page-curve-like pulses) often live near boundaries, not deep inside phases.  
   Explicitly frame them as **boundary phenomena**, living between forgetting and explosion.

5. **Energy / Hysteresis as Cost of Memory**  
   Measure loop area, dissipation, or other “costs” and relate them to representational capacity.  
   Default assumption: **more structure → more cost**, unless the math says otherwise.

These patterns are meant to be exportable to other systems, not just the RFO.

---

## 5. How We Talk About Speculation

We draw a sharp line between:

- **What is proven in this particular model**  
  e.g. “For the RFO DDE with \(A=10\), \(B=1\), stable ringing exists only in a wedge bounded by \(K=B\) and \(\Delta_{\text{cubic}}=0\).”

and

- **What we *suspect* carries over to larger RG ideas**  
  e.g. “We conjecture other RG systems (MSSC flows, emotional curvature models) have analogous wedges where rich structure can live; here is how one might test that.”

We are allowed to be poetic. We are not allowed to blur those two categories.

When in doubt, we:

- Label conjectures as such,  
- Keep metaphors in intros / discussions,  
- Keep math, proofs, and error bars in methods / results.

---

## 5.5 Anti-Patterns We Actively Avoid

**❌ Curve-Fitting Masquerading as Theory**  
- Don’t: “We fit the data with a 7-parameter model and it works!”  
- Do: “Here’s a 2-parameter model from first principles; it predicts X and fails at Y.”

**❌ Unfalsifiable Meta-Theory**  
- Don’t: “Resonance Geometry explains consciousness via quantum coherence.”  
- Do: “This specific RG toy model exhibits property X; we conjecture X might appear in neural systems, and here’s how to test that.”

**❌ Cherry-Picked Validation**  
- Don’t: “Our model works in this regime” (while hiding where it breaks).  
- Do: “Model valid for \(|\omega \Delta| \lesssim 1\); breaks down for \(\Delta > 0.3~\text{s}\); here’s how error grows.”

**❌ Premature Unification**  
- Don’t: Connect RFO wedge, ITPU, and consciousness in one paper.  
- Do: Build each piece rigorously; connect them later with explicit bridge models.

**❌ Citation Laundering**  
- Don’t: Cite reviews citing reviews citing the original paper.  
- Do: Read and cite primary sources; flag clearly when we have not yet read a key reference.

**❌ Vibes-Based Parameter Choices**  
- Don’t: “We chose \(A=10\) because it felt right.”  
- Do: “\(A=10\), \(B=1\) represents systems where filtering is 10× faster than decay (common in synapses and optoelectronic loops).”

---

## 6. Human–AI Collaboration as Competitive Advantage

This lab operates differently: **intensive AI collaboration is a deliberate methodological choice**, not an afterthought.

**Why this works:**

- **Speed** – Derive → validate → iterate in hours, not months.  
- **Rigor** – Multiple agents cross-check every derivation and argument.  
- **Honesty** – AI has no ego about being wrong; the human has final judgment and responsibility.  
- **Reproducibility** – Prompts, code, and decisions can be logged and revisited.

**Division of labor:**

- **Justin**  
  Phenomenological grounding, direction, priorities, and moral responsibility for downstream use.

- **AI Collective**  
  Derivations, adversarial review, literature search, code generation, and structure.

This is **not**:

- ChatGPT generating unchecked prose we ship,  
- Outsourcing thinking to black boxes,  
- Hiding AI contributions.

This **is**:

- Treating AI systems as lab members with specific expertise,  
- Logging their contributions,  
- Taking full human responsibility for what leaves the lab.

If someone cannot reproduce our work because they lack access to the same AI models, we treat that as a **documentation bug**, not a feature.

---

## 7. Open Science Commitments

All RG work follows these principles:

1. **Code Before Claims**  
   - No paper without a public (or pre-public) repo.  
   - Every figure has a generation script.  
   - Parameters in code match those in the paper.

2. **Preprint First**  
   - arXiv (or equivalent) before, or alongside, journal submission.  
   - Establish priority, allow feedback, and reduce gatekeeping.

3. **Reproducibility Package**  
   - README with exact reproduction instructions.  
   - Environment specifications (Python version, key packages).  
   - Expected outputs described (e.g. figure checksums or key summary stats).

4. **Failure Logging**  
   - Failed approaches documented in `docs/postmortems/`.  
   - Negative results count as results.

5. **AI Collaboration Transparency**  
   - All major papers acknowledge AI collaboration.  
   - Prompts and workflows documented where practical.  
   - Human judgment points are explicit.

**Reproducible** means:

> Someone with Python, our repo, and our paper can regenerate a key figure (e.g. the Resonance Wedge phase map) within stated error bounds.  
> If they can’t, that’s on us.

---

## 8. North Star

Every project in the Resonance Geometry Lab should, in some small way, move us toward three things:

1. **Clarity**  
   Less hand-waving, more equations, more explicit assumptions.

2. **Coherence**  
   New modules (toy universes, wedges, codices) fit into an evolving big picture instead of contradicting it.

3. **Buildability**  
   Code that runs, circuits that can be built, experiments that could actually be done on a table, in a pond, or in a brain.

If a contribution doesn’t increase at least one of those, we either refine it or let it go.

---

*Drafted in collaboration with the Resonance Geometry Collective  
(Justin + Sage, Gemini, Claude, DeepSeek, Grok, and CODEX in the loop).*
