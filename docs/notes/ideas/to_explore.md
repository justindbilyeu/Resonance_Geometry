# Ideas to Explore

*A running list of research directions, extensions, and "what ifs"*

---

## High Priority (Q1-Q2 2025)

### 1. Additional Curvature Proxies
**Idea**: Compare multiple geometric measures beyond Laplacian
- Ollivier-Ricci curvature
- Forman-Ricci curvature
- Persistent homology (Betti numbers)
- Attention flow curvature

**Why**: Robustness check, may find better predictor
**Effort**: Medium (1-2 weeks)
**Dependencies**: TruthfulQA extraction pipeline

---

### 2. Layer-Wise Analysis
**Idea**: Track where λ_max first crosses threshold across layers
- Hypothesis: Instability emerges in middle-late layers
- Could reveal "critical layers" for intervention

**Why**: Mechanistic insight, targeted interventions
**Effort**: Low (piggyback on extraction)
**Dependencies**: Curvature extraction working

---

### 3. Cross-Model Validation
**Idea**: Test on GPT-2, Llama-2, Mistral, Gemma
- Does phase boundary generalize?
- Are critical layers consistent?

**Why**: Generalizability claim
**Effort**: High (4-6 weeks across models)
**Dependencies**: Pipeline working on GPT-2

---

## Medium Priority (Q2-Q3 2025)

### 4. Quantum Extension of RG
**Idea**: What does "resonance geometry" mean for quantum systems?
- Quantum connections, quantum curvature
- Entanglement as resonance?

**Why**: Theoretical completeness, interdisciplinary appeal
**Effort**: Very High (months, need quantum info theory background)
**Dependencies**: Classical theory solid

---

### 5. Multi-Agent Resonance
**Idea**: Coupled LLMs, debate protocols, collective intelligence
- How do resonance geometries interact?
- Can collective gauge-fixing reduce individual hallucination?

**Why**: Exciting application, AI safety relevant
**Effort**: High (3-4 months)
**Dependencies**: Single-agent theory validated

---

### 6. Other Failure Modes
**Idea**: Apply geometric framework to:
- Specification gaming in RL
- Reward hacking
- Mode collapse in generative models
- Adversarial examples

**Why**: Broaden impact, test framework generality
**Effort**: Medium-High per mode
**Dependencies**: Hallucination work published

---

### 7. Real-Time Monitoring Dashboard
**Idea**: Inference-time λ_max tracker
- Visualize instability as model generates
- Alert when crossing threshold

**Why**: Practical deployment tool
**Effort**: Medium (2-3 weeks engineering)
**Dependencies**: Efficient λ_max estimation

---

## Low Priority / Speculative (2026+)

### 8. Biological Neural Coding
**Idea**: Does RG framework apply to neuroscience?
- Neural manifolds, representational geometry
- Phase transitions in perception/memory

**Why**: Foundational science, interdisciplinary
**Effort**: Very High (requires neuroscience collaboration)
**Dependencies**: Framework mature

---

### 9. Topological Data Analysis Integration
**Idea**: Combine RG with TDA (persistent homology, mapper)
- Topological features as phase transition signatures

**Why**: Mathematically elegant
**Effort**: High (learn TDA deeply)
**Dependencies**: Geometric foundation solid

---

### 10. Hardware Acceleration (ITPU)
**Idea**: Custom silicon for information-theoretic operations
- MI/entropy accelerators
- Geometric plasticity controllers

**Why**: Scaling, practical impact
**Effort**: Very High (years, need hardware expertise)
**Dependencies**: Software framework validated
**Note**: See `hardware/ITPU.md`

---

### 11. Gauge-Aware Training
**Idea**: Add ξ𝒢 term to loss function during training
- Train models that "know what they don't know"
- Meta-awareness of representational freedom

**Why**: Proactive solution, not just detection
**Effort**: High (months of training experiments)
**Dependencies**: Theory validated, compute resources

---

### 12. Philosophical Implications
**Idea**: What does geometric phase transition tell us about:
- Nature of understanding vs hallucination
- Consciousness and coherence
- Limits of AI systems

**Why**: Broader impact, interdisciplinary dialogue
**Effort**: Low (writing)
**Dependencies**: Empirical results solid

---

## Wild Ideas (Maybe Never)

### 13. Social Dynamics as Resonance
**Idea**: Memes, echo chambers, consensus as geometric phase transitions

**Why**: Fun, provocative
**Effort**: ??? (needs social science validation)
**Status**: Pure speculation

---

### 14. Economic Networks
**Idea**: Financial contagion, market crashes as curvature instabilities

**Why**: High-impact application
**Effort**: Very High (need economics background)
**Status**: Interesting analogy, unclear rigor

---

### 15. Music Theory Connection
**Idea**: Musical consonance/dissonance as literal resonance geometry

**Why**: Beautiful if true
**Effort**: Medium (literature exists)
**Status**: Metaphor or mechanism? Unclear

---

## How to Use This Document

- **Add freely**: No idea too wild for this list
- **Flag priority**: Mark High/Medium/Low based on current focus
- **Update effort**: As you learn more, refine estimates
- **Link to notes**: When exploring, create `docs/notes/[idea-name].md`
- **Celebrate completions**: Move to "Completed Ideas" section below

---

## Completed Ideas

*To be filled as ideas become reality*

---

*Last updated: 2025-01-[DATE]*
*Maintainer: Justin Bilyeu*
