# PhD Dissertation: Geometric Approaches to Information Dynamics

**Candidate**: Justin Bilyeu  
**Status**: Pre-candidacy / Organizing materials  
**Expected Timeline**: 2025-2029 (tentative)  
**Institution**: [TBD]

## Dissertation Overview

This dissertation develops a geometric framework for understanding phase transitions in information-processing systems, with applications to AI safety and reliability.

**Central Thesis**: Complex information dynamics—from biological networks to artificial intelligence—exhibit geometric phase transitions that can be formalized using differential geometry, gauge theory, and dynamical systems theory.

## Dissertation Structure (Draft)

### Part I: Foundations

#### Chapter 1: Introduction
*Why geometry for information? Motivation, scope, contributions*

- The coherence-vs-grounding problem in AI systems
- Historical context: Information theory, neural coding, geometric deep learning
- Thesis roadmap and key claims
- **Status**: Outline only

#### Chapter 2: Mathematical Foundations
*Background for non-specialists*

- Differential geometry (manifolds, connections, curvature)
- Gauge theory and fiber bundles
- Information theory and mutual information
- Dynamical systems and bifurcation theory
- **Status**: Outline only

---

### Part II: General Theory

#### Chapter 3: Geometric Plasticity Framework
*How information flow sculpts adaptive structure*

- Resonant Witness Postulate (RWP)
- Geometric Plasticity (GP) principles
- Ringing boundaries and phase transitions
- Hysteresis and motif universality
- **Source**: `docs/whitepaper/`, `docs/appendices/`
- **Simulations**: `scripts/run_*.py`
- **Status**: ✅ Theory developed, appendices written

---

### Part III: Application to AI Safety

#### Chapter 4: Phase Transitions in LLM Hallucination
*Geometric theory of AI failure modes*

- Master flow equation (gauge + Ricci + phase dynamics)
- Stability operator and λ_max criterion
- Three regimes: grounded, creative, hallucinatory
- SU(2) minimal model and phase diagram
- **Source**: `docs/papers/neurips/manuscript.md`
- **Code**: `rg/sims/meta_flow_min_pair_v2.py`
- **Status**: ✅ Theory complete | 🔄 Empirical validation in progress

#### Chapter 5: Empirical Validation
*Testing predictions on real LLMs*

- Curvature extraction from neural activations
- TruthfulQA benchmark results
- Layer-wise analysis and λ_max correlation
- Intervention studies (RAG, temperature, uncertainty)
- **Code**: `experiments/hallucination/` (to be created)
- **Status**: 🔄 Protocol designed, execution in progress
- **Timeline**: Q1 2025

---

### Part IV: Extensions and Future Directions

#### Chapter 6: [Additional Applications]
*Other failure modes or domains*

- Possibilities: Specification gaming, reward hacking, mode collapse
- Multimodal extensions (vision, RL)
- Multi-agent resonance dynamics
- **Status**: ⏳ Exploratory

#### Chapter 7: Conclusion
*Synthesis, limitations, future work*

- What we learned about geometry and information
- Methodological contributions (AI-assisted theory development)
- Open questions and research directions
- **Status**: ⏳ To be written

---

## Current Progress Tracker

**Completed**:
- [x] General theory framework (GP/RWP)
- [x] Hallucination formalism (gauge theory)
- [x] Minimal simulations (phase diagrams, hysteresis)
- [x] Appendices (proofs for key results)

**In Progress**:
- [ ] Empirical validation protocol (curvature extraction)
- [ ] TruthfulQA experiments
- [ ] Paper submissions (ArXiv v1, workshop)
- [ ] Code reorganization

**Planned**:
- [ ] Additional benchmarks (HaluEval, etc.)
- [ ] Extension to other failure modes
- [ ] Chapters 1-2 (introduction, foundations)
- [ ] Tutorial notebooks and documentation

---

## Key Repository Locations

### Theory & Papers
- **Hallucination paper**: `docs/papers/neurips/manuscript.md`
- **GP whitepaper**: `docs/whitepaper/` (if exists)
- **Appendices**: `docs/appendices/*.md`

### Code
- **Core library**: `src/` (GP/RWP code)
- **Hallucination code**: `rg/sims/`, `rg/validation/` → to be moved to `src/resonance_geometry/hallucination/`
- **Experiment runners**: `scripts/` (general), to create `experiments/hallucination/`

### Results & Data
- **Simulation outputs**: `results/phase/`, `results/hysteresis/`, etc.
- **Figures**: Scattered (to be consolidated in `docs/dissertation/figures/`)

### Development
- **Research log**: `docs/notes/` (to be created)
- **Conversations**: `docs/conversations/` (to be created)
- **Tools**: `tools/` (to be created)

---

## Timeline & Milestones

### 2025 Q1 (Current)
- ✅ Organize repository structure
- 🔄 Complete empirical validation (TruthfulQA)
- 🔄 Submit ArXiv v1 (hallucination paper)
- ⏳ Draft dissertation chapters 1-2 outlines

### 2025 Q2
- Submit to NeurIPS workshop or ICLR workshop
- Expand empirical validation (more models, benchmarks)
- Begin chapter 3 draft (GP theory)
- Apply for fellowships (NSF GRFP, etc.)

### 2025 Q3-Q4
- Additional applications/extensions
- Conference submissions (main tracks)
- Refine theoretical framework
- Build practitioner toolkit

### 2026
- Complete Part III (hallucination + validation)
- Explore Part IV directions
- Additional publications
- Advance to candidacy

### 2027-2028
- Complete all chapters
- Integrate and unify
- External collaborations
- Teaching/mentoring

### 2029
- Thesis writing and defense
- Job market preparation
- Postdoc applications

---

## Open Questions & Research Directions

### Theoretical
1. Can we rigorously derive the master flow from first principles?
2. What determines the structure group G in real neural networks?
3. How does the framework extend to discrete token spaces?
4. Is there a quantum extension of the RG formalism?

### Empirical
1. Which curvature proxy best predicts hallucination?
2. Do the three regimes exist in real LLMs?
3. Can we extract λ_max efficiently during inference?
4. Does meta-resonance training reduce hallucination without harming creativity?

### Methodological
1. How do we validate AI-assisted theoretical development?
2. What are best practices for multi-model collaboration?
3. Can this methodology extend to other scientific domains?

---

## Collaboration & Contributions

This work has been developed through extensive dialogue with multiple AI systems (Claude, Grok, DeepSeek, Gemini, Sage). Full conversation excerpts and development history will be documented in `docs/conversations/`.

**Human collaborators**: [To be added]

**Acknowledgments**: [To be written]

---

## Contact & Updates

- **GitHub**: https://github.com/justindbilyeu/Resonance_Geometry
- **Issues**: Open for questions, suggestions, collaboration requests
- **Status**: Check this file for current progress

---

*Last updated: 2025-01-[DATE]*
*This is a living document and will evolve as the research progresses.*
