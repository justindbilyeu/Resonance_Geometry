# AI Hallucination Paper: NeurIPS Submission

**Title**: A Geometric Theory of AI Hallucination: Phase Transitions in Information-Representation Coupling

**Authors**: Justin Bilyeu (with AI collaboration: Claude, Grok, DeepSeek, Gemini, Sage)

**Status**: Draft → ArXiv v1 (target: Jan 2025) → Workshop submission (target: Q2 2025)

---

## Abstract

Large language models sometimes produce confident falsehoods—hallucinations—even when trained at scale. We propose that hallucination is a geometric phase transition in the coupling between internal representation and external truth. We model this as connection dynamics on a resonance bundle, unifying gauge theory, Ricci flow, and phase dynamics. A minimal SU(2) simulation exhibits three regimes (grounded, creative, hallucinatory) and a linear phase boundary. The framework yields actionable diagnostics (spectral stability λ_max) and control levers. We outline an empirical protocol to test predictions on real LLMs.

---

## Paper Structure

- **Section 1**: Introduction (problem, claim, contributions)
- **Section 2**: Geometry of information-representation coupling
- **Section 3**: Minimal simulation (SU(2) pair dynamics)
- **Section 4**: Results (phase diagram, boundary, hysteresis)
- **Section 5**: Operational levers & predictions
- **Section 6**: Empirical roadmap
- **Section 7**: Related work
- **Section 8**: Limitations
- **Section 9**: Conclusion

---

## Files in This Directory

- **`manuscript.md`**: Main paper text (current version)
- **`supplement.md`**: (To be created) Additional proofs, ablations
- **`appendix_D_deepseek_eta_eff.md`**: Adaptive whitening gain derivation (η_eff formula)
- **`appendix_E_xai_replication.md`**: Independent replication by xAI/Grok (NumPy-only)
- **`figures/`**: All paper figures
  - `phase_diagram_v2.png`
  - `hysteresis_v2.png`
  - `phase_adaptive_overlay.svg` (adaptive η_eff comparison)
  - (More to be added from empirical work)
- **`figures/source/`**: (To be created) Scripts generating figures
- **`code_links.md`**: (To be created) Mapping results → code
- **`reviews/`**: (To be created) Review history
- **`versions/`**: (To be created) Version snapshots

---

## Code Generating Results

### Figure 1: Phase Diagram
- **Script**: `experiments/hallucination/run_phase_map.py`
- **Config**: `hallucination_research/configs/hallu_su2_v2.yaml`
- **Data**: `results/phase_diagram_v2.csv` (or similar)
- **Target**: `make hallu-phase-map`

### Figure 2: Hysteresis
- **Script**: `experiments/hallucination/run_hysteresis.py`
- **Config**: `hallucination_research/configs/hallu_su2_v2.yaml`
- **Data**: `results/hysteresis_v2.csv`
- **Target**: `make hallu-hysteresis`

### Figure 3: Conditioning Sweep (Appendix D)
- **Script**: `experiments/hallucination/run_phase_cond_sweep.py`
- **Data**: `results/phase_cond/phase_cond_sweep.csv`
- **Target**: `make phase-cond-sweep`
- **Description**: Parameter sweep with adaptive whitening gain

### Figure 4: Adaptive Overlay (Appendix D)
- **Script**: `experiments/hallucination/run_phase_adaptive_overlay.py`
- **Output**: `figures/Geometric Theory of AI Hallucination/phase_adaptive_overlay.svg`
- **Target**: `make phase-adaptive-overlay`
- **Description**: Comparison of base vs adaptive phase boundaries

### Figure 5-N: Empirical Results
- **Script**: (To be created) `experiments/hallucination/run_truthfulqa_extraction.py`
- **Analysis**: (To be created) `experiments/hallucination/analyze_results.py`
- **Status**: In progress (Q1 2025)

---

## Version History

### v0.9 (Pre-ArXiv, ~Jan 15 2025)
- Initial complete draft
- Theory sections 1-5 complete
- SU(2) simulations done
- Empirical section is roadmap only
- **Action**: Add Methodology + Epistemic Status sections
- **Action**: Pre-register predictions

### v1.0 (ArXiv Submission, target Jan 20-25 2025)
- Added Methodology section (multi-model collaboration)
- Added Epistemic Status section
- Updated acknowledgments
- Pre-registered empirical predictions
- **Status**: Ready for ArXiv

### v1.1 (Post-Feedback, target Feb-Mar 2025)
- Incorporate community feedback from ArXiv/Alignment Forum
- Add preliminary empirical results (even if partial)
- Refine notation based on comments
- **Status**: Planned

### v2.0 (Workshop Submission, target Q2 2025)
- Full empirical validation results
- Comparison to baselines
- Expanded related work
- **Target**: NeurIPS AI Safety Workshop or ICLR Robustness Workshop

---

## Pre-Registered Predictions

See: `[link to GitHub issue or separate file]`

**H1**: λ_max(L_sym) > threshold correlates with hallucination (AUC > 0.65)

**H2**: Instability emerges in middle-late layers (10-20 for GPT-2)

**H3**: Temperature reduction decreases λ_max (Δλ_max < 0 for 70%+ samples)

**H4**: Permuted data shows no signal (AUC < 0.55)

---

## Related Papers & Precedents

**Geometric ML**:
- Bronstein et al. "Geometric Deep Learning" (2021)
- FIOnet (NeurIPS 2020) - similar physics-inspired approach

**Hallucination Detection**:
- Lin et al. "TruthfulQA" (ACL 2022)
- [Add others as found]

**Information-Theoretic Bounds**:
- [To be added after literature search]

**Gauge Theory Applications**:
- [Standard references to be added]

---

## Submission Timeline

| Date | Milestone |
|------|-----------|
| Jan 15-20 | Finalize v1.0 with new sections |
| Jan 20-25 | Submit to ArXiv |
| Jan 25-30 | Post to Alignment Forum, solicit feedback |
| Feb 1-28 | Run empirical validation |
| Mar 1-15 | Update to v1.1 with preliminary results |
| Mar 15-31 | Prepare workshop submission (if applicable) |
| Apr-May | Workshop submission + revision |

---

## Key Contacts & Reviewers

*(To be filled as collaborations develop)*

- **Advisors**: [TBD]
- **Committee members**: [TBD]
- **External reviewers**: [Community members who provide feedback]
- **Collaborators**: Multi-model AI systems (documented in Methodology)

---

## Notes for Future Submissions

**If submitting to NeurIPS main conference**:
- Requires full empirical validation (not roadmap)
- Need comparison to 3+ baseline methods
- Address scalability beyond SU(2) toy model
- Expected review questions: "Why gauge theory specifically?", "What if results are null?"

**If submitting to workshop**:
- Can include work-in-progress empirical results
- Emphasize novel theoretical framework
- Focus on implications for AI safety
- More tolerance for speculation

**For JMLR (later)**:
- Combine theory paper + empirical validation paper
- No page limits - full exposition possible
- Requires rigorous proofs of all claims
- Timeline: 12-18 months review process

---

*Last updated: 2025-01-[DATE]*
*Maintainer: Justin Bilyeu*
