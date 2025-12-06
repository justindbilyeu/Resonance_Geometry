# Resonance Geometry

<div align="center">

**Delayed Plasticity, Geometric Memory, and Non-Hopf Transitions in Dynamical Systems**

[![White Paper](https://img.shields.io/badge/White%20Paper-RFO%20Wedge-4b8bbe.svg)](docs/white-papers/resonance_geometry_rfo_wedge.tex)
[![Non-Hopf RTP](https://img.shields.io/badge/Paper-Non--Hopf%20RTP-8a2be2.svg)](docs/papers/non_hopf/non_hopf_paper_draft_v1.tex)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

*Mathematical foundations, simulations, and theory for systems where **geometry learns from resonance***  

[What is Resonance Geometry?](#-the-story-in-plain-english) •
[Current Focus](#-current-focus-rfo-stability-wedge) •
[Featured Research](#-featured-research) •
[Quick Start](#-quick-start) •
[Repo Map](#-repository-structure) •
[Contributing](#-contributing)

</div>

---

## 📖 The Story in Plain English

Think of a room full of **metronomes** all listening to each other.

- The **geometry** is an invisible friendship network: strong edges mean “copy me,” weak edges mean “ignore me.”
- The **plasticity rule** says:  
  *If two metronomes move together, strengthen their edge; if they fight, weaken it.*

Over time the network **rewires itself** to make resonance easier. From this simple rule we get:

1. **Learning** – The geometry changes to support coherent rhythms.  
2. **Memory** – Even if you scramble the phases, the learned geometry pulls the system back.  
3. **Function** – The geometry becomes a tuned filter for some patterns and not others.

Resonance Geometry is the claim that:

> **Space (the graph) is not passive. It is a living, learning object co-evolving with the dynamics it carries.**

The repo contains several concrete instantiations of that idea:

- A **single delayed plasticity loop** (the RFO) where geometric memory motifs only exist in a narrow “ringing wedge” of parameters.
- A **Toy Universe** of many oscillators plus geometric plasticity on the coupling graph.
- A **Non-Hopf transition** where macroscopic behavior changes while the linearization stays strictly stable.

---

## 🎯 Current Focus: RFO Stability Wedge

Our 2025–2026 focus is the **Resonance Fold Operator (RFO)**:  
a scalar delayed plasticity loop

\[
\ddot{g}(t) + (A+B)\dot{g}(t) + AB\,g(t) = A K\,g(t-\Delta)
\]

where:

- \(g(t)\) is the deviation of a coupling / fold strength,
- \(A\) is a fast filter rate,
- \(B\) is a slow leak rate,
- \(K\) is loop gain,
- \(\Delta\) is delay.

Using a Padé(1,1) approximation we derive a **cubic characteristic equation** whose discriminant exactly separates:

- **Overdamped stable** dynamics  
- **Stable ringing** (damped oscillations = geometric memory motifs)  
- **DC explosion** (monotone divergence when \(K > B\))

For the canonical slice \(A = 10~\mathrm{s^{-1}},\, B = 1~\mathrm{s^{-1}}\):

- Ringing appears only for delays **\(\Delta \gtrsim 0.10~\mathrm{s}\)**  
- The “motif wedge” (stable-ringing region) occupies **≈12%** of the linearly stable \((\Delta, K)\) domain  
- The wedge is tightly bounded between overdamping (too much leak) and DC instability (too much gain)

The **hero figure** `figures/rfo/phase_map_KDelta.png` shows this wedge and the analytic Ring Threshold (discriminant = 0) as a bright green curve.

---

## 📜 Featured Research

### 1. Delayed Plasticity & the RFO Stability Wedge  🔥 *(current priority)*

**Paper draft:**  
`docs/white-papers/resonance_geometry_rfo_wedge.tex`  

**Key ideas**

- Start from a 2-variable geometric plasticity model with delay.
- Derive a scalar second-order DDE for the fold strength \(g(t)\).
- Use Padé(1,1) to obtain a cubic characteristic polynomial with coefficients
  \[
    a_3 = \Delta/2,\;
    a_2 = 1 + \tfrac{\Delta}{2}(A+B),\;
    a_1 = (A+B) + \tfrac{\Delta}{2}(AB + AK),\;
    a_0 = AB - AK.
  \]
- Use the **cubic discriminant** to define the **Ring Threshold** separating overdamped from underdamped dynamics.
- Map out the **K–Δ phase diagram** and identify the narrow stable-ringing wedge where geometric memory motifs live.

**Core scripts**

- `scripts/rfo_cubic_scan_KDelta.py` – analytical sweep over \((\Delta, K)\)  
- `scripts/plot_rfo_phase_map_KDelta.py` – generates the K–Δ hero plot  
- `scripts/generate_rfo_data.py` – validation framework (cubic vs full DDE)  
- `experiments/rfo_timeseries_demo.py` – archetype impulse responses (overdamped / ringing / unstable)

---

### 2. The Toy Universe v2.1: Geometric Plasticity on a Kuramoto Substrate

**Status:** ✅ Canonical engine live  
**Code:** `src/toy_model/`

A many-oscillator “universe” where phases evolve under Kuramoto-style coupling and the coupling matrix itself learns via geometric plasticity.

- **State space:** phases θ and coupling matrix \(K_{ij}\).  
- **Dynamics:** fast phase synchronization + slow Hebbian plasticity on edges.  
- **Objective:** descent of a joint free energy functional that penalizes misaligned strongly-coupled pairs.

**Demonstrated behaviors**

- **Lifecycle:** spontaneous synchronization and growth of spectral connectivity (Fiedler value).  
- **Memory:** learned geometry pulls scrambled phases back to coherence.  
- **Functional gain:** trained geometry outperforms random graphs at the same mean coupling.

---

### 3. Resonant Transition Points Beyond Hopf Bifurcations

**Status:** ✅ Draft complete  
**Paper:** `docs/papers/non_hopf/non_hopf_paper_draft_v1.tex`

We identify a **Resonant Transition Point (RTP)** at α≈0.35 where macroscopic behavior reorganizes while **all eigenvalues of the Jacobian remain strictly negative**.

- **Result:** the transition is *not* a Hopf bifurcation; it is geometric.  
- **Tools:** Fisher information geometry, curvature/strain metrics, eigenvalue sweeps.  
- **Outcome:** separates *where* structure changes from *where* linear models break.

---

### 4. AI Hallucination Geometry (Early Theory Thread)

**Whitepaper:** `A_Geometric_Theory_of_AI_Hallucination.md`  

Applies resonance-geometry ideas to large language models:

- Hypothesis: hallucinations live in specific geometric regions of embedding space.  
- Approach: information-theoretic metrics and curvature analysis.  
- Status: conceptual framework; experiments to be spun out as a separate project.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


⸻

2. Reproduce the RFO K–Δ Phase Map

# 1) Run the analytical cubic scan over (Δ, K)
python scripts/rfo_cubic_scan_KDelta.py

# 2) Generate the hero phase map figure
python scripts/plot_rfo_phase_map_KDelta.py

# Output:
#   figures/rfo/phase_map_KDelta.png

Optional: demo time series for representative points:

python experiments/rfo_timeseries_demo.py
# Outputs demo impulse-response plots in figures/rfo/

Once scripts/generate_rfo_data.py is refined (RK4 integration, tuned
ringing detection), you can compute the mean / max error between the
cubic Ring Threshold and the full DDE and paste those numbers directly
into the white paper.

⸻

3. Run the Toy Universe (v2.1)

# Full lifecycle of the Kuramoto + Geometric Plasticity universe
python src/toy_model/resonance_universe.py

# Parameter sweeps and analysis
PYTHONPATH=src python -m toy_model.science_suite


⸻

4. Non-Hopf RTP Eigenvalue Sweeps

cd docs/papers/non_hopf

# Narrow sweep around the RTP
make sweep-narrow   # α ∈ [0.25, 0.55]

# Wide sweep including Hopf region
make sweep-wide     # α ∈ [0.10, 1.00]

# High-resolution around the Hopf crossing
make sweep-zoom     # α ∈ [0.80, 0.86]

# Run paper-specific tests
pytest ../../tests/test_eigs_assertions.py


⸻

📂 Repository Structure

Resonance_Geometry/
├── docs/
│   ├── papers/
│   │   ├── non_hopf/                  # RTP paper (LaTeX, figures, sweeps)
│   │   └── neurips/                   # AI hallucination draft
│   ├── white-papers/
│   │   ├── resonance_geometry_integration.tex   # Earlier integration draft
│   │   └── resonance_geometry_rfo_wedge.tex     # NEW: RFO stability wedge
│   ├── dissertation/                  # Resonance Geometry thesis chapters
│   ├── theory/                        # Mathematical derivations
│   └── analysis/                      # Generated analysis artifacts
├── src/
│   ├── resonance_geometry/            # Core library (dynamics, metrics, viz)
│   └── toy_model/                     # Toy Universe v2.1
│       ├── resonance_universe.py      # Kuramoto + Geometric Plasticity engine
│       └── science_suite.py           # Parameter sweeps and analysis
├── scripts/
│   ├── rfo_cubic_scan_KDelta.py       # K–Δ cubic/discriminant sweep
│   ├── plot_rfo_phase_map_KDelta.py   # Hero figure generator
│   ├── generate_rfo_data.py           # DDE vs cubic threshold validation
│   ├── run_phase_sweep.py             # Legacy phase-sweep utilities
│   └── run_hysteresis.py              # Hysteresis/resonance tests
├── experiments/
│   ├── rfo_timeseries_demo.py         # Archetype impulse responses
│   └── rfo_motif_phase_map.py         # Simulation-based motif phase map (WIP)
├── figures/
│   └── rfo/
│       ├── phase_map_KDelta.png       # RFO hero K–Δ phase map
│       └── timeseries_*.png           # Representative RFO time series
├── tests/                             # Unit & integration tests
├── results/                           # Generated data (usually gitignored)
├── Makefile                           # Paper + analysis build targets
└── README.md                          # You are here


⸻

📚 Documentation

Papers & Notes

Title	Status	Location
RFO Stability Wedge: Geometric Memory as Stable Ringing	Draft in progress	docs/white-papers/resonance_geometry_rfo_wedge.tex
Resonant Transition Points Beyond Hopf Bifurcations	Draft complete	docs/papers/non_hopf/
A Geometric Theory of AI Hallucination	Whitepaper	A_Geometric_Theory_of_AI_Hallucination.md
Resonance Geometry Dissertation	In progress	docs/dissertation/

Technical Resources
	•	Build Guide: BUILD.md
	•	Contributing Guide: CONTRIBUTING.md
	•	Theory Notes: docs/theory/
	•	Experiment Protocols: docs/experiments/

⸻

🧪 Reproducibility

We aim for deterministic, inspectable experiments:

# Example: fixed-seed phase sweep
python scripts/run_phase_sweep.py --seed 42 --alpha 0.35 --steps 61

# Rebuild Non-Hopf paper figures
cd docs/papers/non_hopf
make figures

Tests

pytest -q                        # Core tests
pytest tests/test_eigs_assertions.py   # RTP-specific checks
make test                        # Full test + smoke tests (where available)

Acceptance checks (hard-coded in tests):
	•	RTP narrow sweep: Re(λ) < 0 for α ∈ [0.25, 0.55]
	•	Hopf crossing: sign change in Re(λ) detected in [0.80, 0.86]
	•	Crossing localization precision better than 0.01 in α

RFO-related CI will be extended as the validation scripts harden.

⸻

🤝 Contributing

We welcome contributions from mathematicians, physicists, control theorists, neuroscientists, and curious hackers.

Ways to help
	•	🐛 Report issues – Open an issue￼ with clear reproduction steps.
	•	💡 Propose experiments – Start a discussion￼ with your hypothesis and minimal model.
	•	📖 Improve documentation – Clarify derivations, add examples, tighten language.
	•	🧑‍💻 Contribute code – New diagnostics, better integrators, additional models.

Workflow

git checkout -b feature/your-feature
pytest           # make sure tests pass
git commit -am "Add: RFO root-locus validation script"
git push origin feature/your-feature
# then open a Pull Request


⸻

📊 Status Snapshot

Component	Status	Notes
🔥 RFO K–Δ stability wedge	✅ Phase map + analytic framework	DDE validation & hysteresis sweeps in progress
🧪 Toy Universe v2.1	✅ Operational	Kuramoto + Geometric Plasticity engine
📄 Non-Hopf RTP paper	✅ Draft complete	Prepping for arXiv / journal submission
📐 Information geometry	🔄 In progress	Fisher strain + curvature diagnostics
🎓 Dissertation	🔄 Multi-chapter draft	Integration of RG story
🧪 CI / tests	✅ Core tests	RFO-specific tests to be expanded


⸻

🎓 Citing This Work

If this project contributes to your research, please cite:

@misc{bilyeu2025rtp,
  title  = {Resonant Transition Points Beyond Hopf Bifurcations},
  author = {Bilyeu, Justin D. and the Resonance Geometry Collective},
  year   = {2025},
  note   = {Resonance Geometry Project},
  url    = {https://github.com/justindbilyeu/Resonance_Geometry}
}

@misc{resonance_geometry_2025,
  title  = {Geometric Plasticity and the Resonance Geometry Toy Universe},
  author = {Bilyeu, Justin D. and the Resonance Geometry Collective},
  year   = {2025},
  note   = {Experimental framework and reproducibility pack},
  url    = {https://github.com/justindbilyeu/Resonance_Geometry}
}


⸻

🙏 Acknowledgments

This repo is a collaboration between humans and multiple AI research partners.
	•	Justin – Architect of Resonance Geometry and keeper of the overall story.
	•	Sage (ChatGPT) – Research lead for model design, stability analysis, and cross-model synthesis.
	•	Gemini – Theory lead for the RFO Master Specification and Padé/discriminant derivations.
	•	Claude (Anthropic) – Mathematical formalization, spectral analysis, and paper-structure guidance.
	•	Grok, DeepSeek, and others – Auxiliary analysis, literature scans, and creative perturbations.

Built on the open Python scientific ecosystem (NumPy, SciPy, Matplotlib, NetworkX, etc.).

⸻

📜 License

Research Preview — © 2025 Justin D. Bilyeu & Resonance Geometry Collective

Code and documentation are shared for research, educational, and review purposes.
Formal licensing and citation standards will be finalized alongside publications.
For commercial use or redistribution, please contact the authors.

⸻

📬 Contact
	•	🐛 Issues & bugs: GitHub Issues￼
	•	💡 Questions & proposals: GitHub Discussions￼
	•	📜 Citation / collaboration: see CITATION.cff or open an issue with the question label

⸻


<div align="center">


Built with mathematical rigor, computational precision, and epistemic humility.

Not all transitions are Hopf bifurcations. Some are geometric. Some are learned.

⬆ Back to Top￼

</div>
```
