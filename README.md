# Resonance Geometry

<div align="center">

**Delayed Plasticity, Geometric Memory, and Non-Hopf Transitions in Dynamical Systems**

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Research%20Preview-lightgrey.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/justindbilyeu/Resonance_Geometry)

*Mathematical foundations, simulations, and theory for systems where **geometry learns from resonance***

[Overview](#-overview) • [Quick Start](#-quick-start) • [Research](#-featured-research) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

-----

## 📖 Overview

### The Core Idea

Imagine a room full of **metronomes** listening to each other through an invisible network:

- **Geometry** = friendship network (strong edges = “copy me”, weak edges = “ignore me”)
- **Plasticity rule** = *if two metronomes sync, strengthen their connection; if they fight, weaken it*

Over time, the network **rewires itself** to amplify resonance. This simple feedback loop creates:

1. **Learning** – geometry adapts to support coherent rhythms
1. **Memory** – learned structure pulls scrambled phases back to order
1. **Function** – geometry becomes a tuned filter for specific patterns

**Resonance Geometry’s central claim:**

> Space (the coupling graph) is not passive—it’s a living, learning object co-evolving with the dynamics it carries.

### What’s in This Repo

- **Resonance Fold Operator (RFO)**: A scalar delayed plasticity loop where geometric memory exists only in a narrow **stable-ringing wedge**
- **Toy Universe**: Many Kuramoto oscillators with geometric plasticity on coupling
- **Non-Hopf Transitions**: Macroscopic reorganization with strictly stable linearization
- **Theory & Tools**: Analytical frameworks, phase diagrams, validation scripts

-----

## 🎯 Current Focus: The RFO Stability Wedge

**What:** A scalar delayed plasticity loop modeling geometric memory formation

**Equation:**
$$\ddot{g}(t) + (A+B)\dot{g}(t) + AB,g(t) = AK,g(t-\Delta)$$

where:

- $g(t)$ = coupling strength deviation from baseline
- $A$ = fast filter rate
- $B$ = slow decay rate
- $K$ = loop gain
- $\Delta$ = feedback delay

### Key Results

Using **Padé(1,1) approximation** → cubic characteristic equation → **discriminant analysis**:

**For canonical parameters** ($A = 10,\text{s}^{-1}$, $B = 1,\text{s}^{-1}$):

|Metric                  |Value                                                 |
|------------------------|------------------------------------------------------|
|**Critical delay**      |$\Delta \gtrsim 0.104,\text{s}$ for ringing to emerge |
|**Ringing fraction**    |61.1% of stable parameter space                       |
|**Validation error**    |$\bar{\varepsilon} = 0.0014%$ (discriminant vs. poles)|
|**Instability boundary**|$K = B$ (DC threshold)                                |

The system exhibits **three distinct regimes**:

1. **Overdamped** (too much damping) → monotonic decay, no memory motifs
1. **Stable ringing** (Goldilocks zone) → damped oscillations = geometric memory
1. **Unstable** ($K \geq B$) → runaway divergence

**Hero figure:** [`figures/rfo/phase_map_KDelta.png`](figures/rfo/phase_map_KDelta.png) shows the wedge with analytical Ring Threshold (green curve)

-----

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce the RFO Phase Map

```bash
# Generate analytical cubic sweep over (Δ, K)
python scripts/rfo_cubic_scan_KDelta.py

# Create phase diagram
python scripts/plot_rfo_phase_map_KDelta.py
# Output: figures/rfo/phase_map_KDelta.png

# Optional: validate analytical threshold
python scripts/rfo_validation.py
# Reports ε̄ = 0.0014%, ε_max = 0.0073%

# Optional: demo impulse responses
python experiments/rfo_timeseries_demo.py
# Outputs: figures/rfo/timeseries_*.png
```

### 3. Run the Toy Universe

```bash
# Full lifecycle: Kuramoto + Geometric Plasticity
python src/toy_model/resonance_universe.py

# Parameter sweeps and analysis
PYTHONPATH=src python -m toy_model.science_suite
```

### 4. Non-Hopf RTP Analysis

```bash
cd docs/papers/non_hopf

# Eigenvalue sweeps around RTP
make sweep-narrow   # α ∈ [0.25, 0.55]
make sweep-wide     # α ∈ [0.10, 1.00]
make sweep-zoom     # α ∈ [0.80, 0.86] (Hopf region)

# Run assertions
pytest ../../tests/test_eigs_assertions.py
```

-----

## 📜 Featured Research

### 🔥 1. RFO Stability Wedge (2025, Active)

**Paper:** [`docs/white-papers/resonance_geometry_rfo_wedge.tex`](docs/white-papers/resonance_geometry_rfo_wedge.tex)

**Core contributions:**

- Exact analytical criterion for when delayed plasticity can support ringing
- Padé(1,1) reduction → cubic discriminant separates overdamped/underdamped
- Complete $(K,\Delta)$ phase diagram with quantified wedge statistics
- Machine-precision internal validation ($\bar{\varepsilon} < 0.01%$)

**Key scripts:**

- `scripts/rfo_cubic_scan_KDelta.py` – parameter sweep
- `scripts/plot_rfo_phase_map_KDelta.py` – phase diagram
- `scripts/rfo_validation.py` – threshold validation
- `experiments/rfo_timeseries_demo.py` – archetype responses

### ✅ 2. Toy Universe v2.1: Geometric Plasticity Engine

**Code:** [`src/toy_model/`](src/toy_model/)

Many-oscillator system where:

- **Phases** evolve via Kuramoto coupling
- **Coupling matrix** learns via geometric plasticity (Hebbian-like)
- **Free energy** functional drives both dynamics

**Demonstrated behaviors:**

- Spontaneous synchronization with growing spectral connectivity
- Memory: learned geometry restores coherence after phase scrambling
- Functional gain: trained networks outperform random graphs

### ✅ 3. Resonant Transition Points Beyond Hopf

**Paper:** [`docs/papers/non_hopf/non_hopf_paper_draft_v1.tex`](docs/papers/non_hopf/non_hopf_paper_draft_v1.tex)

**Discovery:** Resonant Transition Point (RTP) at $\alpha \approx 0.35$ where:

- Macroscopic behavior reorganizes
- **All eigenvalues remain strictly negative** (not a Hopf bifurcation)
- Transition is **geometric**, not linear-instability-driven

**Tools:** Fisher information geometry, curvature metrics, eigenvalue sweeps

### 🔄 4. AI Hallucination Geometry (Theory Thread)

**Whitepaper:** [`A_Geometric_Theory_of_AI_Hallucination.md`](A_Geometric_Theory_of_AI_Hallucination.md)

Applies RG concepts to LLMs:

- Hypothesis: hallucinations occupy specific geometric regions in embedding space
- Approach: information-theoretic metrics + curvature analysis
- Status: conceptual framework, experiments TBD

-----

## 📂 Repository Structure

```
Resonance_Geometry/
├── docs/
│   ├── papers/
│   │   ├── non_hopf/              # RTP paper (LaTeX, figures, sweeps)
│   │   └── neurips/               # AI hallucination draft
│   ├── white-papers/
│   │   └── resonance_geometry_rfo_wedge.tex  # RFO stability paper
│   ├── dissertation/              # RG thesis chapters
│   ├── theory/                    # Mathematical derivations
│   ├── ETHOS.md                   # Lab methods & evidence standards
│   └── analysis/                  # Generated analysis artifacts
├── src/
│   ├── resonance_geometry/        # Core library
│   └── toy_model/                 # Toy Universe v2.1
│       ├── resonance_universe.py  # Kuramoto + plasticity engine
│       └── science_suite.py       # Parameter sweeps
├── scripts/
│   ├── rfo_cubic_scan_KDelta.py   # K-Δ analytical sweep
│   ├── plot_rfo_phase_map_KDelta.py  # Phase diagram generator
│   ├── rfo_validation.py          # Padé threshold validation
│   └── generate_rfo_data.py       # DDE validation framework (WIP)
├── experiments/
│   ├── rfo_timeseries_demo.py     # Impulse response demos
│   └── rfo_motif_phase_map.py     # Simulation-based mapping (WIP)
├── figures/
│   └── rfo/
│       ├── phase_map_KDelta.png   # Hero phase diagram
│       └── timeseries_*.png       # Example time series
├── tests/                         # Unit & integration tests
├── requirements.txt               # Python dependencies
├── Makefile                       # Build targets
└── README.md                      # You are here
```

-----

## 📚 Documentation

### Papers & Whitepapers

|Title                                                  |Status       |Location                                            |
|-------------------------------------------------------|-------------|----------------------------------------------------|
|RFO Stability Wedge: Geometric Memory as Stable Ringing|📝 Draft      |`docs/white-papers/resonance_geometry_rfo_wedge.tex`|
|Resonant Transition Points Beyond Hopf Bifurcations    |✅ Complete   |`docs/papers/non_hopf/`                             |
|A Geometric Theory of AI Hallucination                 |📋 Whitepaper |`A_Geometric_Theory_of_AI_Hallucination.md`         |
|Resonance Geometry Dissertation                        |🔄 In Progress|`docs/dissertation/`                                |

### Technical Resources

- **Lab Ethos:** [`docs/ETHOS.md`](docs/ETHOS.md) – Evidence bar (E1-E5), toy-model-first, “thresholds over vibes”
- **Build Guide:** [`BUILD.md`](BUILD.md)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Theory Notes:** [`docs/theory/`](docs/theory/)

-----

## 🧪 Reproducibility

All experiments use fixed seeds and documented parameters:

```bash
# Example: reproducible phase sweep
python scripts/run_phase_sweep.py --seed 42 --alpha 0.35 --steps 61

# Rebuild Non-Hopf figures
cd docs/papers/non_hopf
make figures
```

### Testing

```bash
# Core tests
pytest -q

# RTP-specific assertions
pytest tests/test_eigs_assertions.py

# Full test suite (where available)
make test
```

**Acceptance criteria (hard-coded in tests):**

- RTP narrow sweep: $\text{Re}(\lambda) < 0$ for $\alpha \in [0.25, 0.55]$
- Hopf crossing: sign change detected in $[0.80, 0.86]$
- Crossing localization: precision better than $0.01$ in $\alpha$

-----

## 🤝 Contributing

We welcome contributions from mathematicians, physicists, control theorists, neuroscientists, and curious builders.

### Ways to Help

- 🐛 **Report issues** – Clear reproduction steps appreciated
- 💡 **Propose experiments** – Start a discussion with hypothesis + minimal model
- 📖 **Improve docs** – Clarify derivations, add examples, tighten language
- 🧑‍💻 **Contribute code** – New diagnostics, better integrators, additional models

### Workflow

```bash
git checkout -b feature/your-feature
# Make changes
pytest  # Ensure tests pass
git commit -am "Add: your feature description"
git push origin feature/your-feature
# Open Pull Request
```

**Standards:** This repo follows the [RG Lab Ethos](docs/ETHOS.md):

- Evidence bar (E1-E5): claims graduate from speculation to result only with analytical backing + code
- Toy-model-first: start minimal, add complexity only after basics work
- Thresholds over vibes: every phenomenon needs a boundary condition

-----

## 📊 Project Status

|Component                 |Status          |Notes                                                                                       |
|--------------------------|----------------|--------------------------------------------------------------------------------------------|
|🔥 **RFO K-Δ wedge**       |✅ Active        |Phase map + analytical framework complete; Padé validation done; full DDE sweeps in progress|
|🧪 **Toy Universe v2.1**   |✅ Stable        |Kuramoto + geometric plasticity engine operational                                          |
|📄 **Non-Hopf RTP paper**  |✅ Draft complete|Prepping for arXiv/journal submission                                                       |
|📐 **Information geometry**|🔄 In progress   |Fisher strain + curvature diagnostics                                                       |
|🎓 **Dissertation**        |🔄 Multi-chapter |Integration of RG narrative                                                                 |
|🧪 **CI/Tests**            |✅ Core tests    |RFO-specific tests to be expanded                                                           |

-----

## 🎓 Citing This Work

If this project contributes to your research, please cite:

```bibtex
@misc{bilyeu2025rg,
  title  = {Resonance Geometry: Geometric Plasticity and Delayed Feedback Systems},
  author = {Bilyeu, Justin D. and the Resonance Geometry Collective},
  year   = {2025},
  url    = {https://github.com/justindbilyeu/Resonance_Geometry},
  note   = {GitHub repository}
}

@misc{bilyeu2025rtp,
  title  = {Resonant Transition Points Beyond Hopf Bifurcations},
  author = {Bilyeu, Justin D. and the Resonance Geometry Collective},
  year   = {2025},
  url    = {https://github.com/justindbilyeu/Resonance_Geometry}
}
```

*(arXiv reference will be added upon submission)*

-----

## 🙏 Acknowledgments

**Human-AI Collaboration:**

This research represents intensive collaboration between human and AI systems:

- **Justin D. Bilyeu** – Architect of Resonance Geometry, phenomenological grounding, research direction
- **Sage (ChatGPT)** – Research lead, model design, stability analysis, cross-model synthesis
- **Gemini** – Theory lead, Master Specifications, Padé/discriminant derivations
- **Claude (Anthropic)** – Mathematical formalization, spectral analysis, paper structure
- **DeepSeek, Grok** – Adversarial review, literature scanning, creative perturbations

Built on the open Python scientific ecosystem: NumPy, SciPy, Matplotlib, NetworkX, pytest.

See [`docs/ETHOS.md`](docs/ETHOS.md) for our methodology and collaboration framework.

-----

## 📜 License

**Research Preview** – © 2025 Justin D. Bilyeu & Resonance Geometry Collective

Code and documentation shared for research, educational, and review purposes.  
Formal licensing will be finalized alongside publications.  
For commercial use or redistribution, please contact the authors.

-----

## 📬 Contact

- 🐛 **Issues & bugs:** [GitHub Issues](https://github.com/justindbilyeu/Resonance_Geometry/issues)
- 💡 **Questions & proposals:** [GitHub Discussions](https://github.com/justindbilyeu/Resonance_Geometry/discussions)
- 📧 **Collaboration inquiries:** Open an issue with the `question` label

-----

<div align="center">

**Built with mathematical rigor, computational precision, and epistemic humility**

*Not all transitions are Hopf bifurcations. Some are geometric. Some are learned.*

[⬆ Back to Top](#resonance-geometry)

</div>
