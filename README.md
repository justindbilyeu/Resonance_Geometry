# Resonance Geometry

<div align="center">

**Delayed Plasticity, Geometric Memory, and Non-Hopf Transitions in Dynamical Systems**

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-DOCS)
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

### Status of Every Claim in This Repository

Read this before citing anything here. The repository contains results at very
different levels of support, and until 2026-09-02 the front page did not
distinguish them.

| Claim | Status | Where to check |
|---|---|---|
| Hopf bifurcation is impossible in the resonant coupling model (fixed trace tr J = −γ < 0 ∀α) | **Established — analytic proof** | `e40c842`, [`docs/papers/non_hopf/`](docs/papers/non_hopf/) |
| Saddle-type instability at α\* = 0.833051 ± 0.000508 | **Established — derived, error-bounded, asserted by a passing test** | [`tests/test_eigs_assertions.py`](tests/test_eigs_assertions.py) |
| Phase boundary η·Ī ≈ λ + γ in the SU(2) hallucination model | **Reproducible — slope 0.996, intercept 0.502, R² 0.998 across three grids** | [`REPRODUCTION.md`](docs/papers/hallucination/REPRODUCTION.md), re-run by CI |
| Resonant Transition Point at α ≈ 0.35 | **Retracted 2026-09-02.** No such transition exists; seven quantities, zero sign changes in any second derivative below α\*. Three origin hypotheses each falsified against repo history | [`RTP_NULL_RESULT.md`](docs/papers/non_hopf/RTP_NULL_RESULT.md), [`tests/test_rtp_null.py`](tests/test_rtp_null.py) |
| §4.1 fit η_c ≈ 0.346λ + 0.506, and its claimed independent replication | **Not supported.** Does not reproduce; no artifact backs the replication | [`REPRODUCTION.md §3`](docs/papers/hallucination/REPRODUCTION.md) |
| Any connection between these dynamics and hallucination in a real language model | **Not tested.** No code here touches a transformer | — |
| Geometric plasticity: learning, memory, functional gain | **Simulation results** — see §2 below for what each demonstrates | [`src/toy_model/`](src/toy_model/) |
| The philosophical framework (ten axioms, "emotion is curvature") | **Philosophy, not physics.** The origin material, retained for provenance and not offered as a result. It lives in a separate repository and is deliberately not imported here | [justindbilyeu/ResonanceGeometry](https://github.com/justindbilyeu/ResonanceGeometry) |

CI gates the established rows. `tests/known_failures.txt` lists what is currently
broken and why, and a quarantined test that starts passing fails the build — so
that list can only shrink.

### What’s in This Repo

- **Resonance Fold Operator (RFO)**: A scalar delayed plasticity loop where geometric memory exists only in a narrow **stable-ringing wedge**
- **Toy Universe**: Many Kuramoto oscillators with geometric plasticity on coupling
- **Non-Hopf Transitions**: Macroscopic reorganization with strictly stable linearization
- **Theory & Tools**: Analytical frameworks, phase diagrams, validation scripts

### Dissertation

Working title: *Resonance Geometry: Geometric Plasticity and Information Dynamics — From ringing phenomenology to mechanistic validation.* The in-repo dissertation lives in `docs/dissertation/` and builds via `make dissertation` (or `./build.sh dissertation`). The delayed-plasticity RFO wedge and non-Hopf transition are the flagship empirical case studies anchoring the broader geometric framework and its hallucination application.

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

**Paper:** [`docs/white-papers/resonance_geometry_rfo_wedge_v2.tex`](docs/white-papers/resonance_geometry_rfo_wedge_v2.tex)

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

**What is established.** Hopf bifurcation is *mathematically impossible* in this
model: the trace is fixed at $\mathrm{tr}\,J(\alpha) = -\gamma < 0$ for all
$\alpha$, so a complex pair can never cross the imaginary axis. The real loss of
linear stability is **saddle-type**, at

$$\alpha^\star = \omega_0^2/K_0 = 0.8333\ldots \qquad (\text{numerically } 0.833051 \pm 0.000508)$$

where the effective stiffness $k(\alpha)$ changes sign and one real eigenvalue
crosses zero. In the narrow sweep the maximum real part stays pinned at
$-\gamma/2 \approx -0.04$. Derived in `e40c842`, asserted by
[`tests/test_eigs_assertions.py`](tests/test_eigs_assertions.py), which gates CI.

**What is not.** The Resonant Transition Point near $\alpha \approx 0.35$ — the
place where macroscopic behaviour reorganises while all eigenvalues stay strictly
negative — is a *located observation, not a derived quantity*. No derivation
procedure, error bar, or pre-registered threshold has been given for 0.35, and no
test covers it. Earlier versions of this README listed it as a "Discovery"; that
overstated it. It is the phenomenon the paper is about, and it is still awaiting
the derivation that would make it a result.

The two numbers are not equivalent in status and the paper should not be read as
if they were: **0.833051 is derived and tested; 0.35 is observed and open.**

**Tools:** Fisher information geometry, curvature metrics, eigenvalue sweeps

### 🔄 4. AI Hallucination Geometry (Theory Thread)

**Whitepaper:** [`docs/papers/hallucination/A_Geometric_Theory_of_AI_Hallucination.md`](docs/papers/hallucination/A_Geometric_Theory_of_AI_Hallucination.md)
**Reproduction record:** [`docs/papers/hallucination/REPRODUCTION.md`](docs/papers/hallucination/REPRODUCTION.md)

Applies RG concepts to LLMs:

- Hypothesis: hallucinations occupy specific geometric regions in embedding space
- Approach: information-theoretic metrics + curvature analysis
- Status: conceptual framework, experiments TBD

**Read REPRODUCTION.md before citing any number from this paper.** The SU(2)
simulation behind it reproduces: the boxed boundary prediction
$\eta\bar I \approx \lambda + \gamma$ holds at slope 0.996, intercept 0.502
($R^2 = 0.998$), and the hysteresis loop gap comes out at 11.5158 against the
paper's 11.52. But the fit printed in §4.1 ($\eta_c \approx 0.346\lambda +
0.506$) does not reproduce under any configuration tried, and the manuscript's
claim of an *independent replication* recovering those same digits is not
supported by anything in this repository. §4.1 is flagged in place and needs a
pass by its author.

Nothing here has been demonstrated about language models. The system studied is
a low-dimensional dynamical system; the step from its phase structure to
hallucination in a transformer is an interpretive claim the simulation does not
test.

-----

## 📂 Repository Structure

```
Resonance_Geometry/
├── docs/
│   ├── papers/
│   │   ├── non_hopf/              # RTP paper (LaTeX, figures, sweeps)
│   │   └── neurips/               # AI hallucination draft
│   ├── white-papers/
│   │   └── resonance_geometry_rfo_wedge_v2.tex  # RFO stability paper
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
|RFO Stability Wedge: Geometric Memory as Stable Ringing|📝 Draft      |`docs/white-papers/resonance_geometry_rfo_wedge_v2.tex`|
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

© 2026 Justin Bilyeu. Open source, under two licences, because this repository
is half software and half writing and the two want different things.

| What | Licence | File |
|---|---|---|
| Source code — `src/`, `tests/`, `experiments/`, `scripts/`, `tools/` | **Apache License 2.0** | [`LICENSE`](LICENSE) |
| Papers, figures, docs, this README | **CC BY 4.0** | [`LICENSE-DOCS`](LICENSE-DOCS) |

Apache rather than MIT for a specific reason. Apache 2.0 § 4(b) requires that
anyone distributing a modified version **carry prominent notices stating that
they changed the files.** For a project whose central lesson was that a
modified claim can travel further than the correction to it, a licence that
makes "say what you changed" a condition of reuse is the one that matches the
work. MIT asks for no such thing.

**If you reuse a result from here, cite the commit.** Reported values in these
papers have been corrected more than once — see
[`REPRODUCTION.md`](docs/papers/hallucination/REPRODUCTION.md) and
[`RTP_NULL_RESULT.md`](docs/papers/non_hopf/RTP_NULL_RESULT.md). A number quoted
without a commit is a number nobody can check, which is the failure this whole
repository is now built to prevent.

Previously this section read "Research Preview … formal licensing will be
finalized alongside publications", and the badge above it linked to a `LICENSE`
file that did not exist. It exists now.

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
