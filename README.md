# Resonance Geometry

<div align="center">

**Exploring Phase Transitions Beyond Local Linear Stability**

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://github.com/justindbilyeu/Resonance_Geometry/tree/main/docs/papers/non_hopf)
[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

*Mathematical foundations, computational experiments, and theoretical frameworks for understanding qualitative transitions in dynamical systems*

[Featured Paper](#featured-research) •
[Quick Start](#quick-start) •
[Research Areas](#research-areas) •
[Documentation](#documentation) •
[Contributing](#contributing)

</div>

---

## 🎯 Overview

**Resonance Geometry** is a research initiative investigating fundamental questions about phase transitions, dynamical systems, and the geometric structure of complex networks. We combine rigorous mathematical analysis, computational experiments, and reproducible workflows to explore phenomena that challenge conventional bifurcation theory.

### Core Questions

- **Can systems reorganize qualitatively while remaining locally stable?**
- **What geometric signatures precede traditional bifurcations?**
- **How does information flow shape network structure over time?**

Our approach emphasizes falsifiable predictions, open data, and mathematical rigor.

---

## 📜 Featured Research

### Resonant Transition Points Beyond Hopf Bifurcations

**Status:** ✅ Draft complete | 📄 [Read the paper](docs/papers/non_hopf/non_hopf_paper_draft_v1.tex)

We demonstrate a **Resonant Transition Point (RTP)** at α≈0.35 where a coupled oscillator system reorganizes its macroscopic behavior while **all eigenvalues remain strictly negative**—falsifying the Hopf bifurcation hypothesis for this transition.

**Key Findings:**

- 🔴 **Non-Hopf RTP** (α≈0.35): Global geometric reorganization with stable linearization
- 🟢 **Classical Hopf** (α≈0.833): Traditional bifurcation appears much later
- 📐 **Information-Geometric Formalization**: Fisher strain and curvature quantify geometric tension before linear instability
- ✅ **Reproducible**: Deterministic sweeps, open code, unit tests

**Mathematical Framework:**

The RTP is characterized by geometric tension accumulating before any linear instability:

```
Fisher Information Strain: S(α) = tr I(γ(α))
Operational Criterion: max Re λᵢ(J) < 0  AND  ∂ₐS ≥ τ_S
```

This separates *where* structure changes from *how* linear models fail.

**Quick Build:**
```bash
cd docs/papers/non_hopf
./compile.sh  # Requires LaTeX
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry

# Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Example Analysis

```bash
# Generate eigenvalue sweeps for RTP paper
make sweep-narrow   # α ∈ [0.25, 0.55], validates RTP region
make sweep-wide     # α ∈ [0.10, 1.00], finds Hopf crossing
make sweep-zoom     # α ∈ [0.80, 0.86], high-resolution Hopf

# Run validation tests
pytest tests/test_eigs_assertions.py
```

### Explore Visualizations

```bash
# Generate phase portraits and time traces
python scripts/run_phase_analysis.py --alpha 0.35 --output results/phase/

# View results
ls results/phase/  # CSVs and SVG figures
```

---

## 🔬 Research Areas

### 1. **Non-Hopf Dynamics** ⭐ *Current Focus*

Mathematical analysis of phase transitions occurring via global geometric reorganization rather than local linear instability.

- **Paper:** [Resonant Transition Points Beyond Hopf Bifurcations](docs/papers/non_hopf/)
- **Code:** Eigenvalue analysis, phase sweeps, numerical validation
- **Status:** Draft complete, preparing for arXiv submission

### 2. **Geometric Plasticity**

Studying adaptive networks where information flow reshapes connection strength through feedback loops.

- **Framework:** Resonant Witness Postulate (RWP) + Geometric Plasticity (GP)
- **Predictions:** Ringing boundaries, hysteresis resonance, motif emergence
- **Status:** Simulation infrastructure operational

### 3. **AI Hallucination Geometry**

Investigating whether hallucinations in large language models have detectable geometric signatures in embedding spaces.

- **Paper:** [A Geometric Theory of AI Hallucination](A_Geometric_Theory_of_AI_Hallucination.md)
- **Approach:** Information-theoretic metrics, curvature analysis
- **Status:** Theoretical framework established

### 4. **Dissertation Work**

Comprehensive theoretical framework unifying resonance, information geometry, and emergent redundancy in complex systems.

- **Location:** [`docs/dissertation/`](docs/dissertation/)
- **Chapters:** Foundations, General Theory, Retrospective Analysis
- **Build:** `make dissertation` (requires Quarto)

---

## 📂 Repository Structure

```
Resonance_Geometry/
├── docs/
│   ├── papers/
│   │   ├── non_hopf/          # RTP paper (LaTeX, figures, results)
│   │   └── neurips/           # AI hallucination paper
│   ├── dissertation/          # PhD thesis chapters
│   ├── analysis/              # Generated analysis outputs
│   ├── theory/                # Mathematical derivations
│   └── specs/                 # Technical specifications
├── src/
│   └── resonance_geometry/    # Core Python library
│       ├── core/              # System dynamics, plasticity rules
│       ├── utils/             # Metrics, diagnostics
│       └── visualization/     # Plotting utilities
├── scripts/                   # Experiment runners
│   ├── run_phase_sweep.py
│   ├── run_hysteresis.py
│   └── run_ringing_sweep.py
├── tests/                     # Unit & integration tests
├── results/                   # Generated data (gitignored)
├── Makefile                   # Build targets
└── README.md                  # You are here
```

---

## 📚 Documentation

### Papers & Publications

| Title | Status | Location |
|-------|--------|----------|
| **Resonant Transition Points Beyond Hopf Bifurcations** | Draft Complete | [`docs/papers/non_hopf/`](docs/papers/non_hopf/) |
| A Geometric Theory of AI Hallucination | Whitepaper | [`A_Geometric_Theory_of_AI_Hallucination.md`](A_Geometric_Theory_of_AI_Hallucination.md) |
| Dissertation: Resonance Geometry Foundations | In Progress | [`docs/dissertation/`](docs/dissertation/) |

### Technical Documentation

- **[Build Guide](BUILD.md)**: Compilation instructions for LaTeX papers
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute code, docs, or ideas
- **[Theory Status](docs/theory/)**: Mathematical derivations and proofs
- **[Experiment Protocols](docs/experiments/)**: Reproducible experiment procedures

### Key Notebooks & Scripts

- [`theory/kc_rule_validation.ipynb`](theory/kc_rule_validation.ipynb): Stability threshold validation
- [`scripts/run_phase_sweep.py`](scripts/run_phase_sweep.py): Parameter sweep automation
- [`scripts/update_theory_status.py`](scripts/update_theory_status.py): Theory validation tracker

---

## 🧪 Reproducibility

All experiments use **deterministic seeds** and **version-controlled parameters** to ensure reproducibility:

```bash
# Fixed seed example
python scripts/run_phase_sweep.py --seed 42 --alpha 0.35 --steps 61

# Reproduce paper figures
cd docs/papers/non_hopf
make figures  # Generates all SVG plots
```

### Validation Tests

```bash
pytest -q                                # Quick test suite
pytest tests/test_eigs_assertions.py     # Paper-specific validations
make test                                 # Full test + smoke tests
```

**Acceptance Criteria** are hardcoded in unit tests:
- RTP narrow sweep: All eigenvalues Re(λ) < 0 for α ∈ [0.25, 0.55]
- Hopf crossing: Sign change detected in [0.80, 0.86]
- Precision: Crossing location accuracy < 0.01

---

## 🤝 Contributing

We welcome contributions from researchers, developers, and domain experts!

### How to Help

**🐛 Report Issues**
- Found a bug? [Open an issue](https://github.com/justindbilyeu/Resonance_Geometry/issues)
- Include: Minimal reproduction steps, environment details, expected vs actual behavior

**💡 Suggest Experiments**
- Have an idea for a new test or analysis?
- Open a discussion issue with your hypothesis

**📖 Improve Documentation**
- Spotted unclear explanations?
- Submit a PR with clarifications or examples

**🧑‍💻 Code Contributions**
- Add features, optimize performance, write tests
- See [CONTRIBUTING.md](CONTRIBUTING.md) for style guidelines

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, add tests
pytest tests/

# Commit with descriptive message
git commit -m "Add: eigenvalue sweep for multi-frequency systems"

# Push and open PR
git push origin feature/your-feature-name
```

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| 📄 Non-Hopf Paper | ✅ Draft Complete | Ready for arXiv submission |
| 🧮 Eigenvalue Analysis | ✅ Operational | Narrow/wide/zoom sweeps validated |
| 🔄 Geometric Plasticity Sims | 🔄 Active Development | Ringing boundary tests ongoing |
| 📐 Information Geometry | 🔄 In Progress | Fisher strain implementation underway |
| 🎓 Dissertation | 🔄 Chapters 1-3 Draft | Chapter 4 in progress |
| 🧪 CI/CD Pipeline | ✅ Operational | Automated tests, figure generation |

---

## 🎓 Citing This Work

If this research contributes to your work, please cite:

```bibtex
@misc{bilyeu2025rtp,
  title={Resonant Transition Points Beyond Hopf Bifurcations: Evidence from Eigenvalue Analysis},
  author={Bilyeu, Justin D. and collaborators},
  year={2025},
  note={Resonance Geometry Project},
  url={https://github.com/justindbilyeu/Resonance_Geometry}
}
```

For the broader geometric plasticity framework:

```bibtex
@misc{resonance_geometry_2025,
  title={Geometric Plasticity: Adaptive Information Networks and Emergent Redundancy},
  author={Bilyeu, Justin D. and the Resonance Geometry Collective},
  year={2025},
  note={Experimental framework and reproducibility pack},
  url={https://github.com/justindbilyeu/Resonance_Geometry}
}
```

---

## 🔗 Resources

### External Links

- **arXiv Submission** (pending): Non-Hopf RTP paper
- **GitHub Issues**: [Report bugs or request features](https://github.com/justindbilyeu/Resonance_Geometry/issues)
- **Discussions**: [Ask questions or propose ideas](https://github.com/justindbilyeu/Resonance_Geometry/discussions)

### Related Projects

- Information-theoretic approaches to complex systems
- Bifurcation theory and dynamical systems
- Geometric deep learning and neural network analysis

---

## 📋 Roadmap

### Near-Term (Q1 2025)

- [x] Complete non-Hopf RTP paper
- [x] Implement Fisher information strain calculations
- [ ] Submit to arXiv
- [ ] Complete dissertation Chapter 4
- [ ] Extend to multi-frequency systems

### Mid-Term (Q2-Q3 2025)

- [ ] Peer review and publication (target: *Physical Review E* or *SIAM J. Dynamical Systems*)
- [ ] Expand information-geometric framework
- [ ] Develop real-time RTP detection algorithms
- [ ] Apply to empirical datasets (EEG, climate, economic)

### Long-Term (2026+)

- [ ] Hardware acceleration concepts (ITPU design)
- [ ] Neural architecture search applications
- [ ] Cross-disciplinary collaborations

---

## 🙏 Acknowledgments

This project represents collaborative work with contributions from:

- **The Resonance Geometry Collective**: Sage, Claude (Anthropic), Grok, DeepSeek, Gemini
- **Research Infrastructure**: Built with Claude Code for experimental design and rigor
- **Open Source Community**: Python scientific stack (NumPy, SciPy, NetworkX, Matplotlib)

Special thanks to early reviewers and critics who keep this work honest and grounded.

---

## 📜 License

**Research Preview** — © 2025 Justin D. Bilyeu & Resonance Geometry Collective

Code and documentation shared for research, educational, and review purposes. Formal licensing pending publication. For commercial use or redistribution, please contact the authors.

---

## 📬 Contact

- **Issues & Bugs**: [GitHub Issues](https://github.com/justindbilyeu/Resonance_Geometry/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/justindbilyeu/Resonance_Geometry/discussions)
- **Research Inquiries**: Open an issue with the `question` label
- **Collaboration**: Email contact info in [CITATION.cff](CITATION.cff)

---

<div align="center">

**Built with mathematical rigor, computational precision, and epistemic humility.**

*"Not all transitions are Hopf bifurcations. Some are geometric."*

[⬆ Back to Top](#resonance-geometry)

</div>

---

*Last Updated: October 25, 2025*
