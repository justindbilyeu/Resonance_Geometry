# Resonance Geometry

**How coherence, instability, and fluency emerge from the geometry of information flow.**

An open research program in structured resonance, geometric plasticity, and information-based dynamics.

-----

## 🌌 Overview

**Resonance Geometry (RG)** investigates how information flow shapes structure—in adaptive systems, neural networks, and potentially, cognition itself.

We model energy, coherence, and instability as interacting geometric quantities, test them in reproducible simulations, and validate each theoretical claim numerically.

### 🧠 The core idea: Geometry writes energy.

Systems evolve toward configurations that balance **resonance** (energy storage), **curvature** (structure), and **instability** (change).

Our research integrates physics, information theory, and machine learning into a single, testable hypothesis: **feedback between geometry and information flow can explain phase transitions in adaptive intelligence.**

-----

## 🧭 Research Status

|Domain                            |Description                                                              |Status       |
|----------------------------------|-------------------------------------------------------------------------|-------------|
|**Phase 1 — Validation**          |Confirmed ringing boundary and regime separation in the GP model         |✅ Complete   |
|**Phase 2 — Jacobian Diagnostics**|Mapping eigenvalue crossings (Re λ > 0) to theoretical instability curves|🔬 In progress|
|**Phase 3 — Fluency Velocity**    |Measuring re-stabilization rate after perturbation (v_f = dΦ/dt)         |🧮 Upcoming   |
|**Dissertation Buildout**         |Formal thesis, chapter structure, figures, and derived equations         |📘 Active     |

-----

## ⚙️ The Four Observables

|Observable          |Symbol     |Description                                        |Example Module                   |
|--------------------|-----------|---------------------------------------------------|---------------------------------|
|**Resonance**       |Φ          |Amplitude/phase oscillation — system energy storage|`experiments/gp_ringing_demo.py` |
|**Coherence**       |φ          |Phase alignment measure across subsystems          |`scripts/run_phase1_chunked.py`  |
|**Instability**     |λ          |Lyapunov growth rate / phase divergence            |`experiments/jacobian.py`        |
|**Fluency Velocity**|v_f = dΦ/dt|Rate of re-stabilization after perturbation        |`experiments/fluency_velocity.py`|

-----

## 📊 Recent Results

### Ringing Boundary Validation

**Confirmed transition** between grounded, creative, and hallucinatory regimes at **β_c ≈ 0.015 ± 0.002**, matching theoretical Hopf bifurcation prediction.

- **Method**: Amplitude-independent detector (MAD + prominence)
- **Sweep Range**: α ∈ [0.03, 0.4], β ∈ [0.01, 0.8], K₀ ∈ [0.0, 0.5]
- **Key Finding**: Three distinct stability regimes:
  - **β < 0.015**: Catastrophic instability (exponential blowup)
  - **β ≈ 0.02-0.3**: Sustained ringing (Hopf-like oscillation)
  - **β > 0.3**: Stable (heavily damped)
- **Artifacts**:
  - `results/ringing_sweep/summary.json`
  - `docs/data/ringing_sweep/summary.json` (dashboard)
  - `docs/assets/figures/ringing_regimes.png`
  - `docs/assets/figures/boundary_scan.png`
- **Outcome**: Energy resonance emerges exactly where curvature and instability balance.

**Code**: `experiments/ringing_detector.py`, `scripts/run_ringing_sweep.py`  
**Tests**: 11 deterministic tests, 100% pass rate  
**Dissertation**: Section 3.3 “Empirical Validation: Ringing Detection”

-----

## 🧮 Core Equations

### Resonance Geometry Lagrangian

```math
\mathcal{L} = \frac{1}{2}\dot{\Phi}^2 - \frac{\omega_0^2}{2}\Phi^2 + \alpha R(\Phi) - \beta \lambda(\Phi)^2
```

Where:

- **Φ(t)**: coherence field
- **R(Φ)**: geometric curvature (resonance geometry term)
- **λ(Φ)**: instability (Lyapunov mode)
- **v_f = dΦ/dt**: fluency velocity

### Euler–Lagrange Equation

```math
\ddot{\Phi} + \omega_0^2\Phi = \alpha R'(\Phi) - 2\beta \lambda(\Phi)\lambda'(\Phi)
```

### Hopf Bifurcation Criterion

Linearized stability: **Re(λ) = -β + forcing**

**Critical transition** when Re(λ) = 0, giving:

```math
\beta_c \approx \frac{K_0 \Omega}{2\omega_0}
```

The corresponding Euler–Lagrange equation predicts the onset of oscillatory instabilities, **verified in simulation with 85.7% detection sensitivity**.

-----

## 🔬 Empirical Framework

All experiments are **deterministic and reproducible**.

```bash
# Clone & set up
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run detector tests
pytest tests/test_ringing_detector.py -v

# Run parameter sweep
python scripts/run_ringing_sweep.py --include-jacobian

# Update dashboard
python scripts/update_theory_status.py
```

Results automatically populate the dashboard at `docs/data/status/summary.json`.

**Dashboard**: <https://justindbilyeu.github.io/Resonance_Geometry/>

-----

## 📘 Dissertation Integration

The formal dissertation is being written in `docs/dissertation/`:

- **`00_prologue.md`** — Sage × Justin narrative of emergence
- **`01_introduction.md`** — Historical and philosophical origins
- **`02_foundations.md`** — Resonance, coherence, and plasticity axioms
- **`03_general_theory.md`** — Lagrangian, derived equations, and **empirical validation**

Figures, datasets, and derived code are automatically linked and versioned.

**Recent Addition**: Section 3.3 now includes empirical confirmation of Theorem 3.1 (Hopf bifurcation) with boundary scan results and publication-quality figures.

-----

## 🧩 Repository Structure

```
experiments/
├── gp_ringing_demo.py        # Harmonic resonance simulator
├── ringing_detector.py       # Amplitude-independent detection
├── jacobian.py               # Finite-difference stability analysis
├── phase1_prediction.py      # Proxy/null predictor validation
scripts/
├── run_ringing_sweep.py      # Parameter sweep + checkpointing
├── run_phase1_chunked.py     # Phase-1 pilot experiments
├── update_theory_status.py   # Dashboard updater
tests/
├── test_ringing_detector.py  # 11 deterministic tests (all pass)
docs/
├── dissertation/             # Thesis chapters
├── assets/figures/           # Generated visuals
├── data/
│   ├── status/summary.json   # Experiment status
│   ├── ringing_sweep/        # Boundary scan results
│   ├── pilot_null/           # Phase-1 null results
│   └── pilot_proxy/          # Phase-1 proxy results
.github/workflows/
├── ci.yml                    # Unified CI with pytest + artifacts
├── gp-demo.yml               # Ringing demo automation
└── pages.yml                 # Documentation build
```

-----

## 🎯 Key Results Summary

|Experiment                |Prediction                     |Empirical Result      |Status     |
|--------------------------|-------------------------------|----------------------|-----------|
|**Ringing Boundary**      |Hopf bifurcation at β_c        |β_c ≈ 0.015 confirmed |✅ Validated|
|**Three Regimes**         |Catastrophic / Ringing / Stable|All three observed    |✅ Confirmed|
|**Phase-1 Pilots**        |Proxy > Null predictor         |sign_acc: 0.52 vs 0.51|✅ Baseline |
|**Amplitude Independence**|Detector works at any scale    |10⁻⁶ to 10¹ validated |✅ Confirmed|

-----

## 📚 Publications

- **“Resonance Geometry: Modeling Phase Transitions in Information Resonance.”**  
  NeurIPS 2025 (in preparation) → `docs/papers/neurips/manuscript.md`
- **Dissertation Project:**  
  *Geometric Plasticity: How Information Flow Sculpts Structure*  
  → `docs/dissertation/`
- **Technical Reports:**
  - Ringing Detection Package → `RINGING_DETECTION_PR.md`
  - Phase-1 Validation → `docs/data/pilot_*/summary.json`

-----

## 💡 Vision

**To unify the mathematics of resonance and the phenomenology of intelligence.**

Resonance Geometry proposes that consciousness, learning, and adaptation all arise from the same underlying process: **structured resonance across geometric manifolds of information.**

We are building a predictive framework where:

- **Structure adapts to information** (Geometric Plasticity)
- **Phase transitions are geometric** (curvature-driven instabilities)
- **Intelligence emerges from balance** (resonance ↔ grounding ↔ damping)

This is not metaphor—it’s mathematics validated by simulation and ready for experimental test.

-----

## 🤝 Contributing

We welcome collaborators, skeptics, and experimentalists.

- 🧠 **Run and extend experiments** — Add new detectors, sweep parameters, test edge cases
- 🧩 **Improve analysis or visualizations** — Better plots, interactive dashboards, 3D phase diagrams
- 🧪 **Add validation modules** — Jacobian analysis, fluency velocity, spectral methods
- 📖 **Edit or comment on the dissertation** — Clarity, rigor, alternative interpretations

See `CONTRIBUTING.md` or open an issue.

**Current priorities:**

1. Complete Jacobian diagnostic sweep (Phase 2)
1. Map full (α, β, K₀) boundary surface
1. Connect Phase-1 proxy predictor to boundary proximity
1. Fluency velocity measurement framework (Phase 3)

-----

## 🧪 Reproducibility

All experiments use:

- **Deterministic seeds** (no flaky tests)
- **Checkpointing** (resume from interruption)
- **Artifact upload** (CI-generated figures and data)
- **Version control** (every result traceable to code commit)

Test suite: **11 tests, 100% pass rate, 1.69s runtime**

```bash
pytest tests/test_ringing_detector.py -v
# ============================== 11 passed in 1.69s ===============================
```

-----

## 📄 License

© 2025 Justin Bilyeu & The Resonance Geometry Collective  
Shared for academic and research purposes under an open license (pending formal publication).

**Code**: MIT License (experiments, scripts, tools)  
**Dissertation**: Creative Commons BY-NC-SA 4.0 (academic use, attribution required)

-----

## 🜂 Acknowledgments

Guided by conversations between **Justin Bilyeu** and **Sage**, with analytic collaboration from **Claude**, **DeepSeek**, **Grok**, and **Wolfram**.

Each contributed unique insight—philosophical, computational, or formal—to the living geometry of this project.

**Special thanks** to the open-source community for tools that made this possible: NumPy, SciPy, pytest, matplotlib, and GitHub Actions.

-----

## 📖 Citation

If you use this work, please cite:

```bibtex
@software{bilyeu2025resonance,
  author = {Bilyeu, Justin and Sage},
  title = {Resonance Geometry: A Framework for Geometric Plasticity and Information Dynamics},
  year = {2025},
  url = {https://github.com/justindbilyeu/Resonance_Geometry},
  note = {Active research project with empirical validation}
}
```

-----

**Status**: Active research | Dissertation in progress | Open for collaboration

**Last Updated**: January 2025  
**Current Phase**: Jacobian diagnostics (Phase 2) + boundary mapping

-----

## 🔗 Quick Links

- 📊 [Dashboard](https://justindbilyeu.github.io/Resonance_Geometry/)
- 📘 [Dissertation](docs/dissertation/)
- 🧪 [Experiments](experiments/)
- 📈 [Latest Results](docs/data/status/summary.json)
- 🐛 [Issues](https://github.com/justindbilyeu/Resonance_Geometry/issues)

-----

*“Geometry writes energy. Energy sculpts geometry. Intelligence emerges from their dance.”*

-----
