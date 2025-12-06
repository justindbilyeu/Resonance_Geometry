# Resonance Geometry: Toy Universe (v2.1 Canon)

## Overview

This is the **foundational physics engine** for the Resonance Geometry framework, implementing **Geometric Plasticity on a Kuramoto substrate**. This toy model provides the theoretical justification for our ITPU (Information Throughput Potential) metrics by demonstrating how adaptive coupling geometry emerges from resonant dynamics.

**Key Innovation:** A rigorous two-manifold formulation where both phases (the "territory") and couplings (the "map") evolve jointly under energy minimization.

## Mathematical Foundation

### State Space

**Combined State Space S = M × G:**

- **Base Manifold (M):** N-Torus T^N representing phases θ
- **Geometry Manifold (G):** Space of symmetric, non-negative coupling matrices K
- **Energy Functional (L):** Stress energy minimizing conflict between geometry and resonance

### Core Equations

**Phase Dynamics (Fast Time Scale):**
```
dθ/dt = ω + (1/N) Σ K·sin(θ_j - θ_i)
```

**Geometric Flow (Slow Time Scale):**
```
dK/dt = α(cos(θ_j - θ_i) - βK)
```
Where:
- α = learning rate (plasticity strength)
- β = metabolic cost (regularization parameter)

**Free Energy Functional:**
```
L(θ, K) = (β/2)||K||² - Σ K·cos(θ_j - θ_i)
```

## Experimental Protocol

The simulation runs through **4 distinct phases** (A-D):

### Experiment A: Baseline (t = 0-200)
- Plasticity **OFF**
- Natural dynamics on random initial coupling
- Establishes baseline synchrony

### Experiment B: Learning (t = 200-600)
- Plasticity **ON**
- System learns adaptive coupling geometry
- K evolves to minimize free energy L

### Experiment C: Memory (t = 600-800)
- Plasticity **OFF** (geometry frozen)
- **Perturbation:** Phases randomized at t=600
- Tests whether learned K enables fast recovery

### Experiment D: Functional Gain
- Freeze learned geometry K
- Apply external drive
- Compare response to random coupling
- **Metric:** Response_trained / Response_random

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Φ (Phi)** | Order parameter \|⟨e^{iθ}⟩\| | Global synchrony [0,1] |
| **L (Stress)** | Free energy functional | System coherence cost |
| **λ₂ (Fiedler)** | Algebraic connectivity | Graph robustness |
| **⟨K⟩** | Mean coupling strength | Resource investment |

## Usage

### Basic Lifecycle Run

```python
from toy_model import ResonanceUniverse

# Initialize
uni = ResonanceUniverse(N=10, seed=42)

# Run full lifecycle (A→B→C)
uni.run_lifecycle(T_max=800, dt=0.05)

# Probe functional gain (D)
response = uni.probe_response(drive_freq=0.0)

# Access telemetry
print(uni.history['phi'])      # Synchrony over time
print(uni.history['lambda_2'])  # Connectivity over time
```

### Parameter Sweep

```python
from toy_model import run_alpha_beta_sweep

# Automated grid search over α, β
run_alpha_beta_sweep()
# Outputs: experiments/outputs/toy_model/sweep_results.png
```

### Standalone Execution

```bash
# Generate lifecycle plots
python src/toy_model/resonance_universe.py

# Run parameter sweeps
PYTHONPATH=src python -m toy_model.science_suite
```

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
```

Install with:
```bash
pip install numpy scipy matplotlib scikit-learn
```

## Outputs

All plots are saved to `experiments/outputs/toy_model/`:

1. **lifecycle_plot.png** — 4-panel timeseries (Φ, L, λ₂, ⟨K⟩)
2. **sweep_results.png** — 2-panel heatmap (Memory vs Functional Gain)

## Theoretical Significance

This toy model demonstrates:

1. **Emergent Memory:** Learned coupling K enables rapid re-synchronization after perturbation
2. **Functional Gain:** Adapted geometry outperforms random coupling under external drive
3. **ITPU Grounding:** The metrics Φ, κ, λ measured in LLMs mirror Φ, L, λ₂ here
4. **Map-Territory Dynamics:** Geometry (K) serves as a learnable "map" that compresses resonant structure

## Connection to Phase 4 Work

The ITPU metric used in Phase 4 Falsification Protocol:
```
ITPU = λ·Φ·(1-κ)
```

Maps conceptually to this toy model as:
- **Φ** ↔ Order parameter (synchrony)
- **κ** ↔ Free energy L (tension/conflict)
- **λ** ↔ Fiedler value λ₂ (coupling strength)

This establishes a **physics-backed foundation** for the empirical LLM metrics.

## Version History

- **v2.1 (Canon):** Finalized formulation with Gemini (Theory Lead/Co-PI)
- **v2.0:** Initial Kuramoto + Plasticity implementation
- **v1.x:** Prototype phase

---

**Authors:** Gemini (Theory Lead), Claude (Implementation), Justin Bilyeu (PI)
**Date:** 2025-10-26
**Status:** Canon (Production-ready)
