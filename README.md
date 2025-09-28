# Resonance Geometry (RG) & Geometric Plasticity (GP)

[![CI](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml/badge.svg)](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**A testable framework for adaptive systems that tracks how coupling geometry changes with information flow.**

*Where philosophy meets physics through falsifiable mathematics.*

-----

## Overview

Many systems “re-wire” themselves as they learn: neurons synchronize, modules form, signals pass through preferred pathways. **Geometric Plasticity (GP)** models this by letting a network’s coupling geometry `g` evolve to align with measured information flow `Ī` while paying penalties for complexity and delay.

**The GP potential:**

```
V(g; Ī) = -Īᵀg + (λ/2)‖g‖² + (β/2)gᵀLg + (A/2)‖Ī - I(g,t)‖²
```

with gradient-flow dynamics `ġ = -η∇gV`. Here, `I(g,t)` is measured information (e.g., windowed mutual information), `L` encourages smooth structure, and `λ,β,A` trade off simplicity, modularity, and fidelity.

**Focus:** Empirical, falsifiable predictions with end-to-end runnable code.

-----

## Framework Architecture

### 🔬 **Empirical Engine** (Core Science)

- **Geometric Plasticity**: Mathematical framework linking information flow to network geometry
- **Pre-registered predictions**: Testable hypotheses about coupling dynamics and resonance
- **Statistical rigor**: Surrogate controls, multiple comparisons correction, effect size quantification
- **Reproducible pipelines**: Locked parameters, seeded RNG, comprehensive validation

### 🌀 **Philosophical Lattice** (Inspirational Framework)

- **Resonance Axioms**: Conceptual foundations linking information, structure, emotion, and coherence
- **Epistemic boundaries**: Clear separation between [TESTABLE-HYPOTHESIS], [MATHEMATICAL-METAPHOR], and [SPECULATIVE-THEORY]
- **Bridge framework**: Explicit mappings between philosophical concepts and measurable quantities
- **Collaborative vision**: Space for both rigorous empiricism and speculative exploration

**Key Principle**: *Keep the vision and the science parallel but bridged - never conflated.*

-----

## Status (Sept 2025)

### ✅ **Validated (Ready for Replication)**

- **P1 - Alpha resonance**: Sharp rise in alpha-band MI power at coupling `λ*` with hysteresis
- **Synthetic validation**: `gp_ringing_demo.py` produces reproducible MI dynamics and hysteresis curves
- **Statistical framework**: Surrogate controls validate effects beyond chance levels
- **Multi-frequency extension**: GP analysis across neurophysiological frequency bands

### 🧪 **Testing (Current Sprint)**

- **Hysteresis characterization**: Loop area, width, asymmetry as “memory” measures
- **Cross-frequency coupling**: Hierarchical information flow across frequency bands
- **EEG validation pilot**: Testing GP predictions on small neural datasets
- **Bridge validation**: Correlating GP measures with established emotional/cognitive metrics

### 🎯 **Planned (Next Quarter)**

- **P2 - Drive-timescale matching**: Optimal response when drive matches intrinsic timescales
- **P3 - Motif selection**: Geometry preferences for broadcast vs modular architectures
- **Consciousness correlation studies**: GP fixed points and reportable conscious experiences
- **Therapeutic applications**: Emotional regulation via information-geometry coupling

*Note: Earlier cosmological analogies are marked non-essential. Core focus remains GP variational model + measurable information dynamics.*

-----

## Quick Start

### Basic Demo

```bash
# Clone and setup
git clone https://github.com/justindbilyeu/Resonance_Geometry
cd Resonance_Geometry

# Create virtual environment
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r experiments/requirements.txt

# Run basic demo
python experiments/gp_ringing_demo.py
```

## Experiments

- Topological Constraint Test — our 1919 eclipse moment. [docs/experiments/Topological_Constraint_Test.md]

### Multi-Frequency Analysis

```bash
# Run extended analysis across frequency bands
python experiments/gp_ringing_demo.py --multi-frequency

# Outputs include:
# - Standard P1 results (alpha band)
# - Cross-frequency coupling matrices
# - Band-specific lambda* thresholds
# - Statistical validation results
```

**Outputs** (in `results/gp_demo/`):

- `mi_timeseries.png` - MI over time with alpha-band highlighting
- `lambda_schedule.png` - Coupling parameter sweep
- `hysteresis_curve.png` - Alpha-band MI power vs λ (up vs down)
- `summary.json` - Core metrics (λ*, loop area, p-values)
- `multi_frequency_results.json` - Cross-band analysis
- `multi_frequency_validation.json` - Statistical controls

-----

## Predictions & Validation

|Prediction                               |Status     |Description                                        |
|-----------------------------------------|-----------|---------------------------------------------------|
|**P1** - Ringing threshold & hysteresis  |✅ Validated|Sharp MI power rise at λ* with hysteresis loop     |
|**P1-MF** - Frequency-specific thresholds|🧪 Testing  |Different λ* values across frequency bands         |
|**P2** - Drive–timescale matching        |📋 Planned  |Max response when drive matches intrinsic timescale|
|**P3** - Motif selection                 |📋 Planned  |Geometry prefers broadcast vs modular motifs       |

**Details**: See <docs/predictions.md>

### Preregistration & Safeguards

- **Locked parameters**: Window size, frequency bands, estimators
- **Surrogate nulls**: IAAFT/AR preserving temporal structure
- **Multiple-comparisons control**: FDR correction across frequency bands
- **Blinding protocols**: Prevent cherry-picking of results
- **Publish-on-fail criteria**: Null results are scientifically valuable

**Full protocol**: <docs/prereg_P1.md>

-----

## Framework Bridges

### Axiom 1: “Resonance is Information”

- **🔗 GP Connection**: Direct mathematical correspondence in coupling term `-Īᵀg`
- **📊 Status**: [TESTABLE-HYPOTHESIS] - Core GP principle with validated predictions
- **🧪 Tests**: Lambda threshold analysis, information-geometry evolution

### Axiom 2: “Structure Follows Flow”

- **🔗 GP Connection**: Gradient flow `ġ = -η∇gV` directly implements this principle
- **📊 Status**: [TESTABLE-HYPOTHESIS] - Multiple experimental confirmations
- **🧪 Tests**: Coupling adaptation, motif emergence, plasticity dynamics

### Axiom 3: “Emotion is Curvature”

- **🔗 GP Connection**: Hysteresis characteristics as measurable “emotional memory”
- **📊 Status**: [MATHEMATICAL-METAPHOR] - Promising correlations, needs validation
- **🧪 Tests**: Loop area vs emotional persistence, curvature vs affective states

### Axiom 4: “Collapse is Coherence”

- **🔗 GP Connection**: Information integration leading to geometric stability
- **📊 Status**: [SPECULATIVE-THEORY] - Requires significant conceptual development
- **🧪 Tests**: Coherence thresholds, fixed-point emergence, consciousness correlations

**Bridge Documentation**: <docs/frameworks/bridges.md>

-----

## Repository Structure

```
Resonance_Geometry/
├── experiments/
│   ├── gp_ringing_demo.py              # Main GP demonstration
│   ├── multi_frequency_extensions.py   # Cross-frequency analysis
│   └── requirements.txt                # Python dependencies
├── docs/
│   ├── predictions.md                  # Detailed predictions & hypotheses
│   ├── prereg_P1.md                   # Pre-registration protocol
│   ├── frameworks/
│   │   └── bridges.md                 # Axiom-GP bridge mappings
│   ├── experiments/
│   │   ├── Hysteresis_Analysis.md     # Loop characterization methods
│   │   └── Multi_Frequency_Validation.md  # Statistical controls
│   └── specs/
│       └── Multi_Frequency_Plan.md    # Cross-frequency roadmap
├── results/                           # Generated outputs & analysis
├── .github/workflows/                 # Continuous integration
└── README.md                         # This file
```

-----

## Hardware Integration

This project uses the [Information-Theoretic Processing Unit (ITPU)](https://github.com/justindbilyeu/ITPU) for accelerated MI/CMI/TE measurements. Current Python baselines will later run on ITPU hardware for real-time experiments.

**Pipeline**: Python prototyping → ITPU acceleration → Real-time neural interfaces

-----

## Contributing

We welcome contributions across the empirical-philosophical spectrum:

### 🔬 **For Empirical Developers**

- **Start here**: Extend P1 analysis, implement multi-frequency validation
- **Focus**: Strong bridges (Axioms 1-2), reproducible experiments
- **Standards**: Maintain preregistered protocols, use surrogate controls

### 🌀 **For Philosophical Contributors**

- **Start here**: Operationalize Axioms 3-4 with measurable proxies
- **Focus**: Bridge speculative concepts to testable hypotheses
- **Standards**: Use epistemic status tags, propose falsifiable predictions

### 🌉 **For Bridge Builders**

- **Start here**: Design minimal experiments linking concepts to GP measures
- **Focus**: Emotion-hysteresis correlations, consciousness-coherence studies
- **Standards**: Pre-register hypotheses, include falsification criteria

**Contribution Process**:

1. Open an Issue for coordination
1. Maintain epistemic boundaries ([TESTABLE] vs [METAPHOR] vs [SPECULATIVE])
1. Include appropriate statistical controls
1. Design for replication and falsification

### **Welcome Contributions**:

- Replication PRs with different seeds/datasets
- Enhanced surrogate implementations (AR/IAAFT variants)
- EEG pilot studies with public datasets
- Documentation improvements and clarity enhancements
- Bridge validations linking GP measures to established psychological metrics

-----

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{resonance_geometry_2025,
  title={Resonance Geometry: A Testable Framework for Information-Geometry Coupling},
  author={[Your Name/Organization]},
  year={2025},
  url={https://github.com/justindbilyeu/Resonance_Geometry},
  version={0.1.0}
}
```

-----

## License

Apache 2.0 - See <LICENSE> for details.

-----

## Resonant Check

**Clarity**: Empirical framework and philosophical inspiration clearly distinguished with explicit epistemic boundaries.

**Coherence**: GP mathematics bridges measurable information dynamics with visionary concepts through validated experimental protocols.

**Buildability**: Multiple entry points for contributors at different levels, from rigorous replication to speculative bridge-building, all maintaining scientific standards.

*“Where philosophy meets physics through falsifiable mathematics.”*
