```markdown
# Resonance Geometry (RG) & Geometric Plasticity (GP)

[![CI](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml/badge.svg)](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**A testable framework for adaptive systems that tracks how coupling geometry changes with information flow.**

---

## Overview

Many systems "re-wire" themselves as they learn: neurons synchronize, modules form, signals pass through preferred pathways. **Geometric Plasticity (GP)** models this by letting a network's coupling geometry `g` evolve to align with measured information flow `Ī` while paying penalties for complexity and delay.

**The GP potential:**
```

V(g; Ī) = -Īᵀg + (λ/2)‖g‖² + (β/2)gᵀLg + (A/2)‖Ī - I(g,t)‖²

```
with gradient-flow dynamics `ġ = -η∇gV`. Here, `I(g,t)` is measured information (e.g., windowed mutual information), `L` encourages smooth structure, and `λ,β,A` trade off simplicity, modularity, and fidelity.

**Focus:** Empirical, falsifiable predictions with end-to-end runnable code.

---

## Status (Sept 2025)

- ✅ **Pre-registered prediction (P1):** Sharp rise in alpha-band MI power at coupling `λ*` with hysteresis under sweeps
- ✅ **Runnable synthetic demo:** `gp_ringing_demo.py` produces MI time-series, hysteresis curves, and JSON summaries  
- ✅ **Reproducibility safeguards:** Fixed parameters, seeded RNG, surrogate nulls, multiple-comparisons control
- 🧪 **Next:** Replication on small EEG datasets using locked analysis protocol

*Note: Earlier cosmological analogies are marked non-essential. Core focus is GP variational model + measurable information dynamics.*

---

## Quick Start

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

# Run demo
python experiments/gp_ringing_demo.py
```

**Outputs** (in `results/gp_demo/`):

- `mi_timeseries.png` - MI over time with alpha-band highlighting
- `lambda_schedule.png` - Coupling parameter sweep
- `hysteresis_curve.png` - Alpha-band MI power vs λ (up vs down)
- `summary.json` - Metrics (λ*, loop area, p-values)

-----

## Predictions (v1.2)

|Prediction                             |Description                                        |
|---------------------------------------|---------------------------------------------------|
|**P1** - Ringing threshold & hysteresis|Sharp MI power rise at λ* with hysteresis loop     |
|**P2** - Drive–timescale matching      |Max response when drive matches intrinsic timescale|
|**P3** - Motif selection               |Geometry prefers broadcast vs modular motifs       |

**Details:** [`docs/predictions.md`](docs/predictions.md)

-----

## Experimental Rigor

**Preregistration & safeguards:**

- Locked parameters (window size, alpha band, estimators)
- Surrogate nulls (IAAFT/AR) preserving temporal structure
- Multiple-comparisons control
- Blinding and publish-on-fail criteria

**Full plan:** [`docs/prereg_P1.md`](docs/prereg_P1.md)

-----

## Repository Structure

```
Resonance_Geometry/
├── experiments/
│   ├── gp_ringing_demo.py    # Main demonstration
│   └── requirements.txt      # Python dependencies
├── docs/
│   ├── predictions.md        # Detailed predictions
│   └── prereg_P1.md         # Experimental protocol
├── results/                  # Generated outputs
└── .github/workflows/        # Continuous integration
```

-----

## ITPU Integration

This project uses the [Information-Theoretic Processing Unit (ITPU)](https://github.com/justindbilyeu/ITPU) for accelerated MI/CMI/TE measurements. Current Python baselines will later run on ITPU hardware for real-time experiments.

-----

## Contributing

We welcome:

- Replication PRs with different seeds
- Surrogate implementations (AR/IAAFT)
- EEG pilot studies with public datasets
- Documentation improvements

Please open an Issue first for coordination.

-----

## License

Apache 2.0 - See <LICENSE> for details.

```

```
