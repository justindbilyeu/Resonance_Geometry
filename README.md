# Resonance Geometry (RG) & Geometric Plasticity (GP)

[![CI](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml/badge.svg)](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**A computational framework for adaptive networks that formalizes how systems reshape their internal geometry to optimize information flow.**

-----

## Overview

Biological and artificial systems continuously rewire their internal connections to improve function—neurons synchronize, modules emerge, and signals find efficient pathways. **Geometric Plasticity (GP)** provides a testable mathematical model for this process, where a network’s coupling geometry `g` evolves to align with actual information flow `Ī`, while being constrained by complexity, modularity, and fidelity.

The framework centers around a variational potential `V(g; Ī)`:

```
V(g; Ī) = -Īᵀg + (λ/2)‖g‖² + (β/2)gᵀLg + (A/2)‖Ī - I(g,t)‖²
```

governed by gradient-flow dynamics `ġ = -η∇gV`. Here:

- `I(g,t)` is measured information (e.g., windowed mutual information)
- `L` is a Laplacian encouraging smooth or modular structure
- `λ, β, A` control penalties for complexity, structure, and tracking fidelity

This repository focuses on empirical predictions and reproducible experiments—bridging theory with measurable dynamics.

-----

## Current Status (September 2025)

- ✅ **Pre-registered prediction P1:** Demonstrated threshold-triggered rise in alpha-band mutual information and hysteresis under parameter sweeps
- ✅ **Functional synthetic demo:** `gp_ringing_demo.py` generates time-series, hysteresis curves, and structured results
- ✅ **Rigor-enforced testing:** Fixed parameters, seeded RNG, surrogate testing, and multiple-comparisons correction
- 🧪 **Next step:** Replication on small EEG datasets using a locked analysis protocol

*Note: Earlier cosmological analogies are deprecated. Focus is on the GP variational principle and measurable dynamics.*

-----

## Quick Start

```bash
# Clone the repository
git clone https://github.com/justindbilyeu/Resonance_Geometry
cd Resonance_Geometry

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r experiments/requirements.txt

# Run the core demonstration
python experiments/gp_ringing_demo.py
```

**Outputs** (saved to `results/gp_demo/`):

- `mi_timeseries.png` – Mutual information over time with alpha-band emphasis
- `lambda_schedule.png` – Coupling parameter sweep protocol
- `hysteresis_curve.png` – Alpha-band power vs. λ (showing hysteresis)
- `summary.json` – Quantitative results (λ*, loop area, p-values)

-----

## Predictions (v1.2)

|ID    |Prediction                    |Description                                                                       |
|------|------------------------------|----------------------------------------------------------------------------------|
|**P1**|Ringing threshold & hysteresis|Sharp increase in MI power at critical λ*, with hysteresis under sweeps           |
|**P2**|Drive–timescale matching      |Maximal response when external drive matches intrinsic timescale                  |
|**P3**|Motif selection               |Structural preference for broadcast vs. modular motifs under different constraints|

**Full details:** [`docs/predictions.md`](docs/predictions.md)

-----

## Experimental Rigor

This project emphasizes reproducibility and falsifiability:

- **Locked analysis parameters:** Pre-registered window sizes, frequency bands, and estimators
- **Surrogate testing:** Using IAAFT/AR null models to preserve temporal structure
- **Multiple-comparisons control:** Corrected significance testing
- **Blinding & publish-on-fail:** Predefined success criteria and avoidance of p-hacking

**Pre-registration plan:** [`docs/prereg_P1.md`](docs/prereg_P1.md)

-----

## Repository Lineage

Resonance_Geometry now consolidates work that previously lived in three standalone projects:

- [resonance-geometry-docs](https://github.com/justindbilyeu/resonance-geometry-docs) – long-form whitepapers, preregistrations, and philosophy notes.
- [gp-simulations](https://github.com/justindbilyeu/gp-simulations) – Python prototypes for ringing, hysteresis, and motif sweeps.
- [resonance-figures-archive](https://github.com/justindbilyeu/resonance-figures-archive) – published plots, figure templates, and design assets.

The migration timeline and release notes are tracked in [`docs/history/HISTORY.md`](docs/history/HISTORY.md).

-----

## Repository Structure

```
Resonance_Geometry/
├── docs/                        # Whitepapers, preregistrations, philosophy, and history
│   └── history/HISTORY.md       # Migration and release timeline
├── simulations/
│   ├── README.md                # Simulation entry points
│   └── gp_ringing_demo.py       # Main ringing + hysteresis demonstration
├── figures/                     # Published plots and reusable figure templates
├── archive/                     # Snapshots of legacy analyses and datasets
├── results/                     # Generated outputs (not in version control)
└── .github/workflows/           # CI configuration
```

-----

## ITPU Integration

This project leverages the [Information-Theoretic Processing Unit (ITPU)](https://github.com/justindbilyeu/ITPU) for high-performance estimation of mutual information, transfer entropy, and other information dynamics. Python reference implementations are provided, with future support for real-time ITPU hardware acceleration.

-----

## Contributing

We welcome contributions in the following areas:

- Replication attempts with novel random seeds
- New surrogate data implementations (AR, IAAFT)
- Pilot studies using public EEG/MEG datasets
- Documentation improvements and typo fixes

Please open an issue before submitting a pull request to coordinate efforts.

-----

## License

Apache 2.0 – see <LICENSE> for details.
