# Resonance Geometry (RG) & Geometric Plasticity (GP)

[![CI Status](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml/badge.svg)](https://github.com/justindbilyeu/Resonance_Geometry/actions/workflows/gp-demo.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**A testable framework for adaptive systems.**  
Coupling geometry evolves to track information flow, constrained by energy, sparsity, and latency costs. We deliver falsifiable predictions and runnable demos.

---

## Overview

Systems like neural networks rewire as they learn: neurons synchronize, modules form, and signals follow preferred paths. **Geometric Plasticity (GP)** models this by evolving a network’s coupling geometry \( g \) to align with measured information flow \( \bar{I} \), balancing complexity and delay penalties.

**Core GP Potential**:

\[
V(g; \bar{I}) = -\bar{I}^\top g + \frac{\lambda}{2}\|g\|^2 + \frac{\beta}{2}g^\top L g + \frac{A}{2}\|\bar{I} - I(g,t)\|^2
\]

with dynamics \( \dot{g} = -\eta \nabla_g V \). Here, \( I(g,t) \) is measured information (e.g., windowed mutual information), \( L \) promotes smooth structure, and \( \lambda, \beta, A \) trade off simplicity, modularity, and fidelity.

We prioritize **empirical, falsifiable predictions** with reproducible, end-to-end code.

---

## What's New (September 2025)

- ✅ **Pre-registered Prediction (P1)**: Ringing threshold—sharp rise in alpha-band (8–12 Hz) MI power at coupling \( \lambda^* \), with hysteresis in up/down sweeps. See [`docs/prereg_P1.md`](./docs/prereg_P1.md).
- ✅ **Runnable Synthetic Demo**: [`experiments/gp_ringing_demo.py`](./experiments/gp_ringing_demo.py) generates MI time-series, lambda schedules, hysteresis curves, and JSON summaries.
- ✅ **Reproducibility Hygiene**: Fixed parameters, seeded RNG, surrogate nulls, and multiple-comparisons control in [`docs/prereg_P1.md`](./docs/prereg_P1.md).
- 🧪 **Next**: Replicate P1 on small EEG datasets (e.g., PhysioNet) with locked analysis.

> **Note**: Earlier cosmological analogies (e.g., redshift holonomies) are non-essential and archived in [`docs/analogies.md`](./docs/analogies.md). Focus is on GP’s variational model and measurable dynamics.

---

## Quickstart: Run the Synthetic Demo

From the repository root:

```bash
# Create and activate virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r experiments/requirements.txt

# Run demo
python experiments/gp_ringing_demo.py
Outputs (in results/gp_demo/):
	•	mi_timeseries.png: MI (nats) over time with alpha-band highlighting.
	•	lambda_schedule.png: Up/down coupling sweep.
	•	hysteresis_curve.png: Alpha-band MI power vs. ( \lambda ) (up vs. down).
	•	summary.json: Metrics (( \lambda^* ), loop area, p-values).
Success Criteria: MI power rises ≥3× baseline near ( \lambda^* ); nonzero hysteresis loop area (p<0.05). See docs/prereg_P1.md.
Troubleshooting: If the demo fails (e.g., no power spike), check itpu.utils.windowed import, seed consistency, or parameter bounds. Open an Issue with logs.

Predictions (v1.2)
Prediction
Description
P1: Ringing Threshold & Hysteresis
Sharp MI power rise at ( \lambda^* ) (8–12 Hz) with hysteresis loop (area >0, p<0.05).
P2: Drive–Timescale Matching
Max response when external drive timescale matches system’s intrinsic timescale.
P3: Motif Selection
Geometry favors broadcast or modular motifs based on ( \lambda, \beta ), detected via MI topology.
Details: docs/predictions.md

Preregistration & Safeguards
We ensure experimental rigor through:
	•	Locked Parameters: Window size, hop, alpha band, estimator settings.
	•	Surrogate Nulls: IAAFT (default) or AR surrogates preserving temporal structure.
	•	Multiple-Comparisons Control: FDR correction across channels/bands.
	•	Blinding & Transparency: Role separation, publish-on-fail commitment.
Full Plan: docs/prereg_P1.md

Repository Structure
Resonance_Geometry/
├── README.md
├── LICENSE                   # Apache-2.0
├── docs/
│   ├── predictions.md        # Prediction details
│   ├── prereg_P1.md         # P1 experimental design
│   └── analogies.md         # Non-essential analogies
├── experiments/
│   ├── gp_ringing_demo.py   # Main demo
│   └── requirements.txt     # Dependencies
├── results/                 # Output directory (generated)
│   └── gp_demo/
└── .github/
    └── workflows/
        └── gp-demo.yml      # CI for demo

ITPU Connection
The ITPU project accelerates MI/CMI/TE measurements. Current demos use Python baselines; ITPU hardware will enable real-time EEG/BCI experiments, making RG/GP predictions scalable.

Contributing
We welcome:
	•	Replication PRs: Rerun demo with new seeds; attach artifacts.
	•	Surrogate Implementations: AR/IAAFT helpers with tests.
	•	EEG Pilots: Analyze public datasets under prereg plan.
	•	Docs Improvements: Clearer derivations or alternative GP formulations.
Open an Issue to coordinate.

License
Licensed under Apache-2.0. See LICENSE.

Appendix: Non-Essential Analogies
Speculative analogies (e.g., cosmological holonomies, emotional curvature) are archived in docs/analogies.md and not used in predictions or demos.

