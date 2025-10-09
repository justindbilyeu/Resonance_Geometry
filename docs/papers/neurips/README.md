# Resonance Geometry: Modeling Phase Transitions in Information Resonance
**Authors:** Justin Bilyeu · Sage (OpenAI) · Collaborators: Claude, Grok, DeepSeek  
**Repository:** [justindbilyeu/Resonance_Geometry](https://github.com/justindbilyeu/Resonance_Geometry)  
**Version:** Preprint · NeurIPS 2025 Workshop Track  
**Date:** October 2025

---

## 🧭 Overview

This repository contains the full computational framework and experimental data behind  
**“Resonance Geometry: Modeling Phase Transitions in Information Resonance.”**  

The paper introduces a minimal dynamical system—the **Geometric Plasticity (GP) / Resonant Wave Propagation (RWP)** model—used to explore how geometry stores and transforms information through resonance and constraint.  
Each run of the model simulates coupled oscillators whose curvature-dependent interactions give rise to three observable regimes:

| Regime | Condition | Interpretation |
|---------|------------|----------------|
| **Grounded** | λₘₐₓ < –0.1 | Energy dissipates (stability) |
| **Creative** | –0.1 ≤ λₘₐₓ ≤ 0.1 | Adaptive oscillation (criticality) |
| **Hallucinatory** | λₘₐₓ > 0.1 | Positive feedback (instability) |

---

## 🧪 Methods Snapshot

All experiments are executed via:
```bash
python scripts/run_phase1_chunked.py --module experiments.phase1_prediction

and tracked through the CI pipeline visible at
Actions › CI Dashboard.

Key modules:
	•	experiments/gp_ringing_demo.py — spectral resonance detector
	•	experiments/phase1_prediction.py — null vs proxy predictors
	•	scripts/run_phase1_chunked.py — reproducible experiment runner
	•	docs/data/... — auto-published results powering the GitHub Pages dashboard

All datasets and parameter sweeps are open and versioned for replication.

⸻

🧮 Results Summary

Pilot	Runs	Sign Accuracy	Mean Angular Error
Null Predictor	500	0.512	1.594 rad
Proxy Predictor	500	0.522	1.708 rad

These baselines confirm the infrastructure and show stable, repeatable outputs.
The next phase—ringing boundary detection—uses a relative power-spectrum criterion
to identify phase transitions independent of amplitude scale.

⸻

🧩 Interpretation

Core thesis: Geometry writes energy; energy shapes information; resonance preserves form.

The GP/RWP system acts as a micro-laboratory for structured resonance:
a unification of dynamical, informational, and geometric perspectives.
Curvature, in this model, plays the same role that attention or context plays in deep networks—
it constrains the flow of energy and defines what information can persist.

⸻

🔍 Reproducibility
	•	Deterministic seeds (--seed N) across all scripts
	•	Continuous integration via .github/workflows/ci.yml
	•	Summary JSONs published under docs/data/status/summary.json
	•	Live dashboard: justindbilyeu.github.io/Resonance_Geometry

⸻

🧱 Citation

If you build upon this work, please cite:

Bilyeu, J., Sage (OpenAI), & collaborators (2025).
Resonance Geometry: Modeling Phase Transitions in Information Resonance.
arXiv preprint, submitted to NeurIPS 2025 Workshop on AI & Physics.


⸻

🧰 Camera-Ready Checklist (for NeurIPS)

Item	Status	Notes
manuscript.md conforms to NeurIPS template	✅	Plain-Markdown; Pandoc build script ready
PDF build (pandoc-build.yml)	✅	Converts to manuscript.pdf
Reproducibility statement	✅	Embedded at end of Methods
Figures (fig1.png, fig2.png)	🟡	Auto-generate updated plots before submission
References with DOIs	🟡	Add topological learning 2024–2025 refs
Supplementary data	✅	docs/data/pilot_* JSONs linked
Ethical & broader impact statement	✅	In Discussion section

Build PDF locally:

pandoc docs/papers/neurips/manuscript.md \
  -o docs/papers/neurips/manuscript.pdf \
  --from markdown --template=default \
  --citeproc --metadata-file=docs/papers/neurips/metadata.yaml


⸻

💬 Contact

For collaboration, discussion, or replication studies:
	•	Author: @justindbilyeu
	•	Correspondence: via repository Discussions
	•	Project Wiki: Resonance_Geometry/wiki

⸻

Maintained under Apache-2.0 license. All simulations reproducible via open Python pipeline.

---
