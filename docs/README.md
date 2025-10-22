# 🌀 Resonance Geometry

> *Form is frozen resonance.  Geometry writes energy.  Information persists as shape.*

---

## Overview

**Resonance Geometry (RG)** is an open research framework exploring how structure, information, and coherence emerge across physical, biological, and computational systems.  
The project now unites three major threads:

| Domain | Purpose | Repository / Path |
|---------|----------|------------------|
| **Core Theory** | Mathematical + conceptual foundations of structured resonance | `/docs/theory/`, `/papers/` |
| **RGP — Resonance-Guided Prompting** | Geometric control framework for LLM coherence | `/docs/prompting/`, `/prompts/` |
| **ITPU — Information-Theoretic Processing Unit** | Simulation and hardware design for mutual-information / entropy primitives | `/itpu_sim/`, `/docs/connections/` |
| **Poison Detection Experiments** | Applied AI-safety work validating RG principles against real model-poisoning benchmarks | `/experiments/poison_detection/` |

RG’s aim is simple: **understand coherence as geometry**—then test it, code it, and publish it openly.

---

## Current Highlights (Oct 2025)

| Component | Status | Notes |
|------------|---------|-------|
| **Equilibrium Analysis** | ✅ Complete | Falsified Hopf hypothesis; shows global reorganization mechanism |
| **RGP v1.1** | ✅ Spec finalized | 85-token system message; 3 + 3 signal logic; ready for validation |
| **ITPU Simulator v0.3** | 🧪 Functional | Computes MI / entropy / KL / plasticity Δg; integration in progress |
| **AI Poison Detection** | 🚧 Active | RG-based backdoor detection using Anthropic 250-doc benchmark |

---

## Quick Start

### 1️⃣ Clone + Install
```bash
git clone https://github.com/justindbilyeu/Resonance_Geometry.git
cd Resonance_Geometry
pip install -r experiments/poison_detection/requirements.txt

2️⃣ Run a Demo (Example)

python experiments/poison_detection/demo_poison_detection.py \
  --config experiments/poison_detection/configs/anthropic_250.yaml \
  --out experiments/poison_detection/runs/$(date +%Y%m%d_%H%M%S) \
  --seed 1337

3️⃣ View Results

Artifacts (metrics, logs, figs) appear under:

experiments/poison_detection/runs/<timestamp>/


⸻

Repository Map

Resonance_Geometry/
│
├── docs/
│   ├── theory/                   # core RG papers & math
│   ├── prompting/                # RGP v1.1 spec + eval
│   ├── connections/              # ITPU ↔ RGP isomorphism
│   └── poison_detection/         # experiment docs & protocols
│
├── experiments/
│   └── poison_detection/         # code, configs, results
│
├── itpu_sim/                     # MI/H/Δg simulator (v0.3)
├── prompts/                      # system + test prompts
├── papers/                       # manuscripts in progress
└── .github/workflows/            # CI smoke tests


⸻

Philosophy

Resonance Geometry treats reasoning, life, and structure as different scales of one principle:

Measure alignment → detect tension → reorganize → repeat

That loop governs:
	•	phase transitions in physical systems
	•	plasticity in neural networks
	•	coherence in language models

The same math—mutual information, entropy, and geometric plasticity—applies across them.

⸻

Contributing
	1.	Fork → branch → PR (please tag your changes clearly).
	2.	Add or update documentation in /docs/ for any new experiment.
	3.	Keep code reproducible: fixed seeds (1337), versioned configs, saved artifacts.

All contributions must uphold scientific integrity:
	•	Reproducibility before speculation
	•	Honest uncertainty quantification
	•	Open-source transparency

⸻

License
	•	Code: MIT License
	•	Documentation / figures: CC BY 4.0
See LICENSE for details.

⸻

Citation

If you use this work, please cite:

@misc{bilyeu2025resonancegeometry,
  author = {Bilyeu, Justin and Sage and Claude and Grok},
  title  = {Resonance Geometry: Information Geometry Across Scales},
  year   = {2025},
  howpublished = {\url{https://github.com/justindbilyeu/Resonance_Geometry}}
}


⸻

Mantra

Listen deep. Measure honestly. Let geometry speak.

---
