# Resonance Geometry (RG) & Geometric Plasticity (GP)

**A testable framework for adaptive systems.**  
Coupling geometry changes to track information flow—constrained by costs like energy, sparsity, and latency. We publish falsifiable predictions and ship runnable demos.

---

## What this is (in plain English)

Many systems “re-wire” themselves as they learn: neurons synchronize, modules form, signals pass through preferred pathways. **Geometric Plasticity (GP)** models this by letting a network’s coupling geometry **g** evolve to align with measured information flow **\(\bar I\)** while paying penalties for complexity and delay.

One simple GP potential we study:

\[
V(g;\bar I) = -\,\bar I^\top g \;+\; \frac{\lambda}{2}\,\|g\|^2 \;+\; \frac{\beta}{2}\,g^\top L g \;+\; \frac{A}{2}\,\|\bar I - I(g,t)\|^2
\]

with gradient-flow dynamics \(\dot g = -\eta\,\nabla_g V\).  
Here, \(I(g,t)\) is information measured from the system (e.g., windowed mutual information), \(L\) encourages smooth structure, and \(\lambda,\beta,A\) trade off simplicity, modularity, and fidelity.

We focus on **empirical, falsifiable predictions** and provide code you can run end-to-end.

---

## What’s new (Sept 2025)

- ✅ **Pre-registered prediction (P1)**: a **ringing threshold**—a sharp rise in alpha-band MI power at a specific coupling \(\lambda^\*\), with **hysteresis** under up/down sweeps. See `docs/prereg_P1.md`.
- ✅ **Runnable synthetic demo**: `experiments/gp_ringing_demo.py` produces MI time-series, lambda schedule, a hysteresis curve, plus JSON summary.
- ✅ **Reproducibility hygiene**: fixed parameters, seeded RNG, surrogate nulls, and multiple-comparisons control described in `docs/prereg_P1.md`.
- 🧪 **Next**: replicate on small EEG datasets using the same locked analysis.

> Note: Earlier “grand analogies” (e.g., cosmological holonomies) are now marked **analogy-only** and **non-essential**. The working program is the GP variational model + measurable information dynamics.

---

## Quickstart (run the synthetic demo)

From the repo root:

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 2) Install minimal deps for the demo
python -m pip install --upgrade pip
pip install -r experiments/requirements.txt

# 3) Run the ringing/hysteresis demo
python experiments/gp_ringing_demo.py

Outputs (written to results/gp_demo/):
	•	mi_timeseries.png — MI (nats) over time with alpha-band highlighting
	•	lambda_schedule.png — the up/down coupling sweep
	•	hysteresis_curve.png — alpha-band MI power vs. (\lambda) (up vs. down)
	•	summary.json — metrics (estimated (\lambda^*), loop area, surrogate p-values)

A successful run produces a visible MI power rise near (\lambda^*) and a nonzero hysteresis loop area; see exact pass/fail in the prereg doc.

⸻

Predictions (v1.2)
	•	P1 — Ringing threshold & hysteresis (primary endpoint)
As (\lambda) increases, alpha-band MI power exhibits a sharp rise at (\lambda^*); decreasing (\lambda) traces a different path (hysteresis loop). Statistical criteria and nulls are pre-registered.
	•	P2 — Drive–timescale matching
Max response occurs when external drive timescale matches the system’s intrinsic timescale predicted by GP dynamics.
	•	P3 — Motif selection
Depending on ((\lambda,\beta)), the geometry prefers broadcast vs modular motifs; detected by MI topology summaries.

See details: docs/predictions.md

⸻

Preregistration & statistical safeguards

We treat this like a real experiment:
	•	Locked parameters (window size/hop, alpha band, estimator settings)
	•	Surrogate nulls (IAAFT by default; optional AR surrogates) that preserve key temporal structure
	•	Multiple-comparisons control across channels/bands
	•	Blinding & role separation and publish-on-fail criteria

Read the full plan: docs/prereg_P1.md

⸻

Repository layout

Resonance_Geometry/
├── README.md
├── docs/
│   ├── predictions.md
│   └── prereg_P1.md
├── experiments/
│   ├── gp_ringing_demo.py
│   └── requirements.txt
├── results/
│   └── gp_demo/          # created by the demo at runtime
└── .github/
    └── workflows/
        └── gp-demo.yml   # optional CI that runs the demo and uploads artifacts


⸻

How this connects to ITPU (optional)

The ITPU project (Information-Theoretic Processing Unit) accelerates MI/CMI/TE measurements. Today we use Python baselines; later, the same analysis can be run on ITPU hardware for real-time experiments. If you’re collaborating on EEG/BCI, this is our planned acceleration path.

⸻

Contributing

We welcome:
	•	Replication PRs: rerun the synthetic demo with a different seed and attach artifacts
	•	Surrogate implementations: clean AR/IAAFT helpers with tests
	•	EEG pilots: small, public datasets analyzed under the prereg plan
	•	Docs hardening: clearer derivations or alternative formulations of the GP potential

Please open an Issue first for coordination.

⸻

License

Apache-2.0 — see LICENSE.

