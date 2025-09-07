# Resonance Geometry

**Resonance Geometry** is a research program on how **information flow sculpts structure**.  
At its core are two coupled ideas:

- **Resonant Witness Postulate (RWP):** environments preferentially copy (“witness”) stable variables, creating **redundant records**.
- **Geometric Plasticity (GP):** the **coupling geometry adapts** in proportion to the witnesses it carries, closing a feedback loop between **signal and structure**.

This repo houses the theory, simulations, diagnostics, and hardware notes that turn those ideas into testable science.

---

## What’s new (TL;DR)

- **Ringing boundary**: a gain-controlled transition (smooth → underdamped) in redundancy dynamics.  
- **Hysteresis resonance**: loop area peaks at drive period \(T \approx 2\pi \tau_{\text{geom}}\).  
- **Motif selection**: budget & smoothness trade-offs yield **broadcast** vs **modular** coupling motifs.  
- **Fast surrogate**: AR(2) phase-map script for quick ringing scans (100× faster than full sims).  
- **ITPU concept**: hardware notes for an **Information-Theoretic Processing Unit** (mutual-info & entropy acceleration).

---

## Repo layout

- **docs/**
  - **whitepaper/** — *Adaptive Information Networks / Geometric Plasticity* draft(s)
  - **experiments/** — protocol notes (phase map, hysteresis, motifs, surrogate)
  - **hardware/** — `ITPU.md` (custom silicon concept & specs)
- **src/** — core library (RWP system, plasticity rules, metrics, utils)
- **scripts/** — reproducible runners:
  - `run_phase_sweep.py` — full RWP ringing boundary (α×η grid)
  - `run_hysteresis.py` — loop-area vs period with plasticity ON/OFF
  - `run_motif_sweep.py` — broadcast↔modular classification
  - `run_phase_map_surrogate.py` — AR(2) fast proxy for ringing map
- **tests/** — unit & smoke tests for diagnostics and runners
- **results/** — generated CSVs/PNGs (phase/, hysteresis/, motif/, …)

> Live docs & PDFs (if GH Pages is enabled): `docs/` → site build.

---

## Quick start

```bash
# 1) env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, scipy, matplotlib, pandas, pytest, etc.

# 2) phase map (full RWP)
python scripts/run_phase_sweep.py \
  --alphas "0.1,0.3,0.6,0.9" --etas "0.01,0.03,0.05,0.08" \
  --T 150 --M 20 --out_dir results/phase

# 3) hysteresis resonance
python scripts/run_hysteresis.py \
  --alpha 0.4 --eta 0.06 --lam 0.01 --T 200 --amplitude 0.02 \
  --out_dir results/hysteresis

# 4) motif sweep
python scripts/run_motif_sweep.py --out_dir results/motif

# 5) FAST surrogate (AR(2) proxy)
python scripts/run_phase_map_surrogate.py \
  --alphas "0.1,0.4,0.8" --etas "0.02,0.05,0.08" --T 150 \
  --out_dir results/phase_map_surrogate
