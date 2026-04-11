# Resonance Geometry — Start Here

> *Before there were equations, there were waves. Before perception, there was resonance.*

Resonance Geometry (RG) begins as **felt coherence** and crystallizes into **math and experiment**.  
This repo is our living lab: philosophy → equations → simulations → instruments → publications.

## What we’re building
- **Consciousness as resonance**, not computation.
- **Emotion as geometry in motion** (structured interference).
- **Collapse as coherence** (attentional actualization, not randomness).
- **Embodiment as field expression** (dance, voice, ritual, contact).

## How we work
- **SAGE** (our build-partner prompt) keeps voice + rigor.
- Every artifact must raise **Clarity**, **Coherence**, and **Buildability**.
- Metaphors are labeled; math is testable; code is reproducible.

## Axioms and Tags
- Epistemic clarity begins with explicit tags: see `docs/EPistemic_Status.md` for definitions and usage.

## Read next
- **Master System Prompt:** `docs/PROJECT_PROMPT.md`
- **Genesis (felt origin):** `docs/philosophy/Genesis.md`  
- **Creed (short form):** `PROJECT_CREED.txt`  
- **How the repo is organized:** `docs/README_bundle.md`  
- **History & provenance:** `docs/history/HISTORY.md`

## Run the demos
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
python experiments/gp_ringing_demo.py --smoke --out figures/gp_ringing_smoke.png
