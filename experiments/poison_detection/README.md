# RG-Based Poisoning Detection (Φ/κ/RTP)

This experiment tests whether **Resonance Geometry** metrics — **Φ** (coherence), **κ** (tension),
and **RTP** (snap-through reframe) — can detect **backdoor poisoning** in language models.

## Quick Start
```bash
pip install -r requirements.txt
python demo_poison_detection.py
```

Expected behavior: a **drop in Φ** and **spike in κ** at the backdoor trigger, optionally an **RTP** event.

## Files
- `rg_detector.py` — computes Φ/κ and detects RTP from model outputs
- `poison_generator.py` — creates poisoned documents with a configurable trigger
- `demo_poison_detection.py` — 5‑minute end‑to‑end demo (GPT‑2 baseline)
- `../docs/poison_detection/EXPERIMENT_PROTOCOL.md` — full validation protocol (training + ROC/AUC)
- `../docs/poison_detection/IMMEDIATE_ACTIONS.md` — short team plan
- `../docs/poison_detection/FOR_SAGE.md` — conceptual/poetic framing

## Status
- Demo: ready
- Full validation: pending (see `../docs/poison_detection/EXPERIMENT_PROTOCOL.md`)
