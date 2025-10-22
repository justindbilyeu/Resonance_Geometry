# PR Plan — Add RG Poison Detection Experiment

## Title
experiment: RG-based detection of backdoor poisoning (Φ/κ/RTP signatures)

## Summary
Adds a complete experimental package to evaluate whether Resonance Geometry metrics — Φ (coherence),
κ (tension), and RTP (reframe threshold) — can detect backdoor triggers in poisoned LLMs.

## Files & Paths
- `experiments/poison_detection/rg_detector.py`
- `experiments/poison_detection/poison_generator.py`
- `experiments/poison_detection/demo_poison_detection.py`
- `experiments/poison_detection/requirements.txt`
- `experiments/poison_detection/README.md`
- `experiments/poison_detection/docs/Quick_Start.md`
- `experiments/poison_detection/docs/Experimental_Protocol.md`
- `experiments/poison_detection/docs/Immediate_Action_Plan.md`
- `experiments/poison_detection/docs/For_Sage.md`

## Steps for Codex
1. Create the directory `experiments/poison_detection/` and `experiments/poison_detection/docs/`.
2. Add the files above exactly at those paths.
3. Ensure `requirements.txt` contains minimal deps for demo (torch, transformers, numpy, matplotlib).
4. Verify `README.md` renders, and `demo_poison_detection.py` imports local modules.
5. Open a PR with the title and summary above.
6. Apply label: `experiment` and `security`.