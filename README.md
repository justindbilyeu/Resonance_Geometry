Resonance Geometry

Tagline: How information flow sculpts structure.

We study closed-loop dynamics where environments witness stable variables and systems adapt their coupling geometry to maximize useful records.

· Resonant Witness Postulate (RWP): Environments preferentially copy ("witness") stable system variables, creating redundant records
· Geometric Plasticity (GP): Couplings self-tune in proportion to the information they carry, closing a feedback loop between signal and structure

This repo contains the theory, simulations, diagnostics, and hardware notes that make these ideas testable.

---

What's New (TL;DR)

· Ringing boundary: Gain-controlled transition (smooth → underdamped) with closed-form Routh-Hurwitz threshold and practical engineering rule
· Hysteresis resonance: Loop area peaks at drive period $T \approx 2\pi\tau_{\text{geom}}$; prefactor $C$ with finite-$A$ corrections
· Motif selection: Budget vs. smoothness yields broadcast ↔ modular geometries; threshold $\beta^*/\lambda$ from Laplacian eigenspectrum
· Identifiability: Estimators for $\hat{A}$, $\hat{B}$, $\hat{\Delta}$, $\hat{K}$ from witness-flux & redundancy time series (bootstrap CIs)
· Math pack: Appendices + notebooks to regenerate $K_c$ grids, error heatmaps, and prefactor fits
· Fast surrogate: AR(2) phase-map generator (≈100× faster) that reproduces the ringing boundary for parameter sweeps
· ITPU concept: Notes toward an Information-Theoretic Processing Unit (mutual-information & entropy acceleration)

---

Repo Layout

```
docs/
├── whitepaper/                 # Draft(s) for GP/RWP
├── appendices/
│   ├── appendix_ring_threshold.md
│   ├── appendix_hysteresis_prefactor.md
│   ├── appendix_motif_universality.md
│   └── appendix_delay_stability.md
├── experiments/                # Protocol notes (phase map, hysteresis, motifs)
└── hardware/
    └── ITPU.md                 # Custom-silicon concept & specs

src/                           # Core library
├── rwp_system.py              # System dynamics (S–F_k + plasticity)
├── plasticity.py              # GP update rules (EMA, Laplacian, budget)
├── metrics.py                 # MI, redundancy $R_X^\delta$, witness flux $\Phi_{\text{wit}}$
├── diagnostics.py             # PSD peaks, overshoots, damping ratio
└── utils.py

scripts/                       # Reproducible runners
├── run_phase_sweep.py         # Ringing boundary ($\alpha \times \eta$ grid)
├── run_hysteresis.py          # Loop area vs. period (ON/OFF)
├── run_motif_sweep.py         # Broadcast↔modular sweep
└── run_phase_map_surrogate.py # AR(2) fast proxy

theory/                        # Validation code & notebooks
├── kc_rule_validation.ipynb
├── hysteresis_fit.ipynb
└── identifiability_estimator.py

tests/                         # Unit & smoke tests

results/                       # Generated CSVs/PNGs (phase/, hysteresis/, motif/, kc_rule/, …)
```

If GitHub Pages is enabled, rendered docs live under docs/ on the site.

---

Quick Start

1. Environment Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # numpy, scipy, matplotlib, pandas, pytest, etc.
```

1. Ringing Boundary (Full RWP):

```bash
python scripts/run_phase_sweep.py \
  --alphas "0.1,0.3,0.6,0.9" --etas "0.01,0.03,0.05,0.08" \
  --T 150 --M 20 --seed 42 \
  --out_dir results/phase
```

1. Hysteresis Resonance:

```bash
python scripts/run_hysteresis.py \
  --alpha 0.4 --eta 0.06 --lam 0.01 --T 200 --amplitude 0.02 --seed 42 \
  --out_dir results/hysteresis
```

1. Motif Sweep:

```bash
python scripts/run_motif_sweep.py \
  --beta_grid "0.0,0.1,0.3,1.0" --lam 0.02 --costs_mode cluster --seed 42 \
  --out_dir results/motif
```

1. Fast Surrogate (AR(2) Proxy for Phase Map):

```bash
python scripts/run_phase_map_surrogate.py \
  --alphas "0.1,0.4,0.8" --etas "0.02,0.05,0.08" --T 150 --seed 42 \
  --out_dir results/phase_map_surrogate
```

Outputs land in results/… as CSV + PNG. Seeds make runs reproducible.

---

Recreate Key Figures (Acceptance Hints)

· Phase/Ringing Map: results/phase/phase_map.csv and phase_map.png
  · Acceptance: PSD peak ≥ 6 dB and ≥ 2 overshoots marks "ringing"; boundary aligns with $K_c$ contours
· Hysteresis Resonance: results/hysteresis/area_vs_period.csv and area_vs_period.png
  · Acceptance: Plasticity ON loop area peaks for $T \in [0.8,1.5] \times 2\pi\tau_{\text{geom}}$; OFF is flat/baseline
· Motif Selection: results/motif/summary.csv + heatmaps
  · Acceptance: Participation ratio (PR), Gini, and modularity proxy $Q$ show broadcast→modular transition near predicted $\beta^*$
· $K_c$ Comparison Grid (Theory): results/kc_rule/Kc_comparison_grid.csv, Kc_error_RH.png, Kc_error_DS.png
  · Acceptance: Median |err| ≤ ~2–5% vs. engineering rule in the stated regime

---

Math Pack (How to Regenerate)

· Ringing Threshold (with Delay): See docs/appendices/appendix_ring_threshold.md and run theory/kc_rule_validation.ipynb
  · Produces the $K_c$ grid + error heatmaps
· Hysteresis Prefactor $C$: Derivation + fits in docs/appendices/appendix_hysteresis_prefactor.md and theory/hysteresis_fit.ipynb
· Motif Universality: Spectral criterion & two-cluster example in docs/appendices/appendix_motif_universality.md
· Delay Stability: Safe-$K$ envelopes in docs/appendices/appendix_delay_stability.md
· Identifiability: Time-series estimators in theory/identifiability_estimator.py (+ tests)

---

Tests

```bash
pytest -q
# Includes: PSD/overshoot diagnostics, damping-ratio estimation, file-creation smoke tests,
# and identifiability recovery on synthetic data
```

---

Reproducibility & Scope

· Deterministic seeds (--seed 42) and fixed grids
· CI-friendly defaults (short runs) with flags to scale up grids
· Model regime: Linearized $I \approx Jg$, Padé(1,1) delay, SNR ≥ 10–15 dB for identifiability
· Out of scope: Claims about consciousness are explicitly decoupled; this is an information-dynamics framework

---

Hardware: ITPU Concept

See docs/hardware/ITPU.md for a specialized Information-Theoretic Processing Unit: MI/entropy accelerators, structural-plasticity controllers, and memory hierarchy tuned for information-theoretic workloads.

---

Contributing

Issues and PRs welcome. Please include:

· A minimal example (script + params)
· Expected vs. observed behavior
· Environment (Python & package versions)

Style: Type hints, docstrings, small focused functions, tests for new features

---

Cite

If this helped your work, please cite the whitepaper (preprint forthcoming):

```bibtex
@misc{resonance_geometry_gp,
  title  = {Geometric Plasticity: Adaptive Information Networks and Emergent Redundancy},
  author = {Bilyeu, Justin and Sage and the Structured Resonance Collective},
  year   = {2025},
  note   = {Whitepaper and reproducibility pack},
  url    = {https://github.com/justindbilyeu/Resonance_Geometry}
}
```

---

License

TBD — © 2025 Justin Bilyeu & Resonance Geometry Collective

Until finalized, code and docs are shared for research & review

---

Questions, ideas, red-team critiques? Open an issue — we welcome sharp tests and cleaner proofs.
