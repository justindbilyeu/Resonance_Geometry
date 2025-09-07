Resonance Geometry

Tagline: How information flow sculpts structure.
We study closed-loop dynamics where environments witness stable variables and the system adapts its coupling geometry to maximize useful records.

	•	Resonant Witness Postulate (RWP): environments preferentially copy (“witness”) stable system variables, creating redundant records.
	•	Geometric Plasticity (GP): couplings self-tune in proportion to the information they carry, closing a feedback loop between signal and structure.

This repo contains the theory, simulations, diagnostics, and hardware notes that make those ideas testable.

⸻

What’s new (TL;DR)
	•	Ringing boundary: gain-controlled transition (smooth → underdamped) with closed-form Routh–Hurwitz threshold and a practical engineering rule.
	•	Hysteresis resonance: loop area peaks at drive period T \approx 2\pi\,\tau_{\text{geom}}; prefactor C with finite-A corrections.
	•	Motif selection: budget vs. smoothness yields broadcast ↔ modular geometries; threshold \(\beta^\*/\lambda\) from Laplacian eigenspectrum.
	•	Identifiability: estimators for \hat A, \hat B, \hat \Delta, \hat K from witness-flux & redundancy time series (bootstrap CIs).
	•	Math pack: appendices + notebooks to regenerate K_c grids, error heatmaps, and prefactor fits.
	•	Fast surrogate: AR(2) phase-map generator (≈100× faster) that reproduces the ringing boundary for parameter sweeps.
	•	ITPU concept: notes toward an Information-Theoretic Processing Unit (mutual-information & entropy acceleration).

⸻

Repo layout

docs/
  whitepaper/                 ← draft(s) for GP/RWP
  appendices/
    appendix_ring_threshold.md
    appendix_hysteresis_prefactor.md
    appendix_motif_universality.md
    appendix_delay_stability.md
  experiments/                ← protocol notes (phase map, hysteresis, motifs)
  hardware/
    ITPU.md                   ← custom-silicon concept & specs

src/                          ← core library
  rwp_system.py               ← system dynamics (S–F_k + plasticity)
  plasticity.py               ← GP update rules (EMA, Laplacian, budget)
  metrics.py                  ← MI, redundancy R_X^δ, witness flux Φ_wit
  diagnostics.py              ← PSD peaks, overshoots, damping ratio
  utils.py

scripts/                      ← reproducible runners
  run_phase_sweep.py          ← ringing boundary (α×η grid)
  run_hysteresis.py           ← loop area vs. period (ON/OFF)
  run_motif_sweep.py          ← broadcast↔modular sweep
  run_phase_map_surrogate.py  ← AR(2) fast proxy

theory/                       ← validation code & notebooks
  kc_rule_validation.ipynb
  hysteresis_fit.ipynb
  identifiability_estimator.py

tests/                        ← unit & smoke tests

results/                      ← generated CSVs/PNGs (phase/, hysteresis/, motif/, kc_rule/, …)

If GitHub Pages is enabled, rendered docs live under docs/ on the site.

⸻

Quick start

# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, scipy, matplotlib, pandas, pytest, etc.

# 2) Ringing boundary (full RWP)
python scripts/run_phase_sweep.py \
  --alphas "0.1,0.3,0.6,0.9" --etas "0.01,0.03,0.05,0.08" \
  --T 150 --M 20 --seed 42 \
  --out_dir results/phase

# 3) Hysteresis resonance
python scripts/run_hysteresis.py \
  --alpha 0.4 --eta 0.06 --lam 0.01 --T 200 --amplitude 0.02 --seed 42 \
  --out_dir results/hysteresis

# 4) Motif sweep
python scripts/run_motif_sweep.py \
  --beta_grid "0.0,0.1,0.3,1.0" --lam 0.02 --costs_mode cluster --seed 42 \
  --out_dir results/motif

# 5) FAST surrogate (AR(2) proxy for phase map)
python scripts/run_phase_map_surrogate.py \
  --alphas "0.1,0.4,0.8" --etas "0.02,0.05,0.08" --T 150 --seed 42 \
  --out_dir results/phase_map_surrogate

Outputs land in results/… as CSV + PNG. Seeds make runs reproducible.

⸻

Recreate the key figures (acceptance hints)
	•	Phase / Ringing map:
results/phase/phase_map.csv and phase_map.png.
Acceptance: PSD peak ≥ 6 dB and ≥ 2 overshoots marks “ringing”; boundary aligns with K_c contours.
	•	Hysteresis resonance:
results/hysteresis/area_vs_period.csv and area_vs_period.png.
Acceptance: plasticity ON loop area peaks for T \in [0.8,1.5]\times 2\pi\,\tau_{\text{geom}}; OFF is flat/baseline.
	•	Motif selection:
results/motif/summary.csv + heatmaps.
Acceptance: participation ratio (PR), Gini, and modularity proxy Q show broadcast→modular transition near predicted \(\beta^\*\).
	•	Kc comparison grid (theory):
results/kc_rule/Kc_comparison_grid.csv, Kc_error_RH.png, Kc_error_DS.png.
Acceptance: median |\text{err}| ≤ ~2–5% vs. engineering rule in the stated regime.

⸻

Math pack (how to regenerate)
	•	Ringing threshold (with delay): see docs/appendices/appendix_ring_threshold.md and run theory/kc_rule_validation.ipynb.
Produces the K_c grid + error heatmaps.
	•	Hysteresis prefactor C: derivation + fits in
docs/appendices/appendix_hysteresis_prefactor.md and theory/hysteresis_fit.ipynb.
	•	Motif universality: spectral criterion & two-cluster example in
docs/appendices/appendix_motif_universality.md.
	•	Delay stability: safe-K envelopes in
docs/appendices/appendix_delay_stability.md.
	•	Identifiability: time-series estimators in
theory/identifiability_estimator.py (+ tests).

⸻

Tests

pytest -q
# includes: PSD/overshoot diagnostics, damping-ratio estimation, file-creation smoke tests,
# and identifiability recovery on synthetic data.


⸻

Reproducibility & scope
	•	Deterministic seeds (--seed 42) and fixed grids.
	•	CI-friendly defaults (short runs) with flags to scale up grids.
	•	Model regime: linearized I \approx Jg, Padé(1,1) delay, SNR ≥ 10–15 dB for identifiability.
	•	Out of scope: claims about consciousness are explicitly decoupled; this is an information-dynamics framework.

⸻

Hardware: ITPU concept

See docs/hardware/ITPU.md for a specialized Information-Theoretic Processing Unit: MI/entropy accelerators, structural-plasticity controllers, and memory hierarchy tuned for information-theoretic workloads.

⸻

Contributing

Issues and PRs welcome. Please include:
	•	a minimal example (script + params),
	•	expected vs. observed behavior,
	•	environment (Python & package versions).

Style: type-hints, docstrings, small focused functions, tests for new features.

⸻

Cite

If this helped your work, please cite the whitepaper (preprint forthcoming):

@misc{resonance_geometry_gp,
  title  = {Geometric Plasticity: Adaptive Information Networks and Emergent Redundancy},
  author = {Bilyeu, Justin and Sage and the Structured Resonance Collective},
  year   = {2025},
  note   = {Whitepaper and reproducibility pack},
  url    = {https://github.com/justindbilyeu/Resonance_Geometry}
}


⸻

License

TBD — © 2025 Justin Bilyeu & Resonance Geometry Collective.
Until finalized, code and docs are shared for research & review.

⸻

Questions, ideas, red-team critiques? Open an issue — we welcome sharp tests and cleaner proofs.
