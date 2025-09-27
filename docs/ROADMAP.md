# ROADMAP.md

> For a directory-level overview of related philosophy, white paper, codex, simulation, and history materials, see [`README_bundle.md`](README_bundle.md).

## v0.1 (2–3 days): Baseline + CI + Results Seed
- [ ] Package `rwp-core` (GP engine + metrics) and wire tests
- [ ] Reproduce **three minimal demos** with fixed seeds:
      1) Ringing boundary probe (2–3 grid points)
      2) Hysteresis loop at T ≈ 2πτ_geom
      3) Motif contrast (broadcast vs modular)
- [ ] Save artifacts under `results/v0.1/` (CSV + PNG) with deterministic filenames
- [ ] CI: run unit tests + a tiny smoke experiment job

## v0.2 (7–10 days): Full Diagnostics
- [ ] Phase map over (α, η); mark empirical K_c
- [ ] Hysteresis resonance sweep T ∈ [0.3, 5]×2πτ_geom (plasticity ON vs OFF)
- [ ] Motif sweep in (β, B); compute PR, Q, Gini; render heatmaps
- [ ] Robustness: delays Δ, back-action κ, fragment correlations; summarize

## v0.3 (2 weeks): Paper & Repro Pack
- [ ] Freeze configs, pin seeds, export `repro.json`
- [ ] Write whitepaper draft in `docs/papers/gp_whitepaper.md`
- [ ] Create `make figures` target to regenerate all plots
- [ ] Tag release v0.3 and archive on Zenodo for DOI
