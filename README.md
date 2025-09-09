# Resonance Geometry + Geometric Plasticity

**How information flow sculpts structure.**

We study closed-loop dynamics where environments witness (copy) stable variables and systems adapt their coupling geometry to maximize useful records. The result is a feedback loop between signal and structure that leaves measurable, testable fingerprints in time-series, spectra, and spatial patterns.

- **Resonant Witness Postulate (RWP)** — Environments preferentially copy (“witness”) stable system variables, creating redundant records
- **Geometric Plasticity (GP)** — Couplings self-tune in proportion to the information they carry, closing the loop between observed signal and coupling geometry

This repo contains the theory, simulations, diagnostics, and hardware notes that make these claims falsifiable.

-----

## What’s New

- **Ringing Boundary** (closed form): Gain-controlled transition from smooth decay → underdamped response. Routh–Hurwitz threshold and engineering rule derived from Hessian spectrum and graph Laplacian modes
- **Hysteresis Resonance**: Loop area peaks at drive period T ≈ 2π τ_geom; prefactor C includes finite-A corrections with exported fits + error heatmaps
- **Motif Selection**: Budget (λ) vs. smoothness (β) yields broadcast ↔ modular geometries; threshold β*/λ follows from Laplacian eigenspectrum
- **Identifiability**: Estimators for Â, λ̂, β̂, γ̂, K̂ from witness-flux & redundancy time-series with bootstrap CIs
- **Variational GP**: Clean action principle, Rayleigh/Onsager dissipation (first-order flows), and memory kernels that generate effective delay Δ
- **Fast Surrogate**: AR(2) phase-map generator (~100× speed-up) that reproduces the ringing boundary for parameter sweeps
- **ITPU Notes**: Toward an Information-Theoretic Processing Unit for MI/entropy acceleration

-----

## Core Equations

### GP Energy and Flows

Let **g** ∈ ℝⁿ be coupling strengths on a graph with Laplacian **L**, and **Ī** the “witness” (redundant record) variable. For linear case I = γg:

```
V(g,Ī) = -Ī^T g + (λ/2) g^T g + (β/2) g^T L g + (A/2) ||Ī - γg||²
```

**Dissipative flows** (Onsager/Rayleigh):

```
η ġ = -∂_g V,    A İ̄ = -∂_Ī V
```

**Ringing boundary** (instability when):

```
H_gg = (λ + Aγ²)I + βL
min_k{λ + Aγ² + β μ_k} = 0
```

Here μ_k are Laplacian eigenvalues; the first mode crossing zero predicts the emerging pattern/motif.

### Field Extension

- **Conservative track**: Add kinetic and gradient terms in Lagrangian density for coherence propagation
- **Plastic track**: Keep gradient-flow structure (Onsager); spatial terms yield pattern formation with analyzable dispersion

-----

## Repository Structure

```
theory/
├── GP_variational_v1.md          # action, EL, dissipation, memory kernels, stability
├── psi_g_master_equation.md      # Lindblad-style ψ–g coupling + observables
└── axioms_bridge.md              # RWP & Codex mapping (clearly labeled as heuristic)

sims/
├── gp_graph/                     # finite-graph flows; Hessian scans; Kc grids
│   ├── run_graph.py/.nb
│   └── export/ (CSV, SVG, PDF)
├── gp_field/                     # 1D/2D wave vs RD demos (Method of Lines + FD)
│   ├── run_field.py/.nb
│   └── export/
└── two_slit_plasticity/          # ψ–g toy: adaptive decoherence & reintegration
    ├── run_twoslit.py/.nb
    └── export/

figures/
├── phase_maps/
├── instability_surfaces/
└── field_demos/

instruments/
└── resonance_table/
    ├── spec.md                   # BOM, wiring, DAQ, calibration, pass/fail criteria
    └── notes.md                  # troubleshooting + real data dropbox pointers

papers/
├── short_perspective.md
└── methods_appendix.md
```

-----

## Quick Start

### Environment Setup

```bash
# Create fresh Python environment
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt   # or pip
```

### Run Simulations

**Graph GP scans** (Python):

```bash
python sims/gp_graph/run_graph.py --grid default --export figs
```

**Wolfram notebooks** (recommended for figure pack):

```
# Open sims/gp_graph/run_graph.nb and Evaluate Notebook
```

**Field demos** (RD vs wave):

```bash
python sims/gp_field/run_field.py --mode RD --export figs
python sims/gp_field/run_field.py --mode wave --export figs
```

Exports land in `sims/**/export/` and are mirrored in `/figures/` by the Makefile.

-----

## Diagnostics You Can Trust

- **Phase portraits**: g_i vs Ī_i; energy H(t) for conservative track
- **Eigenvalue spectra**: H_gg and full block Hessian; instability surfaces over (λ,β,γ)
- **Dispersion**: Field GP unstable k-bands; space–time plots of wave vs RD regimes
- **Identifiability**: Parameter recovery from synthetic witness-flux signals with bootstrap CIs

-----

## Experiments (Fast Path)

### Resonance Table v0

- **Setup**: Acoustics/EM drive, plate/membrane, frequency sweep → transfer function, SNR, mode map (phone camera + powder/salt)
- **Pass/Fail**: Recover predicted peaks within ±Δf; SNR > X dB; repeatability ±Y%

### Two-slit Plasticity Toy

- **Setup**: Couple ψ to g via simple dissipator; look for ring-up near predicted K_c
- **Validation**: Phase-locking probability vs noise

### PEMF/Trehalose Microtubule Assay (Optional)

- **Setup**: Scan drive and damping; check phase-locking probability vs noise

-----

## Roadmap

- **v1.0** — Variational GP + dissipation + memory; figure pack (Hessian maps, dispersion, wave vs RD)
- **v1.1** — ψ–g master equation (Lindblad sketch) + two-slit plasticity demo
- **v1.2** — ITPU prototype notes + AR(2) surrogate integration into dashboards
- **v1.3** — Instrument data ingestion + parameter ID from real runs

-----

## Contributing

Pull requests welcome. Please maintain:

- **Reproducible scripts**: Deterministic (seeds, tolerances, versions)
- **Generated figures**: Regenerated from code (no hand-edited artifacts in `/figures`)
- **Testable claims**: Tied to tests (unit or numerical), with CSVs or notebooks to replicate

-----

## Citation

```
Resonance Geometry Collective (2025). Resonance Geometry: How information flow sculpts structure. (Versioned repository).
```

Include commit hash and figure export date in methods.

-----

## GP Mini (Drop-in Reference)

**One-line**: Couplings change in the direction that carries more information.

**Energy**:

```
V(g,Ī) = -Ī^T g + (λ/2)g^T g + (β/2)g^T Lg + (A/2)||Ī - γg||²
```

**Flows** (Rayleigh/Onsager):

```
η ġ = -∂_g V,    A İ̄ = -∂_Ī V
```

**Ringing boundary**: Smallest eigenvalue of `H_gg = (λ + Aγ²)I + βL` crosses zero → first unstable Laplacian mode predicts emerging motif (broadcast ↔ modular).

**Why it matters**: GP provides a variational + dissipative route from data flow to geometry. It predicts when a system starts to ring, which shapes appear, and how to recover parameters from witnessed records.

**Run me**:

```bash
python sims/gp_graph/run_graph.py --grid default --export figs
python sims/gp_field/run_field.py --mode RD --export figs
```

**Outputs**: Phase portraits, energy traces, eigen spectra, instability maps, dispersion, and space–time plots.
