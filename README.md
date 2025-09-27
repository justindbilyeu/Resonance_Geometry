# Resonance Geometry (RG) & Geometric Plasticity (GP)

> "And God said, Let there be light: and there was light." — *Genesis 1:3*

Resonance Geometry began as a felt sense that rhythms could hold shape—intuition first, equations later. This repository maps that origin story into math you can inspect, simulations you can rerun, and experiments you can challenge.

### What’s Inside

| Track | Purpose |
| --- | --- |
| **Genesis** | Foundational mythos and narrative arc in the [Genesis prelude](docs/philosophy/Genesis.md). |
| **Codex** | Formal commitments distilled in the [Resonance Axioms](docs/philosophy/Axioms.md). |
| **Equations** | Project-wide drivers and constraints in the [Project Prompt](docs/PROJECT_PROMPT.md). |
| **Sims** | Gradient-flow experiments starting from [`gp_ringing_demo.py`](experiments/gp_ringing_demo.py). |
| **Figures** | Generated artifacts catalogued under [`figures/`](figures/) with context in the [epistemic status ledger](docs/EPistemic_Status.md). |
| **History** | Provenance trail curated in the [History ledger](docs/history/HISTORY.md). |

> **Resonant Check**
> - **Clarity** — plain-language overview matched with direct links to the governing documents.
> - **Coherence** — philosophy, axioms, and experiments harmonized through the gradient-flow variational potential.
> - **Buildability** — editable install path, reproducible tests, and a smoke demo you can run today.

### Quick Run

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e .
pytest
python experiments/gp_ringing_demo.py --smoke
```

Step into the living lattice—tune, test, and extend the resonance.

-----

## Framework Overview

Biological and artificial systems continuously rewire their internal connections to improve function—neurons synchronize, modules emerge, and signals find efficient pathways. **Geometric Plasticity (GP)** provides a testable mathematical model for this process, where a network’s coupling geometry `g` evolves to align with actual information flow `Ī`, while being constrained by complexity, modularity, and fidelity.

The framework centers around a variational potential `V(g; Ī)`:

```
V(g; Ī) = -Īᵀg + (λ/2)‖g‖² + (β/2)gᵀLg + (A/2)‖Ī - I(g,t)‖²
```

governed by gradient-flow dynamics `ġ = -η∇gV`. Here:

- `I(g,t)` is measured information (e.g., windowed mutual information)
- `L` is a Laplacian encouraging smooth or modular structure
- `λ, β, A` control penalties for complexity, structure, and tracking fidelity

This repository focuses on empirical predictions and reproducible experiments—bridging theory with measurable dynamics.

-----

## Current Status (September 2025)

- ✅ **Pre-registered prediction P1:** Demonstrated threshold-triggered rise in alpha-band mutual information and hysteresis under parameter sweeps
- ✅ **Functional synthetic demo:** `gp_ringing_demo.py` generates time-series, hysteresis curves, and structured results
- ✅ **Rigor-enforced testing:** Fixed parameters, seeded RNG, surrogate testing, and multiple-comparisons correction
- 🧪 **Next step:** Replication on small EEG datasets using a locked analysis protocol

*Note: Earlier cosmological analogies are deprecated. Focus is on the GP variational principle and measurable dynamics.*

-----

## Predictions (v1.2)

|ID    |Prediction                    |Description                                                                       |
|------|------------------------------|----------------------------------------------------------------------------------|
|**P1**|Ringing threshold & hysteresis|Sharp increase in MI power at critical λ*, with hysteresis under sweeps           |
|**P2**|Drive–timescale matching      |Maximal response when external drive matches intrinsic timescale                  |
|**P3**|Motif selection               |Structural preference for broadcast vs. modular motifs under different constraints|

**Full details:** [`docs/codex/policies/predictions.md`](docs/codex/policies/predictions.md)

-----

## Experimental Rigor

This project emphasizes reproducibility and falsifiability:

- **Locked analysis parameters:** Pre-registered window sizes, frequency bands, and estimators
- **Surrogate testing:** Using IAAFT/AR null models to preserve temporal structure
- **Multiple-comparisons control:** Corrected significance testing
- **Blinding & publish-on-fail:** Predefined success criteria and avoidance of p-hacking

**Pre-registration plan:** [`docs/codex/policies/prereg_P1.md`](docs/codex/policies/prereg_P1.md)

-----

## Repository Structure

```
Resonance_Geometry/
├── docs/
│   ├── philosophy/               # Conceptual framing and essays
│   ├── white-papers/             # Drafts, appendices, compiled PDFs
│   ├── codex/                    # Resonance codex variants & policies
│   └── history/                  # Provenance logs, inventories, lineage
├── simulations/                 # Simulation modules (headless)
├── figures/                     # Generated figures (tracked outputs)
├── archive/                     # Legacy repositories (REAL/GP/RG)
├── experiments/
│   └── gp_ringing_demo.py       # Main demonstration script
├── scripts/                     # Utility helpers (inventory generation, etc.)
├── results/                     # Legacy outputs (gitignored)
└── .github/workflows/           # CI configuration (sims.yml, etc.)
```

-----

## ITPU Integration

This project leverages the [Information-Theoretic Processing Unit (ITPU)](https://github.com/justindbilyeu/ITPU) for high-performance estimation of mutual information, transfer entropy, and other information dynamics. Python reference implementations are provided, with future support for real-time ITPU hardware acceleration.

-----

## Contributing

We welcome contributions in the following areas:

- Replication attempts with novel random seeds
- New surrogate data implementations (AR, IAAFT)
- Pilot studies using public EEG/MEG datasets
- Documentation improvements and typo fixes

Please open an issue before submitting a pull request to coordinate efforts.

-----

## License

Apache 2.0 – see <LICENSE> for details.

-----

## Repository Lineage

This canonical hub intentionally preserves the provenance of three legacy efforts:

- [`REAL`](https://github.com/justindbilyeu/REAL)
- [`Geometric-Plasticity-`](https://github.com/justindbilyeu/Geometric-Plasticity-)
- [`ResonanceGeometry`](https://github.com/justindbilyeu/ResonanceGeometry)

The `/archive/` directory houses frozen snapshots (or placeholders when network access is unavailable) for each source. Provenance, deduplication notes, and follow-up actions are tracked in [`docs/history/HISTORY.md`](docs/history/HISTORY.md).

For a guided tour of the merged documentation set, start with [`docs/README_bundle.md`](docs/README_bundle.md) and the epistemic-status context in [`docs/Epistemic_Status_Box.md`](docs/Epistemic_Status_Box.md).
