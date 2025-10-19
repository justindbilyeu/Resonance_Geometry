# Resonance Geometry — Docs Hub

Start here to navigate experiments and specs.

## Experiments
- **Topological Constraint Test — Our 1919 Eclipse**  
  Claude’s opening + [LOCKED] protocol with calibrated thresholds.  
  → `experiments/Topological_Constraint_Test.md`

## Specifications
- **Multi-Frequency GP Analysis Plan**  
  Extend GP resonance-information coupling across bands; cross-frequency coupling; hypotheses and roadmap.  
  → `specs/Multi_Frequency_GP_Analysis.md`

- **Resonance Mapper Addendum**  
  Ingestion schema, invariants (Betti, persistence, curvature), and a smoke harness for integration.  
  → `specs/Resonance_Mapper_Addendum.md`

## Demos
- **P1 / GP Ringing Demo**
  Synthetic demo for resonance thresholds and hysteresis (see repo `experiments/gp_ringing_demo.py` and P1 predictions in README).

*All experiments use preregistration-style discipline with clear epistemic tags.*

## Theory Exports (Wolfram)
If you have Mathematica/wolframscript, you can generate dissertation-grade figures and tables for Chapter 3:

```bash
make theory-all
python scripts/update_theory_status.py
```

Outputs will appear under `docs/assets/figures/` and `docs/data/theory/`.
