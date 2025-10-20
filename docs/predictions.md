# Falsifiable Predictions

This document tracks specific, testable predictions made by Resonance Geometry theory.
Each prediction includes: what we expect, how to test it, and the result.

**Purpose:** To maintain scientific rigor by making predictions **before** running experiments,
and recording outcomes honestly regardless of whether they support the theory.

---

## Status Key
- 🔵 **Untested** - Prediction made, experiment not yet run
- 🟡 **In Progress** - Experiment underway
- ✅ **Confirmed** - Prediction held within error bounds
- ⚠️ **Partial** - Prediction partially confirmed
- ❌ **Falsified** - Prediction did not hold

---

## Prediction 1: High-Resolution β_c Localization

**Date Proposed:** 2025-01-20  
**Status:** 🔵 Untested

**Prediction:**  
If we sweep β from 0.010 to 0.020 in steps of 0.0001 (100× finer than current grid),
the eigenvalue zero-crossing should occur at **β_c = 0.015 ± 0.0005**.

**Rationale:**  
If the convergence is structural, finer resolution should tighten the bound, not shift the center.

**Test Method:**
```bash
python scripts/run_jacobian_sweep.py --beta-min 0.010 --beta-max 0.020 --beta-steps 100
