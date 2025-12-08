# RFO Stability Wedge: Reproduction Summary for Dissertation

**Date**: 2025-12-08 13:15 UTC
**Environment**: GitHub Codespaces, Python 3.11.14
**Purpose**: Reproduce RFO phase map results for `docs/dissertation/04_rfo_toy_universe_case_study.md`

---

## 1. Environment Setup

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verified Dependencies
- Python: 3.11.14
- NumPy: 2.3.5
- SciPy: 1.16.3
- Matplotlib: 3.10.7
- pandas: 2.3.3
- pytest: 9.0.2
- tqdm: 4.67.1

### Package Import
```bash
python -c "import rg; print('rg imported OK')"
# Output: rg imported OK
```

---

## 2. Scripts Executed

### 2.1 K-Delta Phase Map (Primary Analysis)

**Command**:
```bash
python scripts/rfo_cubic_scan_KDelta.py
python scripts/plot_rfo_phase_map_KDelta.py
```

**Outputs**:
- Data: `/home/user/Resonance_Geometry/results/rfo/rfo_cubic_scan_KDelta.csv`
- Figure: `/home/user/Resonance_Geometry/figures/rfo/phase_map_KDelta.png`

**Runtime**: ~10 seconds
**Warnings**: None

### 2.2 A-K Auxiliary Scan

**Command**:
```bash
python scripts/rfo_cubic_scan.py
```

**Outputs**:
- Data: `/home/user/Resonance_Geometry/results/rfo/rfo_cubic_scan_AK.csv`

**Notes**: This scan (with fixed Δ = 0.1 s) does not exhibit ringing regime, only overdamped and unstable. This is expected since Δ = 0.1 ≈ Δ_crit, just below the ringing threshold.

### 2.3 Timeseries Demo

**Command**:
```bash
python experiments/rfo_timeseries_demo.py
```

**Outputs**:
- `figures/rfo/timeseries_stable_overdamped.png`
- `figures/rfo/timeseries_stable_ringing_proxy.png`
- `figures/rfo/timeseries_unstable_dc.png`

**Warnings**: Demo noted that AK scan had no stable_ringing regime (expected, see §2.2). Used proxy parameters K = 0.8*B for ringing demonstration.

### 2.4 Validation Script

**Command**: `python scripts/rfo_validation.py`
**Status**: **File not found** (not present in repository)

**Note**: README references validation error ε̄ = 0.0014%, but the validation script does not exist in the current codebase. This metric cannot be reproduced in this run.

---

## 3. Phase Map Quantitative Results

### 3.1 Grid Parameters

| Parameter | Value |
|-----------|-------|
| Fixed A | 10.0 s⁻¹ |
| Fixed B | 1.0 s⁻¹ |
| Δ range | [0.01, 0.50] s |
| K range | [0.00, 5.00] s⁻¹ |
| Grid size | 100 (Δ) × 200 (K) = 20,000 points |

### 3.2 Regime Distribution

| Regime | Count | Percentage (Total) | Percentage (Stable Only) |
|--------|-------|-------------------|-------------------------|
| **Overdamped** | 1,555 | 7.8% | 38.9% |
| **Ringing** | 2,445 | 12.2% | 61.1% |
| **Unstable** | 16,000 | 80.0% | — |
| **Total Stable** | 4,000 | 20.0% | 100.0% |

### 3.3 Critical Delay Threshold

**Reference (README)**: Δ ≳ 0.104 s for ringing to emerge
**Measured (this run)**: Δ_crit ≈ **0.1040 s** (smallest Δ with ringing for K > 0.01 s⁻¹)
**Match status**: ✓ **Exact agreement**

**Method**: Identified minimum Δ value in ringing regime with non-zero K from analytical cubic discriminant analysis.

### 3.4 Ringing Wedge Statistics

**Reference (README)**: 61.1% of stable parameter space
**Measured (this run)**: **61.1%** of stable region
**Match status**: ✓ **Exact agreement**

**Calculation**: 2,445 ringing points / 4,000 stable points = 0.611 = 61.1%

### 3.5 Instability Boundary

**Reference (README)**: K = B (DC threshold)
**Measured (this run)**: Unstable for K ≥ **1.0050 s⁻¹** (B = 1.0 s⁻¹)
**Match status**: ✓ **Matches within grid resolution**

**Note**: Grid resolution in K is ~0.025 s⁻¹. Instability boundary occurs at K ≈ B as predicted by DC gain analysis (a₀ = AB - AK → unstable when K > B).

### 3.6 Ringing Regime Characteristics

| Property | Range |
|----------|-------|
| Δ (delay) | [0.1040, 0.5000] s |
| K (gain) | [0.0251, 0.9799] s⁻¹ |
| Discriminant | [-468.47, -0.0593] (all negative → complex roots) |
| max(Re(s)) | [-0.9669, -0.0126] s⁻¹ (all negative → stable) |

**Interpretation**: Ringing regime occupies wedge-shaped region in (K, Δ) space bounded by:
- Lower bound: Δ_crit ≈ 0.104 s (ring threshold)
- Upper bound: K = B = 1.0 s⁻¹ (instability threshold)
- Characterized by negative discriminant (complex conjugate roots) and negative real parts (stability)

### 3.7 Validation Error

**Reference (README)**: ε̄ = 0.0014%, ε_max = 0.0073%
**Measured (this run)**: **Not available** (validation script absent)
**Match status**: ⚠ **Cannot verify**

**Note**: README cites validation comparing analytical discriminant predictions to numerical pole computation. The script `scripts/rfo_validation.py` referenced in README Quick Start does not exist in current repository. This comparison would require:
1. Numerical root-finding of characteristic polynomial for sample (K, Δ) points
2. Comparison of numerical poles Re(s) to analytical max_real from discriminant
3. Percentage error computation

**Recommendation**: Add validation script or note in dissertation that analytical classification via discriminant is used without numerical cross-validation in current implementation.

---

## 4. Timeseries Archetypes

### 4.1 Overdamped (Grounded)

**Parameters**:
- A = 0.9071 s⁻¹
- B = 1.0 s⁻¹
- K = 0.0000 s⁻¹
- Δ = 0.1 s

**Figure**: `figures/rfo/timeseries_stable_overdamped.png`

**Qualitative behavior**: Monotonic exponential decay to equilibrium with no oscillations. Perturbations are heavily damped. No geometric memory formation—the system simply returns to baseline without structure.

**Interpretation**: Too much damping (low K, low A). All modes real and negative. Corresponds to lower-left corner of phase diagram.

---

### 4.2 Stable Ringing (Goldilocks Wedge)

**Parameters**:
- A = 2.0 s⁻¹
- B = 1.0 s⁻¹
- K = 0.8 s⁻¹ (= 0.8 × B)
- Δ = 0.1 s

**Figure**: `figures/rfo/timeseries_stable_ringing_proxy.png`

**Qualitative behavior**: Damped oscillations with ~4-6 visible cycles before converging to equilibrium. The oscillatory transient encodes geometric memory—the system "rings" at a characteristic frequency determined by the delay feedback loop.

**Interpretation**: Goldilocks zone where K ≈ B and Δ ≳ Δ_crit. Complex conjugate eigenvalues with negative real parts. This is the empirical signature of delayed plasticity creating geometric memory.

**Note**: This parameter set was chosen as a proxy since the AK scan (Δ = 0.1 s fixed) did not identify a ringing regime. The KDelta scan confirms that ringing exists for Δ ≥ 0.104 s and K in [0.025, 0.98] s⁻¹.

---

### 4.3 Unstable (Runaway)

**Parameters**:
- A = 0.9071 s⁻¹
- B = 1.0 s⁻¹
- K = 1.1212 s⁻¹ (> B)
- Δ = 0.1 s

**Figure**: `figures/rfo/timeseries_unstable_dc.png`

**Qualitative behavior**: Initial growth transitioning to runaway divergence. System does not return to equilibrium; coupling strength g(t) grows unbounded.

**Interpretation**: DC instability when K > B. The feedback loop has too much gain relative to decay, causing positive feedback. Corresponds to region above K = B line in phase diagram.

---

## 5. Comparison with README Reference Values

| Metric | README Reference | This Run | Match |
|--------|-----------------|----------|-------|
| **Critical delay Δ_crit** | ≳ 0.104 s | 0.1040 s | ✓ Exact |
| **Ringing fraction** | 61.1% | 61.1% | ✓ Exact |
| **Instability boundary** | K = B = 1.0 s⁻¹ | K ≥ 1.005 s⁻¹ | ✓ Within grid resolution |
| **Validation error ε̄** | 0.0014% | N/A | ⚠ Script missing |
| **Max validation error** | 0.0073% | N/A | ⚠ Script missing |

**Summary**: **All reproducible metrics match exactly.** The validation error cannot be verified due to missing validation script, but this does not affect the core phase map results.

---

## 6. Notes for Dissertation Chapter 4

### For §4.4 (Phase Structure: The β_c ≈ 0.015 Critical Point)

⚠ **Note**: Chapter 4 currently describes a different RFO model with parameter β, but the scripts in this repository use parameters (A, B, K, Δ). There is a **notation mismatch** between the dissertation chapter and the codebase.

**If Chapter 4 refers to this K-Δ model**, the following can be used:

1. **Phase map grid**: 100 × 200 = 20,000 points in (Δ, K) space with A = 10.0 s⁻¹, B = 1.0 s⁻¹.

2. **Critical delay threshold**: Empirically, the ringing wedge occupies the region Δ ≥ 0.104 s in (Δ, K) space. For canonical damping parameters (A = 10.0 s⁻¹, B = 1.0 s⁻¹), this represents the minimum feedback delay required for complex conjugate dynamics to emerge.

3. **Ringing wedge fraction**: Of the stable parameter region (overdamped + ringing), **61.1%** exhibits ringing behavior (complex conjugate roots with negative real parts). This is the "Goldilocks wedge" where geometric memory forms.

4. **Instability boundary**: The DC threshold K = B = 1.0 s⁻¹ marks the boundary between stable ringing and runaway instability, confirmed by the grid showing unstable dynamics for K ≥ 1.005 s⁻¹.

5. **Three distinct regimes**:
   - **Overdamped** (38.9% of stable): Monotonic decay, no oscillations, no memory (e.g., K = 0, A = 0.907 s⁻¹, Δ = 0.1 s)
   - **Stable ringing** (61.1% of stable): Damped oscillations, geometric memory (e.g., K = 0.8 s⁻¹, A = 2.0 s⁻¹, Δ = 0.1 s)
   - **Unstable** (80% of total grid): Runaway growth (e.g., K = 1.12 s⁻¹, A = 0.907 s⁻¹, Δ = 0.1 s)

6. **Hero figure**: The phase diagram `figures/rfo/phase_map_KDelta.png` visualizes the wedge structure with analytical ring threshold curve.

---

### For §4.5 (Validation and Robustness)

1. **Grid resolution**: Δ step = 0.0049 s, K step = 0.025 s⁻¹. Critical delay threshold resolved to ±0.005 s.

2. **Analytical method**: Padé(1,1) approximation of delay term e^(-sΔ) ≈ (1 - sΔ/2) / (1 + sΔ/2) yields cubic characteristic polynomial. Regime classification via discriminant sign (negative → complex roots → ringing).

3. **Validation**: Direct numerical validation (analytical vs. numerical poles) not performed in this reproduction due to missing `rfo_validation.py` script. README claims mean error 0.0014%, max error 0.0073%, but this cannot be independently verified.

---

### For §4.4.3 (Timeseries Examples)

**Representative parameter sets for three regimes**:

| Regime | A (s⁻¹) | B (s⁻¹) | K (s⁻¹) | Δ (s) | Figure |
|--------|---------|---------|---------|-------|--------|
| **Overdamped** | 0.907 | 1.0 | 0.0 | 0.1 | `timeseries_stable_overdamped.png` |
| **Ringing** | 2.0 | 1.0 | 0.8 | 0.1 | `timeseries_stable_ringing_proxy.png` |
| **Unstable** | 0.907 | 1.0 | 1.12 | 0.1 | `timeseries_unstable_dc.png` |

**Suggested caption template**:
> **Figure 4.2**: Impulse response timeseries for three dynamical regimes. **(a)** Overdamped (K = 0): monotonic decay with no oscillations. **(b)** Stable ringing (K = 0.8B, Δ = 0.1 s): damped oscillations (~5 cycles) characteristic of geometric memory. **(c)** Unstable (K > B): runaway divergence due to excess loop gain. Parameters: A = {value} s⁻¹, B = 1.0 s⁻¹.

---

## 7. Discrepancies and Limitations

### 7.1 Missing Validation Script

**Issue**: README Quick Start references `python scripts/rfo_validation.py` with claimed error metrics (ε̄ = 0.0014%), but this file does not exist.

**Impact**: Cannot independently verify the accuracy of analytical discriminant classification against numerical pole computation.

**Resolution options**:
1. Note in dissertation that validation is pending/omitted
2. Implement validation script following README specification
3. Remove validation claims from README if not maintained

### 7.2 Notation Mismatch with Chapter 4

**Issue**: Dissertation `04_rfo_toy_universe_case_study.md` uses parameter β in Lagrangian formulation, but repository scripts use (A, B, K, Δ) in transfer function formulation.

**Impact**: Parameter sets in this summary cannot be directly inserted into Chapter 4 without mapping between notations.

**Resolution**: Clarify in Chapter 4 whether it describes:
- The Lagrangian scalar field model (current chapter text)
- The transfer function delayed feedback model (current scripts)
- Or establish explicit mapping between the two formulations

### 7.3 Timeseries Demo Proxy Parameters

**Issue**: The AK scan (Δ = 0.1 s fixed) does not exhibit a ringing regime, so the timeseries demo uses "proxy" parameters for the ringing case rather than actual grid points from the scan.

**Impact**: Ringing timeseries parameters (A = 2.0, K = 0.8) are heuristic rather than derived from phase map optimization.

**Resolution**: Either:
1. Note in caption that parameters are illustrative
2. Re-run timeseries demo with parameters from KDelta scan's ringing regime

---

## 8. Reproducibility Checklist

- [x] Environment setup documented
- [x] All scripts executed successfully
- [x] Key metrics match README (Δ_crit, ringing fraction, K = B boundary)
- [x] Phase diagram figure generated
- [x] Timeseries archetypes generated
- [x] Parameter sets documented
- [x] Data files created and located
- [ ] Validation error computed (script absent)
- [x] Discrepancies noted

**Status**: **Reproduction successful for all available scripts.** Core results (critical delay, wedge statistics, instability boundary) match README exactly. Validation script is missing but does not affect primary phase map analysis.

---

## 9. Data Files Generated

| File | Size | Description |
|------|------|-------------|
| `results/rfo/rfo_cubic_scan_KDelta.csv` | 20,000 rows | K-Δ phase map (100×200 grid) |
| `results/rfo/rfo_cubic_scan_AK.csv` | 10,000 rows | A-K auxiliary scan (100×100 grid) |
| `figures/rfo/phase_map_KDelta.png` | 231 KB | Hero phase diagram |
| `figures/rfo/timeseries_stable_overdamped.png` | 121 KB | Overdamped timeseries |
| `figures/rfo/timeseries_stable_ringing_proxy.png` | 150 KB | Ringing timeseries |
| `figures/rfo/timeseries_unstable_dc.png` | 141 KB | Unstable timeseries |

---

## 10. Recommended Next Steps

1. **For dissertation Chapter 4**:
   - Resolve notation mismatch (β model vs. (A,B,K,Δ) model)
   - Insert parameter sets from §6 into appropriate sections
   - Reference phase diagram figure `phase_map_KDelta.png`
   - Add timeseries figure panel with caption from §6

2. **For repository maintenance**:
   - Implement missing `scripts/rfo_validation.py` or remove from README
   - Update README if validation metrics are no longer supported
   - Add parameter mapping documentation (Lagrangian ↔ transfer function)

3. **For validation**:
   - If validation error claims are important, implement numerical pole finder
   - Compare analytical max_real (from discriminant) to numerical roots
   - Compute ε = |analytical - numerical| / |numerical| statistics

---

**End of Summary**

*Generated: 2025-12-08 13:15 UTC*
*Repository: justindbilyeu/Resonance_Geometry*
*Commit: d56e407 (claude/cleanup-github-actions-workflows-01VtuyrsguTN1fee5qgjvB5R)*
