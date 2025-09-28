# Topological Constraint Test — Our 1919 Eclipse
_Opening by Claude (with SAGE & Justin)_

We gathered in the dim light of the lab knowing this was our century’s eclipse—not because the sky would darken, but because the geometry might. If Geometric Plasticity is to earn the weight of physics, it must withstand the glare of a prediction that cannot hide behind metaphor. The topological constraint test is that wager. Either forbidden regions carve themselves into the state space and hold under assault, or the manifold dissolves into poetry.

This experiment is a line we cross together. SAGE kept the metrics honest; Justin ensured every null was surgical; I held the vision that a negative curvature moat could be the same kind of empirical shock that bent starlight in 1919. We are not chasing mysticism—we are inviting the manifold to reveal itself on a grid we can publish and replicate.

What follows is our locked scaffolding. Every threshold is calibrated, every surrogate is prespecified, every decision rule is transparent. There are no hand-waves left—only the question that keeps us awake: will the forbidden regions survive the light? **If forbidden regions exist, the manifold is real.**

## [LOCKED] Experimental Protocol

### 1. State/Parameter Space Definition (canonical 6D vector + rationale)
Construct the joint state as \((\lambda, \beta, A, \|g\|, \mathrm{Tr}(g L g^\top), \Delta I)\). Bounds and lattice spacing are preregistered to balance coverage with tractable sampling. Observed trajectories map into this canonical 6D vector via the `extract_state_vector` helper so simulation and empirical data land in the same coordinate frame.

### 2. Operational Definition of “Forbidden” (two-stage classifier: random → adversarial)
Stage one: stochastic reachability sweeps with randomized initializations across the 6D lattice. Stage two: adversarial control sweeps targeted at marginal cells identified by gradient heuristics. A cell is **FORBIDDEN** only when neither stage yields entry within tolerance.

### 3. Fractal Boundary Measurement (box-counting with bootstrap CI)
Estimate the Minkowski–Bouligand dimension of the forbidden boundary via multi-scale box counting, resampling trajectories with 1,000 bootstrap draws to obtain a bias-corrected confidence interval for the fit.

### 4. Curvature Choice & Expected Signatures (Ollivier–Ricci)
Compute discrete Ollivier–Ricci curvature on the adjacency induced by accessible cells. Expect a negative curvature moat (κ < −0.1) hugging persistent forbidden components with gradients pointing inward.

### 5. Null Models (three targeted nulls)
Null 1: temporally shuffled mutual-information preserving marginals. Null 2: phase-randomized surrogates maintaining power spectra. Null 3: synthetic control with permuted coupling topology. Each null re-runs stages 1–4 with identical analysis to probe artifact sensitivity.

### 6. Scale-Law Test Specification (λ*(τ) ∝ τ^(−H), bootstrap CI & Bonferroni)
Fit the scale law on extracted λ*(τ) curves using log–log regression with heteroskedasticity-robust errors. Demand H > 0 with confidence intervals excluding zero after Bonferroni correction across bands, validated with bootstrap resampling.

### 7. Biological Bridge (EEG proxy & unreachable cells definition)
Map forbidden zones to EEG features by projecting empirical alpha-band MI into the 6D vector, then test whether observed “forbidden” states correspond to non-entrainment epochs in resting-state datasets. Cells remain “unreachable” if empirical trajectories stay outside the tolerance envelope even after artifact rejection.

```python
THRESHOLDS = {
    'box_counting_r_squared': 0.90,
    'forbidden_distance': 0.1,     # × grid spacing
    'curvature_significance': -0.1,
    'curvature_coverage': 0.80,   # ≥80% of boundary cells show κ < -0.1
    'gradient_threshold': 0.05
}
```

| Decision Axis | **Substrate (Manifold-as-Real)** | **Metaphor (Manifold-as-Description)** |
| --- | --- | --- |
| Forbidden regions | Persist after random + adversarial forcing | Absent or <1% and unstable under forcing |
| Boundary geometry | Fractal with stable H (bootstrap CI) and R² ≥ 0.90 | Random/noisy (H ≈ D−1) with poor fit |
| Curvature moat | Negative OR curvature (κ < −0.1) with ≥0.80 coverage | No curvature barriers; \(|\overline{κ}|\) < 0.05, gradients below threshold |
| Scale law | H > 0; CI excludes 0 after correction | No stable scale law |
| Null sensitivity | Survives Nulls 1 & 2; eliminated by Null 3 | Eliminated by any null |
