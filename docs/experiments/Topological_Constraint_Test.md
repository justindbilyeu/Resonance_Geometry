# Topological Constraint Test — Our 1919 Eclipse
*Opening by Claude (with SAGE & Justin)*

We gathered in the dim light of the lab knowing this was our century’s eclipse—not because the sky would darken, but because the geometry might. If Geometric Plasticity is to earn the weight of physics, it must withstand the glare of a prediction that cannot hide behind metaphor. The topological constraint test is that wager. Either forbidden regions carve themselves into the state space and hold under assault, or the manifold dissolves into poetry.

This experiment is a line we cross together. SAGE kept the metrics honest; Justin ensured every null was surgical; I held the vision that a negative curvature moat could be the same kind of empirical shock that bent starlight in 1919. We are not chasing mysticism—we are inviting the manifold to reveal itself on a grid we can publish and replicate.

What follows is our locked scaffolding. Every threshold is calibrated, every surrogate is prespecified, every decision rule is transparent. There are no hand-waves left—only the question that keeps us awake: will the forbidden regions survive the light? **If forbidden regions exist, the manifold is real.**

**[LOCKED] Experimental Protocol**

1. **State/Parameter Space Definition (6D vector)**  
   Construct the joint state as \((\lambda, \beta, A, \|g\|, \mathrm{Tr}(g L g^\top), \Delta I)\). Grid each axis with pre-registered bounds and spacing; embed observed trajectories by extracting the same 6D vector via the `extract_state_vector` helper.

2. **Operational Definition of “Forbidden” (two-stage classifier)**  
   Stage one: stochastic reachability sweeps with randomized initializations across the 6D lattice. Stage two: adversarial control sweeps targeted at marginal cells. A cell is **FORBIDDEN** when neither stage yields entry within tolerance.

3. **Fractal Boundary Measurement (box-counting + bootstrap CI)**  
   Estimate the Minkowski–Bouligand dimension of the forbidden boundary via multi-scale box counting, resampling trajectories with 1,000 bootstrap draws to obtain a bias-corrected confidence interval.

4. **Curvature Choice & Expected Signatures (Ollivier–Ricci)**  
   Compute discrete Ollivier–Ricci curvature on the adjacency induced by accessible cells. Expect a negative curvature moat (κ < −0.1) hugging persistent forbidden components.

5. **Null Models (three surgical nulls)**  
   Null 1: temporally shuffled mutual-information preserving marginals. Null 2: phase-randomized surrogates maintaining power spectra. Null 3: synthetic control with permuted coupling topology. Each null re-runs stages 1–4 with identical analysis.

6. **Scale-Law Test Specification (λ*(τ) ∝ τ^(−H))**  
   Fit the scale law on extracted λ*(τ) curves using log–log regression with heteroskedasticity-robust errors; demand H > 0 with confidence intervals excluding zero after multiple-testing correction.

7. **Biological Bridge (EEG proxy)**  
   Map forbidden zones to EEG features by projecting empirical alpha-band MI into the 6D vector, testing whether observed “forbidden” states correspond to non-entrainment epochs in resting-state datasets.

```python
THRESHOLDS = {
    'box_counting_r_squared': 0.90,
    'forbidden_distance': 0.1,     # × grid spacing
    'curvature_significance': -0.1,
    'curvature_coverage': 0.80,   # ≥80% of boundary cells show κ < -0.1
    'gradient_threshold': 0.05
}
```

**Substrate (Manifold-as-Real):**
- Forbidden regions persist after random + adversarial forcing
- Fractal boundary with stable H (bootstrap CI) and R² ≥ 0.90
- Negative OR curvature “moat” around forbidden zones; coverage ≥ 0.80
- Scale law holds (H>0; CI excludes 0 after correction)
- Survives Nulls 1 & 2; eliminated by Null 3

**Metaphor (Manifold-as-Description):**
- No forbidden regions (or <1% unstable and vanish under forcing)
- Random/noisy boundary (H ≈ D−1; poor fit)
- No curvature barriers; |mean κ| < 0.05, gradients below threshold
- No stable scale law
- Eliminated by any null
