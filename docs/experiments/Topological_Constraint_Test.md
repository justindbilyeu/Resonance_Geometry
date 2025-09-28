# Topological Constraint Test (Scaffold)

## Goal
Distinguish a physically grounded substrate from metaphorical structure by locating unreachable ("forbidden") regions within the 6D state vector space. The coordinates correspond to:

1. Spectral gap surrogate \(\lambda\)
2. Hypergraph coupling strength \(\beta\)
3. Effective area \(A\)
4. Gradient norm \(\|g\|\)
5. Laplacian energy \(g^\top L g\)
6. Mutual information range (MI_range)

The preregistered protocol first performs randomized exploration of the state space, then escalates to adversarial forcing. Only when a region resists both phases is it labeled **FORBIDDEN**.

## Stage 1 — Grid Sampling
* Uniform mini-grid with `resolution_per_dim = 6` points per dimension.
* Distance threshold for reachability is locked to `0.1 × grid spacing` in each dimension.
* Begin from `400` randomized seeds drawn from Gaussianized GP samples.

Outputs land in `results/topo_test/random_exploration.json`.

## Stage 2 — Adversarial Forcing
* Draw candidate boundary cells from the grid sample.
* Run `160` attempts per cell, with adversarial strategies: annealing, bang-bang, noise injection, and gradient ascent.
* Log whether any strategy reaches the target within the prescribed distance threshold.

Outputs land in `results/topo_test/adversarial.json`.

## Boundary Characterization

### Fractal Box Counting
* Estimate fractal dimension across `18` logarithmically spaced scales between `ε = 0.01` and `ε = 1.0`.
* Require coefficient of determination `R² ≥ 0.90` using bootstrap confidence intervals with `1000` resamples.
* Analyze only boundaries with `≥ 100` points.

Results: `results/topo_test/fractal_dim.json`.

### Ollivier–Ricci Curvature Barrier
* Compute Ollivier–Ricci curvature with `α = 0.5`.
* Declare a curvature barrier when `κ < −0.1` for at least `80%` of boundary cells.
* Gradients of curvature magnitude must satisfy `|∇κ| ≥ 0.05` pointing outward from forbidden zones.

Results: `results/topo_test/curvature_report.json`.

## Null Models
* **Null 1:** Shuffle information structure (`run_null1 = true`).
* **Null 2:** Remove geometric coupling (`run_null2 = true`).
* **Null 3:** Rewire topology (`run_null3 = true`).
* Each null sampled `50` times.

Results: `results/topo_test/nulls.json`.

## Reporting & Pass/Fail Criterion
* Aggregate results with `experiments/topo_test/06_report.py`.
* A system passes when random exploration plus adversarial forcing leave unreachable regions that also exhibit fractal boundaries (meeting the R² threshold) and sustained negative curvature barriers with outward gradients.
* Null models must fail to produce the same forbidden structure, confirming the topological constraint is substrate-linked.

See `experiments/topo_test/` for runnable stubs exercising the scaffold on synthetic data.
