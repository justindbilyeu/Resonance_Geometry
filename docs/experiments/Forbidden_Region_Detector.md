# Forbidden Region Detector MVP

This sprint delivers a minimal forbidden-region detector on top of a self-contained toy GP evolution. The detector:

- Runs randomized exploration over a 4D grid spanning the parameters $(\lambda, \beta, A)$ and the emergent $\lVert g \rVert$ norm bin.
- Logs which cells were visited and marks the remainder as candidate forbidden regions.
- Estimates the largest connected forbidden structure via the official
  [NetworkX](https://networkx.org/) library rather than the historical shim,
  ensuring we exercise the battle-tested connected components routines.
- Emits a JSON summary for downstream pipelines.
- Saves lightweight heatmap projections as quick-look PNGs.

## Success / Fail Criteria

- If the fraction of forbidden cells is **≥ 1%**, the detector recommends **ESCALATE**.
- If the fraction is **< 1%**, it recommends **STAND_DOWN**.

## How to Run

```bash
python experiments/forbidden_region_detector.py
python experiments/adversarial_forcing.py
```

## Outputs

Running the detector writes artifacts to:

- `results/forbidden_v0/forbidden_summary.json`: summary metrics and decision hint.
- `results/forbidden_v0/visited_4d.npy`: boolean occupancy grid.
- `figures/forbidden_v0/forbidden_lam_beta.png`: % forbidden per $(\lambda, \beta)$ slice.
- `figures/forbidden_v0/forbidden_lam_A.png`: % forbidden per $(\lambda, A)$ slice.
- `figures/forbidden_v0/forbidden_beta_A.png`: % forbidden per $(\beta, A)$ slice.

The adversarial forcing pipeline consumes the JSON/Numpy outputs and writes:

- `results/forbidden_v0/adversarial_report.json`: summary of strategy hits vs. tested forbidden cells.
