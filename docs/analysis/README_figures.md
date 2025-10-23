# Reproducing paper figure data

Generate the equilibrium sweep outputs and phase-space trajectories locally with:

```bash
make paper-figs
```

This command produces the following artifacts:

- `docs/analysis/eigs_scan_alpha.csv`
- `docs/analysis/eigs_scan_summary.json`
- `figures/eigenvalue_real_vs_alpha.png`
- `docs/analysis/figures/eigenvalue_real_vs_alpha.png`
- `results/phase/traces/traj_alpha_*.json`

CSV/JSON trajectory data are versioned so CI can validate schemas. PNG figures are generated locally or in CI and published as workflow artifacts; they are ignored by git.
