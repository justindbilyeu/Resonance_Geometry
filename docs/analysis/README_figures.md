# Reproducing paper figure data

Generate the equilibrium sweep outputs and phase-space trajectories locally with:

```bash
make paper-figs
```

This command produces the following artifacts:

- `docs/analysis/eigs_scan_alpha.csv`
- `docs/analysis/eigs_scan_summary.json`
- `figures/eigenvalue_real_vs_alpha.png`
- `results/phase/traces/traj_alpha_*.json`

Only the CSV/JSON data files are committed. The PNG is generated locally and uploaded as a pull-request artifact by CI.
