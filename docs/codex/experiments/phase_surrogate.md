# Surrogate Ringing Map (AR(2))

This experiment provides a **fast proxy** for the ringing boundary using a 2nd-order AR model:
- radius `r = 1 − η`, angle `θ = α π`, yielding AR(2) coefficients `φ1 = 2 r cos θ`, `φ2 = − r²`.
- We drive the system with an impulse and detect ringing via:
  - PSD peak ≥ 6 dB (baseline = median excluding DC),
  - ≥ 2 overshoots,
  - optional damping ratio ζ from a Hilbert-envelope fit.

**Outputs:** `results/phase_map_surrogate/phase_map.csv|png`

**Why this exists:** It’s ~100× faster than full RWP, so you can sweep coarse grids, then confirm interesting slices with `scripts/run_phase_sweep.py` (the full model).

**Calibration to RWP:** Treat ζ as a **monotone proxy** for loop-gain K. For a slice in (α,η), run the full RWP sims, estimate `τ_geom`, and report both `ζ_surrogate` and `K=η τ_geom / λ_eff` to align boundaries.
