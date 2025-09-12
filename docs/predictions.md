# Resonance Geometry → Geometric Plasticity: Predictions (v1.2)

We treat GP as gradient flow on an information manifold. Couplings **g** adapt to maximize an information target with regularizers (energy, smoothness), yielding testable phenomena.

## P1 — Ringing Threshold (Primary)
**Claim.** As coupling λ increases, windowed MI(t) exhibits an onset of oscillatory power with a sharp rise at λ* (the “ringing” threshold).

**Preregistered endpoint (alpha band 8–12 Hz):**
- Welch PSD of MI(t), power integrated over 8–12 Hz.
- Up-sweep vs down-sweep curves vs λ.
- Success: ≥3× increase in alpha-band MI power at λ* (d ≥ 0.5 acceptable; see prereg for exact thresholding) with permutation-based p < 0.01 vs null surrogates that preserve autocorrelation.

## P2 — Hysteresis
**Claim.** Under slow parameter sweeps, MI-power vs λ differs between up- and down-sweeps (loop area > 0).

**Endpoint.** Signed loop area > 0 with p < 0.01 vs phase-randomized or IAAFT surrogates.

## P3 — Motif Selection
**Claim.** Networks under GP select broadcast vs modular motifs depending on drive statistics.

**Endpoint.** Change in graph measures (modularity, participation coefficient) with MI-based functional edges; effect size d ≥ 0.5, p < 0.01 (corrected).

## Controls & Nulls (summarized)
- Locked window/hop, estimator params (bins, k).
- Surrogates preserving autocorrelation (IAAFT or AR).
- Task-matched controls when applicable.
- Multiple comparison correction (FDR).

See `docs/prereg_P1.md` for the full analysis plan and failure criteria.
