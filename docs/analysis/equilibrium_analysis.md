# Equilibrium & Eigenvalue Scan (Playground)

This report accompanies `scripts/equilibrium_analysis.py`.

- **Method:** solve non-trivial equilibria from ω₀² φ = K₀ sin(α φ); linearize dynamics; compute max Re(λ) across equilibria per α.
- **Default params:** ω₀²=1.0, γ=0.08, K₀=1.2; α ∈ [0.25, 0.55]

Artifacts:
- `docs/analysis/eigs_scan_alpha.csv`
- `docs/analysis/eigs_scan_summary.json`
- `docs/analysis/figures/eigenvalue_real_vs_alpha.png`

Interpretation guidance:
- If max Re(λ) < 0 across the observed RTP window (α≈0.35), it supports a **non-Hopf / global reorganization** mechanism.
- If a clean zero-crossing occurs near RTP, it supports a **local Hopf** mechanism (unlikely per prior runs).
