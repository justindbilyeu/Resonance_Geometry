# Epistemic Status

- **Confidence:** Medium-high for the core Geometric Plasticity predictions that
  have reproducible simulations (`simulations/ringing_threshold.py`). Empirical
  validation on biological datasets remains pending.
- **Evidence Base:** Synthetic simulations and analytical derivations captured in
  the white paper drafts. No peer-reviewed publications yet.
- **Replication Plan:**
  1. Re-run the lightweight simulations locally or via CI to verify deterministic
     outputs in `figures/`.
  2. Extend to EEG/MEG datasets following the preregistered analysis in
     `docs/codex/policies/prereg_P1.md`.
  3. Document any new datasets or parameter sweeps in `docs/history/HISTORY.md`.
- **Update Cadence:** Quarterly for major theory revisions; continuous for minor
  errata via pull requests.
- **Open Questions:**
  - How do hardware-accelerated estimators (ITPU) change the feasible parameter
    sweep size?
  - Can the REAL cosmological framing be reconciled with the GP formalism, or
    should it remain archived legacy inspiration?
  - What experimental protocols best distinguish GP predictions from alternative
    network adaptation theories?

Please duplicate this box (or link to it) in new documents that make strong
claims so readers can quickly gauge maturity, validation status, and next steps.
