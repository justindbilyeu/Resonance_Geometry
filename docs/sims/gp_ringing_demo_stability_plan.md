# GP Ringing Demo Stability Repair Plan

## Context

Running `experiments/gp_ringing_demo.py` triggered overflows and NaNs inside the
`simulate_coupled` oscillator. When `lam` approached its maximum coupling, the
linear AR(2) recurrence became unstable, producing non-finite values that caused
`numpy.histogram2d` to crash while estimating mutual information. The demo could
not complete as a result.

## Repair Actions

1. **Stabilise the oscillator** – introduce a gentle saturation non-linearity in
   `simulate_coupled` so the state variables remain finite even when the coupling
   temporarily pushes the linear dynamics past their stability boundary.
2. **Backfill tests** – cover both the CLI helper for the spin-foam smoke test
   and the MI demo so the regression is detected automatically.
3. **Regression check** – rerun the pytest suite and exercise the demo script to
   confirm the outputs are generated without warnings or crashes.

## Expected Outcomes

- `python experiments/gp_ringing_demo.py` completes successfully and writes the
  summary artefacts.
- The new tests fail if either the CLI helpers or the synthetic oscillator
  return non-finite data, providing guard rails for future edits.
