# Pre-Registration: P1 Ringing Threshold (v1.2)

## Hypothesis
There exists λ* such that alpha-band (8–12 Hz) power of windowed MI(t) increases sharply relative to λ < λ*, and a hysteresis gap emerges between up- and down-sweeps.

## Design
- Synthetic demo (this repo) for continuous integration.
- Real EEG pilot (external repo or folder) with eyes-open/closed blocks as mild drive proxy; λ is an abstract coupling index mapped to block-wise manipulation (documented separately).

## Locked Parameters
- Sampling rate: 128 Hz (synthetic).
- Window size: 256 samples (~2.0 s), hop: 64 samples (~0.5 s).
- Estimator: histogram MI, bins=64, nats.
- PSD: Welch, nperseg=128, noverlap=64; alpha band=8–12 Hz.
- Sweep: up (λ: 0→0.9) then down (0.9→0); 60 s each.

## Endpoints
- **Primary:** Alpha-band MI power vs λ; success if ≥3× rise at λ* with permutation p<0.01 (surrogates preserve autocorr via IAAFT/AR).
- **Secondary:** Hysteresis loop area > 0 with p<0.01.

## Multiple Comparisons
- If exploring extra bands or channels: Benjamini–Hochberg FDR at q=0.05.

## Artifact/Outlier Rules
- Synthetic: none.
- Real EEG: fixed thresholds and ICA rejection defined before data access.

## Failure Criteria
- No significant alpha-power rise (p≥0.01) or effect size < 0.5 → **fail**.
- Hysteresis loop area not >0 or not significant → **fail**.
- Publish null; do not retrofit theory to fit outcomes.

## Blinding & Roles
- Parameter locking and code freeze before data loading.
- Separate persons for code vs data labeling (for EEG pilot).
