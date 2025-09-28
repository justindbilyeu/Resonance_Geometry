# Multi-Frequency GP Analysis Plan

**Purpose:** Extend Geometric Plasticity (GP) analyses beyond the alpha band to test resonance-information coupling across canonical neurophysiological frequencies.

**Epistemic Status:** [TESTABLE-HYPOTHESIS]

## Frequency Bands

```python
FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 80.0)
}

GP_BANDS = {
    # Optional finer subdivisions if the dataset supports stable estimates
    "alpha_low": (8.0, 10.0),
    "alpha_high": (10.0, 12.0),
    "gamma_high": (80.0, 120.0)
}
```

## Analysis Tracks

1. **Band-specific MI / λ\* Extraction**  
   - Compute mutual information (MI) time-series per band using matched filters.  
   - Extract λ\* (coupling threshold) and hysteresis metrics within each band.  
   - Compare λ\* ordering across bands to probe timescale sensitivity.

2. **Cross-Frequency Coupling Matrices**  
   - Assemble MI-based coupling matrices that capture directed influence between band-limited signals.  
   - Evaluate stability with phase-randomized surrogates and permutation controls.  
   - Produce frequency–frequency resonance heatmaps for downstream Mapper ingestion.

3. **Frequency-Specific Hysteresis Profiles**  
   - Estimate hysteresis area, loop asymmetry, and transition sharpness for each band.  
   - Track trajectory drift across λ sweeps to flag metastable basins.  
   - Correlate hysteresis fingerprints with cross-band coupling strengths.

## Hypotheses

- **H1:** λ\* decreases monotonically with intrinsic oscillation frequency (delta > … > gamma), reflecting faster bands requiring less coupling to align.  
- **H2:** Cross-frequency coupling matrices exhibit significant asymmetry, with alpha→gamma driving exceeding gamma→alpha under task-driven conditions.  
- **H3:** Bands with higher hysteresis area display stronger Betti-1 persistence in Mapper reconstructions, linking resonance memory to topological signatures.

## Phased Roadmap

- **Week 1:** Finalize band definitions, filtering pipeline, and surrogate library. Validate MI extraction on synthetic benchmarks.  
- **Week 2:** Implement λ\* detection and hysteresis metrics per band; integrate with existing GP demo outputs.  
- **Week 3:** Build cross-frequency coupling matrices and surrogate-based significance tests; document Mapper-ready schema.  
- **Week 4:** Aggregate results, run null models, and draft preregistration appendix with decision thresholds.

## Expected Outcomes

- Ranked λ\* thresholds across canonical bands with confidence intervals.  
- Cross-frequency coupling matrices annotated with surrogate p-values.  
- Hysteresis profile library (area, symmetry, sharpness) feeding into Mapper addendum.  
- Annotated dataset `multi_frequency_results.json` ready for ingestion.

## Technical Specifications

- **Filtering:** Zero-phase FIR bandpass filters (Kaiser window, ≥60 dB stopband) with edge padding to suppress ringing.  
- **Surrogates:** Phase-randomized Fourier surrogates (per band) plus time-shift permutations for cross-band coupling controls.  
- **Multiple Comparisons:** Benjamini–Hochberg FDR (q=0.05) within bands; Bonferroni across bands for λ\* detection.  
- **λ\* Detection:** Identify bifurcation by locating maximal derivative of MI(λ); confirm with hysteresis crossing.  
- **MI Estimation:** Kraskov k-NN estimator with k=4 and bias correction; bootstrap (N=500) for CI.

## Connection to Resonance Axioms

- **Axiom 1 (Resonance precedes geometry):** Frequency-resolved λ\* establishes the coupling level at which information flow locks geometry per band.  
- **Axiom 2 (Geometry encodes memory):** Band-specific hysteresis profiles quantify geometric memory in the coupling manifold.  
- **Axiom 3 (Coherence bridges systems):** Cross-frequency coupling matrices expose the coherence channels linking subsystems across timescales.

## Minimal Function Signatures (Pseudocode)

```python
def extract_band_mi(time_series, band, fs, filter_cache=None):
    """Return band-limited MI time-series and metadata for λ* estimation."""


def estimate_lambda_star(mi_ts, lambda_schedule):
    """Compute λ* and hysteresis metrics from MI trajectories."""


def compute_cross_frequency_coupling(band_mi_dict):
    """Return coupling matrix plus surrogate-based significance levels."""


def summarize_hysteresis_profiles(band_metrics):
    """Produce hysteresis area, asymmetry, and sharpness per band."""


def assemble_multi_frequency_results(metadata, band_metrics, coupling_matrix):
    """Create Mapper-ready JSON following the Resonance Mapper schema."""
```
