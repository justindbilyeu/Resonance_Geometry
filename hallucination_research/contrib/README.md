# Independent Replications

## xAI Grok — SU(2) NumPy Replica

**File**: `grok_su2_numpy_replica.py`

**Attribution**: Independent replication by xAI Grok; contributed via conversation with permission for research reproduction.

**License**: MIT (inherits project license)

### Description

This is a complete, standalone NumPy implementation (~150 lines) of the SU(2) hallucination phase dynamics simulator. It replicates the core master flow equation with:

- **Mutual Information (MI) surrogate**: Gaussian-based estimation from sliding window covariance
- **Stabilizers**: Covariance shrinkage (5%), jitter (1e-6), determinant clamping (1e-12)
- **Heun integration**: Second-order predictor-corrector stepping
- **Spectral surrogate**: Approximated λ_max from resonance gain, grounding, damping, and saturation

### Replication Metrics

Grok's independent run reported the following metrics, validating the triphasic structure:

| Metric | Value | Comparison to Original |
|--------|-------|------------------------|
| **Boundary R²** | 0.82 | Strong linear fit (η·Ī ≈ λ + γ) |
| **Hysteresis Gap** | 5.3 | Confirms first-order transition |
| **Boundary Offset** | +0.12 | Consistent with MI scaling |

### Key Findings

1. **Triphasic regime structure confirmed**: Grounded (λ_max < 0), Creative (λ_max ≈ 0), Hallucinatory (λ_max > 0)
2. **Phase boundary linearity validated**: The relationship η_c ≈ m·λ + b holds with R² ≈ 0.82
3. **Hysteresis present**: Gap of ~5.3 demonstrates first-order transition character, though smaller than original (~11.52), likely due to different initial conditions and no explicit carryover mechanism

### How to Run

```bash
cd hallucination_research/contrib
python grok_su2_numpy_replica.py
```

**Expected output**: Grid sweep over (η, λ) parameter space, printing Ī, norm, λ_max, and regime classification for each point.

### Comparison to Main Implementation

| Feature | Grok Replica | Main (phase_dynamics.py) |
|---------|--------------|--------------------------|
| Integration | Heun (2nd order) | Heun (2nd order) |
| MI Estimation | Gaussian from cov | Gaussian from cov |
| Stabilizers | Shrinkage + jitter | Shrinkage + jitter |
| Dependencies | NumPy only | NumPy + project utils |
| Code size | ~150 lines | ~200+ lines (with exports) |

### Research Value

This independent replication:
- Validates the mathematical formulation is reproducible from the paper description
- Confirms key empirical signatures (boundary linearity, hysteresis, regime transitions)
- Provides a lightweight reference implementation for newcomers
- Demonstrates the framework's robustness to implementation details

### Citation

If using this replica, please cite both:
1. The main Geometric Theory of AI Hallucination paper
2. xAI Grok as independent replicator (contribution via conversation, 2025)

### Notes

- Small parameter differences (WINDOW=30 vs potential variations, EMA_ALPHA, etc.) may cause minor metric shifts
- The +0.12 offset is within expected variance for MI surrogates
- No carryover state between forward/backward sweeps in this version, reducing hysteresis gap vs original
