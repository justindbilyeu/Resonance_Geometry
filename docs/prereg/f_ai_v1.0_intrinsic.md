# F_AI Intrinsic v1.0 - Preregistration Specification

**Version**: 1.0
**Date**: 2025-12-23
**Status**: Locked

## Overview

This document specifies the intrinsic F_AI metric for evaluating AI system dynamics through geometric and information-theoretic lenses. The metric is designed to be computed from conversational episodes without external ground truth.

## Core Components

### 1. Coupling (C)

Measures semantic alignment between roles within an episode using cosine similarity of episode-level embeddings.

```
C(e) = (1/|E|) ∑_{(r,q)∈E} cos(v_r^[e], v_q^[e])
```

Where:
- `E` is the set of edges in the topology graph
- `v_r^[e]` is the episode-level embedding for role `r` (mean of sentence embeddings)
- Higher values indicate better semantic coherence

### 2. Holonomy Uncertainty (U)

Quantifies geometric inconsistency via loop holonomy using Procrustes O(d) transport.

```
U(e) = (1/|L|) ∑_{ℓ∈L} H_loop(ℓ)

H_loop(ℓ) = 1 - (1/d) Tr(U_ℓ)

U_ℓ = ∏_{(r→q)∈ℓ} R_{r→q}
```

Where:
- `R_{r→q}` is the Procrustes rotation matrix from role r to role q
- `L` is the set of preregistered cycles
- Normalized by dimension `d` (fix #3)
- Higher values indicate greater geometric inconsistency

### 3. Goldilocks Disagreement (D)

Measures deviation from optimal disagreement level between belief-state roles.

```
D(e) = (JSD(p_B || p_S) - D₀)²
```

Where:
- `p_B`, `p_S` are feature distributions from CountVectorizer
- `D₀ = 0.25` (target disagreement, preregistered)
- Features: ngram_range=(1,2), lowercase=True, normalized to probabilities
- Lower values indicate appropriate disagreement level (fix #4)

### 4. Persistence (P)

Tracks consistency of introspective state across episodes in a session.

```
P(e) = cos(v_I^[e], v_I^[e-1])
```

Where:
- `v_I^[e]` is the introspective role embedding
- For first episode in session: P(e) = 0
- Higher values indicate more stable introspection

### 5. Complexity Penalty (N)

Accounts for both token count and topological complexity.

```
N(e) = log(1 + TotalTokens(e)) + λ_edge |E|
```

Where:
- `TotalTokens(e)` is the sum of tokens across all messages
- `|E|` is the number of edges in the topology graph (fix #5)
- `λ_edge = 1.0` (preregistered coefficient)
- Higher values indicate more complex episodes

## Functional Form

```
F_AI^intrinsic(e) = α·C̃(e) - β·Ũ(e) - λ·D̃(e) + μ·P̃(e) - ν·Ñ(e)
```

### Robust Z-Score Normalization

Each component X is normalized using robust statistics from calibration split:

```
X̃ = (X - median(X_cal)) / (IQR(X_cal) + ε)
```

Where:
- `X_cal` is the calibration dataset (e.g., first 50 episodes per topology)
- `IQR` = Q₃ - Q₁ (interquartile range)
- `ε = 1e-8` (numerical stability)

### Locked Coefficients (v1.0)

- α (coupling weight) = 1.0
- β (holonomy weight) = 1.0
- λ (disagreement weight) = 1.0
- μ (persistence weight) = 0.5
- ν (complexity weight) = 0.2

## Implementation Requirements

### Six Critical Fixes

1. **Index Disambiguation** (fix #1): Clearly distinguish sentence-level indices (s) from episode-level indices (e)
2. **Shape Matching for Procrustes** (fix #2): Use deterministic sampling to match matrix dimensions
3. **Holonomy Normalization** (fix #3): Divide trace by dimension d
4. **Goldilocks Term** (fix #4): Include disagreement penalty in functional
5. **Topology Penalty** (fix #5): Include |E| term in complexity
6. **Correlation Reporting** (fix #6): Report correlation + confidence intervals, avoid hard thresholds

### Embedding Backend

**Default**: TF-IDF (no heavy dependencies)
- Sentence-level: TfidfVectorizer(max_features=384, ngram_range=(1,2))
- Fast, deterministic, CI-friendly

**Optional**: sentence-transformers
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Requires torch, only enabled if installed
- Better semantic quality

### Transport Details

Procrustes alignment between roles r → q:

1. Build matrices V̂_r, V̂_q ∈ ℝ^(m×d) from sentence embeddings
2. Choose m = min(n_r, n_q, m_max) with deterministic sampling (fix #2)
   - Either: first m sentences
   - Or: seeded uniform sampling for reproducibility
3. Center rows: V̂ ← V̂ - mean(V̂, axis=0)
4. Normalize: V̂ ← V̂ / ||V̂||_F

Solve:
```
R_{r→q} = argmin_{R∈O(d)} ||V̂_q - V̂_r R||_F²
```

Transport residual:
```
T(r→q) = ||V̂_q - V̂_r R||_F² / (||V̂_q||_F² + ε)
```

### Calibration

Use calibration split for robust normalization:
- **Option A**: First N episodes per topology (default N=50)
- **Option B**: Explicit calibration file

Store normalization parameters:
- Median, IQR for each component (C, U, D, P, N)
- Per-topology if topologies differ significantly

## Output Format

### Per-Episode CSV

Columns: `session_id, episode_id, topology_id, F_AI, C, U, D, P, N, C_z, U_z, D_z, P_z, N_z`

### Per-Session Summary

Columns: `session_id, topology_id, n_episodes, F_AI_mean, F_AI_median, C_mean, U_mean, D_mean, P_mean, N_mean`

### Per-Topology Comparison

Columns: `topology_id, n_episodes, n_sessions, F_AI_mean, F_AI_std, [component means]`

### Normalization Parameters

JSON file with:
```json
{
  "calibration_method": "first_n",
  "calibration_n": 50,
  "per_topology": {
    "role_split_BSIM": {
      "C": {"median": 0.65, "iqr": 0.12},
      "U": {"median": 0.08, "iqr": 0.05},
      ...
    }
  }
}
```

## Validation Criteria

1. **Unit tests pass**: All component functions tested
2. **Procrustes shape handling**: Mismatched sentence counts handled deterministically
3. **Holonomy normalization**: H_loop values in [0, 2] as expected
4. **Goldilocks**: D penalty included in F_AI calculation
5. **Topology penalty**: |E| term present in complexity
6. **Reproducibility**: Same inputs → same outputs (fix random seeds)
7. **CI compliance**: Runs with TF-IDF backend, no torch required

## References

- Orthogonal Procrustes: Schönemann, P. H. (1966)
- Jensen-Shannon Divergence: Lin, J. (1991)
- Robust statistics: Rousseeuw & Croux (1993)

---

**Lock Status**: This specification is locked for v1.0. Any changes require a new version.
