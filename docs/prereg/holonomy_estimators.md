# Holonomy Estimators - Technical Specification

**Version**: 1.0
**Date**: 2025-12-23
**Status**: Locked

## Overview

This document specifies the holonomy estimation procedure using Procrustes O(d) transport for the F_AI intrinsic metric. Holonomy quantifies geometric inconsistency in multi-role AI systems.

## Theoretical Background

### What is Holonomy?

In differential geometry, holonomy measures how much a vector changes when parallel-transported around a closed loop. Non-zero holonomy indicates curvature in the underlying space.

For AI systems, we use discrete holonomy to detect geometric inconsistencies between role representations:
- **Zero holonomy**: Roles are geometrically consistent
- **High holonomy**: Roles exhibit geometric tension or inconsistency

### Why Procrustes?

Procrustes alignment finds the optimal orthogonal transformation between point clouds, preserving:
- Distances (isometry)
- Angles (conformal structure)
- Orientation (proper rotations)

This makes it ideal for comparing semantic embeddings across roles.

## Mathematical Formulation

### 1. Embedding Space

Each role r has a set of sentence embeddings:
```
{v_{r,1}, v_{r,2}, ..., v_{r,n_r}} ⊂ ℝ^d
```

Where:
- `d` = embedding dimension (384 for both TF-IDF and MiniLM)
- `n_r` = number of sentences in role r

### 2. Procrustes Alignment

For edge r → q in topology graph:

**Step 1**: Sample sentences
```
m = min(n_r, n_q, m_max)
```
- Use first m sentences (deterministic)
- Or: seeded uniform sampling if shuffling desired
- Default m_max = 100 (computational efficiency)

**Step 2**: Build matrices
```
V̂_r = [v_{r,1}, ..., v_{r,m}]ᵀ ∈ ℝ^(m×d)
V̂_q = [v_{q,1}, ..., v_{q,m}]ᵀ ∈ ℝ^(m×d)
```

**Step 3**: Preprocess
```
V̂_r ← V̂_r - mean(V̂_r, axis=0)  # Center
V̂_r ← V̂_r / ||V̂_r||_F          # Normalize

V̂_q ← V̂_q - mean(V̂_q, axis=0)
V̂_q ← V̂_q / ||V̂_q||_F
```

**Step 4**: Solve Procrustes
```
R_{r→q} = argmin_{R∈O(d)} ||V̂_q - V̂_r R||_F²
```

Solution via SVD:
```
M = V̂_rᵀ V̂_q
U, Σ, Vᵀ = SVD(M)
R_{r→q} = U Vᵀ
```

**Step 5**: Compute transport residual
```
T(r→q) = ||V̂_q - V̂_r R_{r→q}||_F² / (||V̂_q||_F² + ε)
```

### 3. Loop Holonomy

For a cycle ℓ = (r₁ → r₂ → ... → r_k → r₁):

**Step 1**: Compose rotations
```
U_ℓ = R_{r_k→r₁} · ... · R_{r₂→r₃} · R_{r₁→r₂}
```

**Step 2**: Compute holonomy
```
H_loop(ℓ) = 1 - (1/d) Tr(U_ℓ)
```

Properties:
- `U_ℓ ∈ O(d)` (orthogonal matrix)
- `Tr(U_ℓ) ∈ [-d, d]`
- `H_loop ∈ [0, 2]`
  - H_loop = 0: identity (zero holonomy)
  - H_loop = 1: trace zero (maximal mixing)
  - H_loop = 2: -identity (full reversal)

**Critical Fix #3**: Divide by d to normalize across dimensions.

### 4. Aggregate Uncertainty

Average over preregistered cycles:
```
U(e) = (1/|L|) ∑_{ℓ∈L} H_loop(ℓ)
```

## Preregistered Cycles

Cycles must be specified before seeing data. Examples for role_split_BSIM topology:

### Primary Cycles

1. **Belief-State cycle**: B → S → I → B
   - Tests consistency between belief formation and state tracking

2. **Meta cycle**: B → S → M → B
   - Tests meta-cognitive consistency

3. **Full cycle**: U → B → S → I → M → U
   - Tests overall system consistency

### Secondary Cycles

4. **State-Introspection**: S → I → M → S
5. **Belief-Meta**: B → I → M → B

Cycles are defined in `configs/f_ai/loop_cycles.json`.

## Implementation Details

### Index Disambiguation (Fix #1)

Use clear variable names:
- `s`, `sentence_idx`: sentence-level indices
- `e`, `episode_idx`: episode-level indices
- `i`, `j`, `k`: loop indices

Example:
```python
# GOOD
for sentence_idx in range(n_sentences):
    v = embeddings[sentence_idx]

# BAD (ambiguous)
for i in range(n):
    v = embeddings[i]
```

### Shape Matching (Fix #2)

```python
def match_shapes(V_r, V_q, m_max=100, seed=42):
    """
    Deterministically sample to match shapes.

    Args:
        V_r: (n_r, d) array
        V_q: (n_q, d) array
        m_max: maximum samples
        seed: random seed for reproducibility

    Returns:
        V_r_matched: (m, d) array
        V_q_matched: (m, d) array
    """
    n_r, n_q = len(V_r), len(V_q)
    m = min(n_r, n_q, m_max)

    # Option A: First m (deterministic, simple)
    return V_r[:m], V_q[:m]

    # Option B: Seeded sampling (deterministic, better coverage)
    # rng = np.random.RandomState(seed)
    # idx_r = rng.choice(n_r, size=m, replace=False)
    # idx_q = rng.choice(n_q, size=m, replace=False)
    # return V_r[idx_r], V_q[idx_q]
```

### Numerical Stability

```python
def procrustes_alignment(V_r, V_q, eps=1e-8):
    """Numerically stable Procrustes alignment."""
    # Center
    V_r = V_r - V_r.mean(axis=0)
    V_q = V_q - V_q.mean(axis=0)

    # Normalize (with stability)
    norm_r = np.linalg.norm(V_r, ord='fro')
    norm_q = np.linalg.norm(V_q, ord='fro')

    if norm_r > eps:
        V_r = V_r / norm_r
    if norm_q > eps:
        V_q = V_q / norm_q

    # SVD
    M = V_r.T @ V_q
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    return R
```

### Loop Composition

```python
def compute_loop_holonomy(rotations, cycle, d):
    """
    Compute holonomy for a cycle.

    Args:
        rotations: dict mapping edge tuples to R matrices
        cycle: list of role names [(r1, r2), (r2, r3), ..., (rk, r1)]
        d: embedding dimension

    Returns:
        H_loop: scalar in [0, 2]
    """
    U = np.eye(d)
    for edge in cycle:
        if edge not in rotations:
            raise ValueError(f"Missing rotation for edge {edge}")
        U = rotations[edge] @ U  # Right-multiply

    trace = np.trace(U)
    H_loop = 1.0 - trace / d  # Fix #3: normalize by d

    return H_loop
```

## Validation Tests

### Test 1: Rotation Invariance

Applying the same rotation to all roles should not change holonomy:
```python
def test_rotation_invariance():
    # Random orthogonal rotation
    Q = scipy.stats.ortho_group.rvs(d)

    # Transform all embeddings
    V_r_rot = {role: V[role] @ Q for role in roles}

    # Holonomy should be unchanged
    H_original = compute_holonomy(V)
    H_rotated = compute_holonomy(V_rot)

    assert np.allclose(H_original, H_rotated)
```

### Test 2: Zero Holonomy for Consistent Spaces

If all roles have identical geometry, holonomy should be near zero:
```python
def test_zero_holonomy():
    # All roles have same embeddings (up to permutation)
    V_identical = {role: V_base for role in roles}

    H = compute_holonomy(V_identical)

    assert H < 0.01  # Near zero
```

### Test 3: Dimension Normalization

Holonomy should scale properly across dimensions:
```python
def test_dimension_scaling():
    for d in [10, 50, 100, 384]:
        V = generate_test_embeddings(d=d)
        H = compute_holonomy(V)
        assert 0 <= H <= 2
```

## Estimator A: Locked Configuration (v1.0)

For v1.0, we lock the following choices:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Method | Procrustes O(d) | Preserves geometry, computationally efficient |
| Sampling | First m sentences | Deterministic, simple |
| m_max | 100 | Balance coverage vs. computation |
| Normalization | Center + Frobenius | Standard preprocessing |
| Trace formula | 1 - Tr(U_ℓ)/d | Normalized to [0, 2] |
| Cycles | See configs | Preregistered, topology-specific |

Alternative estimators (future versions):
- Estimator B: Full optimal transport (Wasserstein)
- Estimator C: Riemannian metrics
- Estimator D: Information geometry (Fisher-Rao)

## References

1. Schönemann, P. H. (1966). "A generalized solution of the orthogonal Procrustes problem". Psychometrika.
2. Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
3. Ambrosio, L., & Gigli, N. (2013). "A user's guide to optimal transport". Modelling and Optimisation of Flows on Networks.

---

**Lock Status**: Estimator A locked for v1.0. Extensions require new version.
