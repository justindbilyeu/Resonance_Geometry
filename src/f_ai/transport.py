"""
Procrustes transport and holonomy computation.

Fix #2: Deterministic shape matching for Procrustes alignment
Fix #3: Holonomy normalized by dimension d
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def match_shapes_deterministic(
    V_r: np.ndarray,
    V_q: np.ndarray,
    m_max: int = 100,
    method: str = "first",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match shapes of two embedding matrices deterministically.

    Fix #2: Deterministic sampling to ensure same dimensions for Procrustes.

    Args:
        V_r: Sentence embeddings for role r, shape (n_r, d)
        V_q: Sentence embeddings for role q, shape (n_q, d)
        m_max: Maximum number of sentences to use
        method: "first" (take first m) or "seeded" (seeded random sampling)
        seed: Random seed for reproducibility (if method="seeded")

    Returns:
        V_r_matched: (m, d) array
        V_q_matched: (m, d) array
    """
    n_r, d_r = V_r.shape
    n_q, d_q = V_q.shape

    if d_r != d_q:
        raise ValueError(f"Embedding dimensions must match: {d_r} != {d_q}")

    # Determine m
    m = min(n_r, n_q, m_max)

    if m == 0:
        # Return empty arrays
        return np.zeros((0, d_r)), np.zeros((0, d_q))

    if method == "first":
        # Simple: take first m sentences (deterministic)
        return V_r[:m], V_q[:m]

    elif method == "seeded":
        # Seeded random sampling (deterministic with fixed seed)
        rng = np.random.RandomState(seed)

        if n_r >= m:
            idx_r = rng.choice(n_r, size=m, replace=False)
        else:
            idx_r = np.arange(n_r)

        if n_q >= m:
            idx_q = rng.choice(n_q, size=m, replace=False)
        else:
            idx_q = np.arange(n_q)

        return V_r[idx_r], V_q[idx_q]

    else:
        raise ValueError(f"Unknown method: {method}")


def procrustes_alignment(
    V_r: np.ndarray, V_q: np.ndarray, epsilon: float = 1e-8
) -> Tuple[np.ndarray, float]:
    """
    Compute Procrustes alignment between two point clouds.

    Solves: R = argmin_{R ∈ O(d)} ||V_q - V_r R||_F^2

    Args:
        V_r: Source embeddings, shape (m, d)
        V_q: Target embeddings, shape (m, d)
        epsilon: Numerical stability constant

    Returns:
        R: Rotation matrix, shape (d, d)
        residual: Transport residual (normalized Frobenius error)
    """
    if V_r.shape != V_q.shape:
        raise ValueError(
            f"Shape mismatch: V_r {V_r.shape} != V_q {V_q.shape}. "
            "Use match_shapes_deterministic first."
        )

    if V_r.shape[0] == 0:
        # Empty matrices
        d = V_r.shape[1] if len(V_r.shape) > 1 else 1
        return np.eye(d), 0.0

    # Center
    V_r_centered = V_r - V_r.mean(axis=0, keepdims=True)
    V_q_centered = V_q - V_q.mean(axis=0, keepdims=True)

    # Normalize Frobenius norm
    norm_r = np.linalg.norm(V_r_centered, ord="fro")
    norm_q = np.linalg.norm(V_q_centered, ord="fro")

    if norm_r > epsilon:
        V_r_centered = V_r_centered / norm_r
    if norm_q > epsilon:
        V_q_centered = V_q_centered / norm_q

    # Compute cross-covariance matrix
    M = V_r_centered.T @ V_q_centered  # (d, d)

    # SVD
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Rotation matrix
    R = U @ Vt

    # Ensure proper rotation (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        # Flip last singular vector
        U[:, -1] *= -1
        R = U @ Vt

    # Compute transport residual
    V_q_transported = V_r_centered @ R
    error = np.linalg.norm(V_q_centered - V_q_transported, ord="fro") ** 2
    normalizer = np.linalg.norm(V_q_centered, ord="fro") ** 2 + epsilon
    residual = error / normalizer

    return R, float(residual)


def compute_edge_transport(
    V_r: np.ndarray,
    V_q: np.ndarray,
    m_max: int = 100,
    epsilon: float = 1e-8,
) -> Dict[str, any]:
    """
    Compute transport for an edge r → q.

    Fix #2: Includes deterministic shape matching.

    Args:
        V_r: Sentence embeddings for role r, shape (n_r, d)
        V_q: Sentence embeddings for role q, shape (n_q, d)
        m_max: Maximum sentences for alignment
        epsilon: Numerical stability

    Returns:
        Dict with:
            - "R": rotation matrix (d, d)
            - "residual": transport residual
            - "m": number of sentences used
    """
    # Match shapes (Fix #2)
    V_r_matched, V_q_matched = match_shapes_deterministic(
        V_r, V_q, m_max=m_max, method="first"
    )

    # Procrustes alignment
    R, residual = procrustes_alignment(V_r_matched, V_q_matched, epsilon=epsilon)

    return {
        "R": R,
        "residual": residual,
        "m": V_r_matched.shape[0],
    }


def compute_loop_holonomy(
    rotations: Dict[Tuple[str, str], np.ndarray],
    cycle: List[Tuple[str, str]],
    dimension: int,
) -> float:
    """
    Compute holonomy for a cycle.

    Fix #3: Normalized by dimension d.

    Args:
        rotations: Dict mapping edge (r, q) to rotation matrix R
        cycle: List of edges [(r1, r2), (r2, r3), ..., (rk, r1)]
        dimension: Embedding dimension d

    Returns:
        H_loop: Holonomy value in [0, 2]
            - 0: zero holonomy (identity)
            - 1: maximal mixing (trace = 0)
            - 2: full reversal (trace = -d)
    """
    # Initialize as identity
    U = np.eye(dimension)

    # Compose rotations
    for edge in cycle:
        if edge not in rotations:
            raise ValueError(f"Missing rotation for edge {edge}")

        R = rotations[edge]
        U = R @ U  # Right-multiply (or U @ R for left-multiply, consistent choice)

    # Compute trace
    trace = np.trace(U)

    # Normalized holonomy (Fix #3: divide by d)
    H_loop = 1.0 - trace / dimension

    return float(H_loop)


def compute_all_transports(
    role_embeddings: Dict[str, Dict[str, np.ndarray]],
    edges: List[Tuple[str, str]],
    m_max: int = 100,
    epsilon: float = 1e-8,
) -> Dict[Tuple[str, str], Dict[str, any]]:
    """
    Compute transports for all edges in topology.

    Args:
        role_embeddings: Dict mapping role to embedding data
            Each value is a dict with "sentence_embeddings" key
        edges: List of edges [(r, q), ...]
        m_max: Maximum sentences for alignment
        epsilon: Numerical stability

    Returns:
        Dict mapping edge to transport data (R, residual, m)
    """
    transports = {}

    for edge in edges:
        r, q = edge

        if r not in role_embeddings:
            raise ValueError(f"Missing embeddings for role {r}")
        if q not in role_embeddings:
            raise ValueError(f"Missing embeddings for role {q}")

        V_r = role_embeddings[r]["sentence_embeddings"]
        V_q = role_embeddings[q]["sentence_embeddings"]

        transport = compute_edge_transport(V_r, V_q, m_max=m_max, epsilon=epsilon)
        transports[edge] = transport

    return transports


def compute_all_holonomies(
    transports: Dict[Tuple[str, str], Dict[str, any]],
    cycles: List[Dict[str, any]],
    dimension: int,
) -> Dict[str, float]:
    """
    Compute holonomy for all cycles.

    Args:
        transports: Dict mapping edge to transport data (with "R" key)
        cycles: List of cycle definitions (each has "name" and "path")
        dimension: Embedding dimension

    Returns:
        Dict mapping cycle name to holonomy value
    """
    holonomies = {}

    # Extract rotation matrices
    rotations = {edge: data["R"] for edge, data in transports.items()}

    for cycle_def in cycles:
        name = cycle_def["name"]
        path = [tuple(edge) for edge in cycle_def["path"]]

        H = compute_loop_holonomy(rotations, path, dimension)
        holonomies[name] = H

    return holonomies


def compute_average_holonomy(holonomies: Dict[str, float]) -> float:
    """
    Compute average holonomy across all cycles.

    Args:
        holonomies: Dict mapping cycle name to holonomy value

    Returns:
        Average holonomy (U metric)
    """
    if not holonomies:
        return 0.0

    return float(np.mean(list(holonomies.values())))
