"""
Metric computation for F_AI components.

Implements:
- C: Coupling
- U: Holonomy uncertainty
- D: Goldilocks disagreement (Fix #4)
- P: Persistence
- N: Complexity (Fix #5: includes topology penalty)
- Robust z-score normalization
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .types import NormalizationParams, CalibrationData


def compute_coupling(
    role_embeddings: Dict[str, Dict[str, np.ndarray]],
    edges: List[Tuple[str, str]],
) -> float:
    """
    Compute coupling C: average cosine similarity across edges.

    C(e) = (1/|E|) ∑_{(r,q)∈E} cos(v_r^[e], v_q^[e])

    Args:
        role_embeddings: Dict mapping role to embedding data
            Each value has "episode_embedding" key
        edges: List of edges [(r, q), ...]

    Returns:
        Coupling value
    """
    if not edges:
        return 0.0

    similarities = []

    for r, q in edges:
        if r not in role_embeddings or q not in role_embeddings:
            continue

        v_r = role_embeddings[r]["episode_embedding"]
        v_q = role_embeddings[q]["episode_embedding"]

        # Cosine similarity
        norm_r = np.linalg.norm(v_r)
        norm_q = np.linalg.norm(v_q)

        if norm_r > 0 and norm_q > 0:
            cos_sim = np.dot(v_r, v_q) / (norm_r * norm_q)
            similarities.append(cos_sim)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def compute_goldilocks_disagreement(
    episode,
    belief_role: str = "B",
    state_role: str = "S",
    D0: float = 0.25,
    ngram_range: tuple = (1, 2),
    lowercase: bool = True,
) -> float:
    """
    Compute Goldilocks disagreement D.

    Fix #4: Included in F_AI functional.

    D(e) = (JSD(p_B || p_S) - D₀)²

    Args:
        episode: Episode object
        belief_role: Belief role identifier
        state_role: State role identifier
        D0: Target disagreement level
        ngram_range: N-gram range for features
        lowercase: Whether to lowercase

    Returns:
        Goldilocks disagreement value
    """
    from .features import compute_role_feature_distributions, compute_jsd

    # Get feature distributions
    distributions = compute_role_feature_distributions(
        episode, [belief_role, state_role], ngram_range=ngram_range, lowercase=lowercase
    )

    p_B = distributions.get(belief_role, np.array([]))
    p_S = distributions.get(state_role, np.array([]))

    # Compute JSD
    jsd = compute_jsd(p_B, p_S)

    # Goldilocks penalty
    disagreement = (jsd - D0) ** 2

    return float(disagreement)


def compute_persistence(
    current_episode_embedding: np.ndarray,
    previous_episode_embedding: Optional[np.ndarray],
) -> float:
    """
    Compute persistence P: cosine similarity between consecutive introspection states.

    P(e) = cos(v_I^[e], v_I^[e-1])

    Args:
        current_episode_embedding: Introspection embedding for current episode
        previous_episode_embedding: Introspection embedding for previous episode
            (None for first episode in session)

    Returns:
        Persistence value (0 if no previous episode)
    """
    if previous_episode_embedding is None:
        return 0.0

    # Cosine similarity
    norm_curr = np.linalg.norm(current_episode_embedding)
    norm_prev = np.linalg.norm(previous_episode_embedding)

    if norm_curr == 0 or norm_prev == 0:
        return 0.0

    cos_sim = np.dot(current_episode_embedding, previous_episode_embedding) / (
        norm_curr * norm_prev
    )

    return float(cos_sim)


def compute_complexity(
    total_tokens: int,
    n_edges: int,
    lambda_edge: float = 1.0,
) -> float:
    """
    Compute complexity N: token count + topology penalty.

    Fix #5: Includes topology penalty |E|.

    N(e) = log(1 + TotalTokens(e)) + λ_edge |E|

    Args:
        total_tokens: Total token count in episode
        n_edges: Number of edges in topology graph
        lambda_edge: Coefficient for edge penalty

    Returns:
        Complexity value
    """
    token_complexity = np.log(1 + total_tokens)
    topology_complexity = lambda_edge * n_edges

    return float(token_complexity + topology_complexity)


def compute_robust_normalization_params(
    values: np.ndarray, epsilon: float = 1e-8
) -> NormalizationParams:
    """
    Compute robust normalization parameters (median, IQR).

    Args:
        values: Array of values from calibration split
        epsilon: Numerical stability constant

    Returns:
        NormalizationParams object
    """
    if len(values) == 0:
        return NormalizationParams(median=0.0, iqr=1.0, epsilon=epsilon)

    median = float(np.median(values))

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1

    return NormalizationParams(median=median, iqr=iqr, epsilon=epsilon)


def calibrate_normalization(
    calibration_metrics: List[Dict[str, float]],
    epsilon: float = 1e-8,
) -> Dict[str, NormalizationParams]:
    """
    Compute normalization parameters from calibration metrics.

    Args:
        calibration_metrics: List of metric dicts, each with C, U, D, P, N keys
        epsilon: Numerical stability constant

    Returns:
        Dict mapping component name to NormalizationParams
    """
    # Extract arrays for each component
    components = ["C", "U", "D", "P", "N"]
    params = {}

    for comp in components:
        values = np.array([m[comp] for m in calibration_metrics if comp in m])
        params[comp] = compute_robust_normalization_params(values, epsilon=epsilon)

    return params


def create_calibration_data(
    topology_id: str,
    calibration_metrics: List[Dict[str, float]],
    epsilon: float = 1e-8,
) -> CalibrationData:
    """
    Create CalibrationData object from calibration metrics.

    Args:
        topology_id: Topology identifier
        calibration_metrics: List of metric dicts
        epsilon: Numerical stability

    Returns:
        CalibrationData object
    """
    params = calibrate_normalization(calibration_metrics, epsilon=epsilon)

    return CalibrationData(
        topology_id=topology_id,
        C_params=params["C"],
        U_params=params["U"],
        D_params=params["D"],
        P_params=params["P"],
        N_params=params["N"],
        n_calibration_episodes=len(calibration_metrics),
    )


def normalize_metrics(
    raw_metrics: Dict[str, float],
    calibration: CalibrationData,
) -> Dict[str, float]:
    """
    Normalize metrics using calibration parameters.

    Args:
        raw_metrics: Dict with C, U, D, P, N values
        calibration: CalibrationData with normalization params

    Returns:
        Dict with C_z, U_z, D_z, P_z, N_z (normalized values)
    """
    normalized = {}

    mapping = {
        "C": calibration.C_params,
        "U": calibration.U_params,
        "D": calibration.D_params,
        "P": calibration.P_params,
        "N": calibration.N_params,
    }

    for comp, params in mapping.items():
        if comp in raw_metrics:
            normalized[f"{comp}_z"] = params.normalize(raw_metrics[comp])
        else:
            normalized[f"{comp}_z"] = 0.0

    return normalized


def compute_f_ai_score(
    normalized_metrics: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 1.0,
    mu: float = 0.5,
    nu: float = 0.2,
) -> float:
    """
    Compute final F_AI score from normalized metrics.

    F_AI = α·C̃ - β·Ũ - λ·D̃ + μ·P̃ - ν·Ñ

    Args:
        normalized_metrics: Dict with C_z, U_z, D_z, P_z, N_z
        alpha: Coupling weight
        beta: Holonomy uncertainty weight
        lambda_: Goldilocks disagreement weight
        mu: Persistence weight
        nu: Complexity weight

    Returns:
        F_AI score
    """
    C_z = normalized_metrics.get("C_z", 0.0)
    U_z = normalized_metrics.get("U_z", 0.0)
    D_z = normalized_metrics.get("D_z", 0.0)
    P_z = normalized_metrics.get("P_z", 0.0)
    N_z = normalized_metrics.get("N_z", 0.0)

    F_AI = alpha * C_z - beta * U_z - lambda_ * D_z + mu * P_z - nu * N_z

    return float(F_AI)
