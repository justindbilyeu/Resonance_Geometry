"""
High-level functional API for computing F_AI.

Orchestrates all components to compute F_AI scores for episodes and batches.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .types import Episode, EpisodeMetrics, CalibrationData
from .embeddings import EpisodeEmbedder, embed_episode_roles
from .transport import (
    compute_all_transports,
    compute_all_holonomies,
    compute_average_holonomy,
)
from .metrics import (
    compute_coupling,
    compute_goldilocks_disagreement,
    compute_persistence,
    compute_complexity,
    create_calibration_data,
    normalize_metrics,
    compute_f_ai_score,
)


def compute_episode_raw_metrics(
    episode: Episode,
    embedder: EpisodeEmbedder,
    topology_config: Dict,
    cycles: List[Dict],
    previous_introspection_embedding: Optional[np.ndarray] = None,
    D0: float = 0.25,
    lambda_edge: float = 1.0,
    m_max: int = 100,
    epsilon: float = 1e-8,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute raw metrics (C, U, D, P, N) for an episode.

    Args:
        episode: Episode object
        embedder: EpisodeEmbedder instance
        topology_config: Topology configuration
        cycles: List of cycle definitions
        previous_introspection_embedding: Previous episode's I embedding (for P)
        D0: Target disagreement
        lambda_edge: Edge penalty coefficient
        m_max: Max sentences for Procrustes
        epsilon: Numerical stability

    Returns:
        Tuple of:
            - Dict with raw metric values (C, U, D, P, N)
            - Current introspection embedding (for next episode)
    """
    # Get roles and edges from topology
    roles = topology_config["roles"]
    edges = [tuple(e) for e in topology_config["edges"]]

    # Embed all roles
    role_embeddings = embed_episode_roles(episode, embedder, roles)

    # 1. Coupling (C)
    C = compute_coupling(role_embeddings, edges)

    # 2. Holonomy uncertainty (U)
    transports = compute_all_transports(
        role_embeddings, edges, m_max=m_max, epsilon=epsilon
    )
    holonomies = compute_all_holonomies(transports, cycles, embedder.get_dimension())
    U = compute_average_holonomy(holonomies)

    # 3. Goldilocks disagreement (D)
    # Assumes B and S roles exist (check first)
    if episode.has_role("B") and episode.has_role("S"):
        D = compute_goldilocks_disagreement(
            episode, belief_role="B", state_role="S", D0=D0
        )
    else:
        D = 0.0

    # 4. Persistence (P)
    # Get current introspection embedding
    current_I_embedding = None
    if episode.has_role("I") and "I" in role_embeddings:
        current_I_embedding = role_embeddings["I"]["episode_embedding"]
        P = compute_persistence(current_I_embedding, previous_introspection_embedding)
    else:
        P = 0.0

    # 5. Complexity (N)
    total_tokens = episode.total_tokens()
    n_edges = len(edges)
    N = compute_complexity(total_tokens, n_edges, lambda_edge=lambda_edge)

    metrics = {"C": C, "U": U, "D": D, "P": P, "N": N}

    return metrics, current_I_embedding


def compute_f_ai_episode(
    episode: Episode,
    embedder: EpisodeEmbedder,
    topology_config: Dict,
    cycles: List[Dict],
    calibration: CalibrationData,
    previous_introspection_embedding: Optional[np.ndarray] = None,
    coefficients: Optional[Dict[str, float]] = None,
) -> Tuple[EpisodeMetrics, np.ndarray]:
    """
    Compute full F_AI metrics for an episode.

    Args:
        episode: Episode object
        embedder: EpisodeEmbedder
        topology_config: Topology configuration
        cycles: Cycle definitions
        calibration: CalibrationData for normalization
        previous_introspection_embedding: Previous I embedding
        coefficients: Coefficient overrides (optional)

    Returns:
        Tuple of:
            - EpisodeMetrics object
            - Current introspection embedding
    """
    # Default coefficients
    default_coeffs = {
        "alpha": 1.0,
        "beta": 1.0,
        "lambda": 1.0,
        "mu": 0.5,
        "nu": 0.2,
        "D0": 0.25,
        "lambda_edge": 1.0,
        "m_max": 100,
        "epsilon": 1e-8,
    }

    if coefficients:
        default_coeffs.update(coefficients)

    # Compute raw metrics
    raw_metrics, current_I_emb = compute_episode_raw_metrics(
        episode,
        embedder,
        topology_config,
        cycles,
        previous_introspection_embedding=previous_introspection_embedding,
        D0=default_coeffs["D0"],
        lambda_edge=default_coeffs["lambda_edge"],
        m_max=int(default_coeffs["m_max"]),
        epsilon=default_coeffs["epsilon"],
    )

    # Normalize metrics
    normalized = normalize_metrics(raw_metrics, calibration)

    # Compute F_AI score
    F_AI = compute_f_ai_score(
        normalized,
        alpha=default_coeffs["alpha"],
        beta=default_coeffs["beta"],
        lambda_=default_coeffs["lambda"],
        mu=default_coeffs["mu"],
        nu=default_coeffs["nu"],
    )

    # Create EpisodeMetrics object
    metrics_obj = EpisodeMetrics(
        session_id=episode.session_id,
        episode_id=episode.episode_id,
        topology_id=episode.topology_id,
        C=raw_metrics["C"],
        U=raw_metrics["U"],
        D=raw_metrics["D"],
        P=raw_metrics["P"],
        N=raw_metrics["N"],
        C_z=normalized["C_z"],
        U_z=normalized["U_z"],
        D_z=normalized["D_z"],
        P_z=normalized["P_z"],
        N_z=normalized["N_z"],
        F_AI=F_AI,
        n_messages=len(episode.messages),
        n_tokens=episode.total_tokens(),
        roles_present=episode.get_roles(),
    )

    return metrics_obj, current_I_emb


def compute_f_ai_batch(
    episodes: List[Episode],
    embedder: EpisodeEmbedder,
    topology_configs: Dict[str, Dict],
    cycles_by_topology: Dict[str, List[Dict]],
    calibration_by_topology: Dict[str, CalibrationData],
    coefficients: Optional[Dict[str, float]] = None,
) -> List[EpisodeMetrics]:
    """
    Compute F_AI for a batch of episodes.

    Handles multiple sessions and topologies. Tracks persistence within sessions.

    Args:
        episodes: List of Episode objects
        embedder: EpisodeEmbedder
        topology_configs: Dict mapping topology_id to config
        cycles_by_topology: Dict mapping topology_id to cycles
        calibration_by_topology: Dict mapping topology_id to CalibrationData
        coefficients: Optional coefficient overrides

    Returns:
        List of EpisodeMetrics objects
    """
    from .segment import group_episodes_by_session

    # Group by session for persistence tracking
    sessions = group_episodes_by_session(episodes)

    all_metrics = []

    for session_id, session_episodes in sessions.items():
        # Track previous I embedding within session
        prev_I_emb = None

        for ep in session_episodes:
            # Get topology-specific configs
            topology_id = ep.topology_id

            if topology_id not in topology_configs:
                raise ValueError(f"Unknown topology: {topology_id}")

            topology_config = topology_configs[topology_id]
            cycles = cycles_by_topology.get(topology_id, [])
            calibration = calibration_by_topology.get(topology_id)

            if calibration is None:
                raise ValueError(
                    f"No calibration data for topology: {topology_id}"
                )

            # Compute metrics
            metrics, current_I_emb = compute_f_ai_episode(
                ep,
                embedder,
                topology_config,
                cycles,
                calibration,
                previous_introspection_embedding=prev_I_emb,
                coefficients=coefficients,
            )

            all_metrics.append(metrics)

            # Update previous I embedding for next episode
            prev_I_emb = current_I_emb

    return all_metrics


def compute_calibration_from_episodes(
    calibration_episodes: List[Episode],
    embedder: EpisodeEmbedder,
    topology_configs: Dict[str, Dict],
    cycles_by_topology: Dict[str, List[Dict]],
    coefficients: Optional[Dict[str, float]] = None,
) -> Dict[str, CalibrationData]:
    """
    Compute calibration data from calibration episodes.

    Args:
        calibration_episodes: List of calibration Episode objects
        embedder: EpisodeEmbedder
        topology_configs: Dict mapping topology_id to config
        cycles_by_topology: Dict mapping topology_id to cycles
        coefficients: Optional coefficient overrides

    Returns:
        Dict mapping topology_id to CalibrationData
    """
    from .segment import group_episodes_by_topology, group_episodes_by_session

    # Group by topology
    by_topology = group_episodes_by_topology(calibration_episodes)

    calibration_by_topology = {}

    for topology_id, topo_episodes in by_topology.items():
        # Get configs
        topology_config = topology_configs[topology_id]
        cycles = cycles_by_topology.get(topology_id, [])

        # Group by session for persistence
        sessions = group_episodes_by_session(topo_episodes)

        # Compute raw metrics for all episodes
        raw_metrics_list = []

        for session_id, session_episodes in sessions.items():
            prev_I_emb = None

            for ep in session_episodes:
                raw_metrics, current_I_emb = compute_episode_raw_metrics(
                    ep,
                    embedder,
                    topology_config,
                    cycles,
                    previous_introspection_embedding=prev_I_emb,
                    D0=coefficients.get("D0", 0.25) if coefficients else 0.25,
                    lambda_edge=(
                        coefficients.get("lambda_edge", 1.0)
                        if coefficients
                        else 1.0
                    ),
                    m_max=int(coefficients.get("m_max", 100)) if coefficients else 100,
                    epsilon=(
                        coefficients.get("epsilon", 1e-8) if coefficients else 1e-8
                    ),
                )

                raw_metrics_list.append(raw_metrics)
                prev_I_emb = current_I_emb

        # Create calibration data
        calibration = create_calibration_data(topology_id, raw_metrics_list)
        calibration_by_topology[topology_id] = calibration

    return calibration_by_topology
