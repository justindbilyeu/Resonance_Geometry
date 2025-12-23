"""
Episode segmentation and grouping utilities.
"""

from typing import List, Dict, Any
from collections import defaultdict
from .types import Episode, Message
import re


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Simple rule-based sentence splitter. For production, could use
    more sophisticated methods (e.g., spaCy, nltk).

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple regex-based sentence splitting
    # Handles periods, question marks, exclamation marks
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # If no sentences found, return the whole text
    if not sentences:
        return [text] if text.strip() else []

    return sentences


def get_role_sentences(episode: Episode, role: str) -> List[str]:
    """
    Extract sentences for a specific role in an episode.

    Args:
        episode: Episode object
        role: Role identifier

    Returns:
        List of sentences for that role (Fix #1: sentence-level)
    """
    messages = episode.get_messages_by_role(role)
    sentences = []

    for msg in messages:
        msg_sentences = split_into_sentences(msg.text)
        sentences.extend(msg_sentences)

    return sentences


def get_role_sentence_embeddings_indices(
    episode: Episode, role: str
) -> Dict[str, Any]:
    """
    Get metadata about sentence indices for a role.

    Args:
        episode: Episode object
        role: Role identifier

    Returns:
        Dict with sentence count and message mapping

    Note: Fix #1 - Clearly track sentence-level vs episode-level indices
    """
    messages = episode.get_messages_by_role(role)
    sentence_idx = 0
    mapping = []

    for msg_idx, msg in enumerate(messages):
        msg_sentences = split_into_sentences(msg.text)
        for sent in msg_sentences:
            mapping.append(
                {
                    "sentence_idx": sentence_idx,  # Fix #1: clear naming
                    "message_idx": msg_idx,
                    "sentence_text": sent,
                }
            )
            sentence_idx += 1

    return {"n_sentences": sentence_idx, "mapping": mapping}


def group_episodes_by_session(episodes: List[Episode]) -> Dict[str, List[Episode]]:
    """
    Group episodes by session_id.

    Args:
        episodes: List of Episode objects

    Returns:
        Dict mapping session_id to list of episodes (sorted by episode_id)
    """
    sessions = defaultdict(list)

    for ep in episodes:
        sessions[ep.session_id].append(ep)

    # Sort each session by episode_id
    for session_id in sessions:
        sessions[session_id].sort(key=lambda ep: ep.episode_id)

    return dict(sessions)


def group_episodes_by_topology(
    episodes: List[Episode],
) -> Dict[str, List[Episode]]:
    """
    Group episodes by topology_id.

    Args:
        episodes: List of Episode objects

    Returns:
        Dict mapping topology_id to list of episodes
    """
    topologies = defaultdict(list)

    for ep in episodes:
        topologies[ep.topology_id].append(ep)

    return dict(topologies)


def get_calibration_split(
    episodes: List[Episode],
    method: str = "first_n",
    n: int = 50,
    calibration_file: str = None,
) -> List[Episode]:
    """
    Get calibration split for normalization.

    Args:
        episodes: All episodes
        method: "first_n" or "from_file"
        n: Number of episodes per topology (for first_n)
        calibration_file: Path to file with calibration episode IDs

    Returns:
        Calibration episodes
    """
    if method == "first_n":
        # Group by topology
        by_topology = group_episodes_by_topology(episodes)
        calibration = []

        for topology_id, topo_episodes in by_topology.items():
            # Sort by session and episode ID
            topo_episodes.sort(key=lambda ep: (ep.session_id, ep.episode_id))
            # Take first n
            calibration.extend(topo_episodes[:n])

        return calibration

    elif method == "from_file":
        if not calibration_file:
            raise ValueError("calibration_file required for from_file method")

        # Load calibration IDs from file
        # Expected format: one line per episode "session_id:episode_id"
        calibration_ids = set()
        with open(calibration_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    calibration_ids.add(line)

        # Filter episodes
        calibration = [
            ep
            for ep in episodes
            if f"{ep.session_id}:{ep.episode_id}" in calibration_ids
        ]

        return calibration

    else:
        raise ValueError(f"Unknown calibration method: {method}")


def get_evaluation_split(
    episodes: List[Episode], calibration_episodes: List[Episode]
) -> List[Episode]:
    """
    Get evaluation split (episodes not in calibration).

    Args:
        episodes: All episodes
        calibration_episodes: Calibration episodes

    Returns:
        Evaluation episodes
    """
    calibration_ids = set(
        (ep.session_id, ep.episode_id) for ep in calibration_episodes
    )

    evaluation = [
        ep
        for ep in episodes
        if (ep.session_id, ep.episode_id) not in calibration_ids
    ]

    return evaluation
