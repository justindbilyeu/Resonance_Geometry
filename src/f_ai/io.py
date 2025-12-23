"""
I/O utilities for F_AI: JSONL reading/writing and schema validation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from .types import Episode, Message


def load_json_config(filepath: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_topology_config(filepath: str, topology_id: str) -> Dict[str, Any]:
    """
    Load specific topology configuration.

    Args:
        filepath: Path to topology_configs.json
        topology_id: Topology identifier

    Returns:
        Topology configuration dict

    Raises:
        ValueError: If topology_id not found
    """
    config = load_json_config(filepath)
    topologies = config.get("topologies", {})

    if topology_id not in topologies:
        raise ValueError(
            f"Topology '{topology_id}' not found. "
            f"Available: {list(topologies.keys())}"
        )

    return topologies[topology_id]


def load_loop_cycles(filepath: str, topology_id: str) -> List[Dict[str, Any]]:
    """
    Load loop cycles for a topology.

    Args:
        filepath: Path to loop_cycles.json
        topology_id: Topology identifier

    Returns:
        List of cycle definitions
    """
    config = load_json_config(filepath)
    cycles = config.get("cycles", {})

    if topology_id not in cycles:
        raise ValueError(
            f"No cycles defined for topology '{topology_id}'. "
            f"Available: {list(cycles.keys())}"
        )

    return cycles[topology_id]


def load_coefficients(filepath: str) -> Dict[str, float]:
    """
    Load F_AI coefficients.

    Args:
        filepath: Path to coefficients.json

    Returns:
        Dict with coefficient values
    """
    config = load_json_config(filepath)
    coeffs = config.get("coefficients", {})

    # Extract numeric values
    result = {k: v["value"] for k, v in coeffs.items()}

    # Add other parameters
    other = config.get("other_parameters", {})
    result.update({k: v["value"] for k, v in other.items()})

    return result


def read_jsonl(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Read JSONL file line by line.

    Args:
        filepath: Path to JSONL file

    Yields:
        Parsed JSON objects
    """
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")


def write_jsonl(filepath: str, data: List[Dict[str, Any]]) -> None:
    """
    Write data to JSONL file.

    Args:
        filepath: Output path
        data: List of dicts to write
    """
    with open(filepath, "w") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")


def validate_episode_dict(
    episode_dict: Dict[str, Any],
    topology_configs: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Validate episode dictionary against schema.

    Args:
        episode_dict: Episode data
        topology_configs: Optional topology configurations for validation

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Required fields
    required = ["session_id", "episode_id", "topology_id", "messages"]
    for field in required:
        if field not in episode_dict:
            raise ValueError(f"Missing required field: {field}")

    # Validate types
    if not isinstance(episode_dict["session_id"], str):
        raise ValueError("session_id must be string")
    if not isinstance(episode_dict["episode_id"], int):
        raise ValueError("episode_id must be integer")
    if episode_dict["episode_id"] < 1:
        raise ValueError("episode_id must be >= 1")
    if not isinstance(episode_dict["topology_id"], str):
        raise ValueError("topology_id must be string")

    # Validate messages
    messages = episode_dict["messages"]
    if not isinstance(messages, list):
        raise ValueError("messages must be list")
    if len(messages) == 0:
        raise ValueError("messages cannot be empty")

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {idx} must be dict")
        if "role" not in msg:
            raise ValueError(f"Message {idx} missing 'role'")
        if "text" not in msg:
            raise ValueError(f"Message {idx} missing 'text'")
        if not isinstance(msg["role"], str):
            raise ValueError(f"Message {idx} role must be string")
        if not isinstance(msg["text"], str):
            raise ValueError(f"Message {idx} text must be string")

    # Validate against topology if provided
    if topology_configs:
        topology_id = episode_dict["topology_id"]
        if topology_id not in topology_configs:
            raise ValueError(
                f"Unknown topology_id: {topology_id}. "
                f"Available: {list(topology_configs.keys())}"
            )

        topology = topology_configs[topology_id]
        required_roles = set(topology.get("required_roles", []))
        optional_roles = set(topology.get("optional_roles", []))
        allowed_roles = required_roles | optional_roles

        episode_roles = set(msg["role"] for msg in messages)

        # Check invalid roles
        invalid = episode_roles - allowed_roles
        if invalid:
            raise ValueError(
                f"Invalid roles for topology {topology_id}: {invalid}"
            )

        # Check missing required roles
        missing = required_roles - episode_roles
        if missing:
            raise ValueError(
                f"Missing required roles for topology {topology_id}: {missing}"
            )

    return True


def parse_episode(episode_dict: Dict[str, Any]) -> Episode:
    """
    Parse episode dictionary into Episode object.

    Args:
        episode_dict: Episode data

    Returns:
        Episode object
    """
    messages = [
        Message(
            role=msg["role"],
            text=msg["text"],
            timestamp=msg.get("timestamp"),
            metadata=msg.get("metadata", {}),
        )
        for msg in episode_dict["messages"]
    ]

    return Episode(
        session_id=episode_dict["session_id"],
        episode_id=episode_dict["episode_id"],
        topology_id=episode_dict["topology_id"],
        messages=messages,
        metadata=episode_dict.get("metadata", {}),
    )


def load_episodes(
    filepath: str,
    topology_config_path: Optional[str] = None,
    validate: bool = True,
) -> List[Episode]:
    """
    Load episodes from JSONL file.

    Args:
        filepath: Path to JSONL file
        topology_config_path: Optional path to topology configs for validation
        validate: Whether to validate against schema

    Returns:
        List of Episode objects
    """
    episodes = []
    topology_configs = None

    if validate and topology_config_path:
        config = load_json_config(topology_config_path)
        topology_configs = config.get("topologies", {})

    for episode_dict in read_jsonl(filepath):
        if validate:
            validate_episode_dict(episode_dict, topology_configs)
        episode = parse_episode(episode_dict)
        episodes.append(episode)

    return episodes


def save_episodes(filepath: str, episodes: List[Episode]) -> None:
    """
    Save episodes to JSONL file.

    Args:
        filepath: Output path
        episodes: List of Episode objects
    """
    data = []
    for ep in episodes:
        episode_dict = {
            "session_id": ep.session_id,
            "episode_id": ep.episode_id,
            "topology_id": ep.topology_id,
            "messages": [
                {
                    "role": msg.role,
                    "text": msg.text,
                    **({"timestamp": msg.timestamp} if msg.timestamp else {}),
                    **({"metadata": msg.metadata} if msg.metadata else {}),
                }
                for msg in ep.messages
            ],
        }
        if ep.metadata:
            episode_dict["metadata"] = ep.metadata

        data.append(episode_dict)

    write_jsonl(filepath, data)
