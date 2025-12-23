"""
Data types for F_AI intrinsic metric.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Message:
    """A single message in an episode."""

    role: str
    text: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate message."""
        if not self.role:
            raise ValueError("Message role cannot be empty")
        if not self.text:
            raise ValueError("Message text cannot be empty")

    def token_count(self) -> int:
        """Approximate token count (simple word-based)."""
        return len(self.text.split())


@dataclass
class Episode:
    """A conversational episode with multiple roles."""

    session_id: str
    episode_id: int
    topology_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate episode."""
        if not self.session_id:
            raise ValueError("Episode session_id cannot be empty")
        if self.episode_id < 1:
            raise ValueError("Episode episode_id must be >= 1")
        if not self.topology_id:
            raise ValueError("Episode topology_id cannot be empty")
        if not self.messages:
            raise ValueError("Episode must have at least one message")

    def get_roles(self) -> List[str]:
        """Get list of unique roles in episode."""
        return list(dict.fromkeys(msg.role for msg in self.messages))

    def get_messages_by_role(self, role: str) -> List[Message]:
        """Get all messages for a specific role."""
        return [msg for msg in self.messages if msg.role == role]

    def get_text_by_role(self, role: str) -> List[str]:
        """Get all text for a specific role."""
        return [msg.text for msg in self.messages if msg.role == role]

    def total_tokens(self) -> int:
        """Total token count across all messages."""
        return sum(msg.token_count() for msg in self.messages)

    def has_role(self, role: str) -> bool:
        """Check if episode contains a specific role."""
        return any(msg.role == role for msg in self.messages)

    def validate_topology(self, topology_config: Dict[str, Any]) -> bool:
        """
        Validate episode against topology configuration.

        Args:
            topology_config: Topology configuration dict

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required_roles = set(topology_config.get("required_roles", []))
        optional_roles = set(topology_config.get("optional_roles", []))
        allowed_roles = required_roles | optional_roles
        episode_roles = set(self.get_roles())

        # Check for invalid roles
        invalid_roles = episode_roles - allowed_roles
        if invalid_roles:
            raise ValueError(
                f"Episode contains invalid roles for topology {self.topology_id}: "
                f"{invalid_roles}"
            )

        # Check for missing required roles
        missing_roles = required_roles - episode_roles
        if missing_roles:
            raise ValueError(
                f"Episode missing required roles for topology {self.topology_id}: "
                f"{missing_roles}"
            )

        return True


@dataclass
class EpisodeMetrics:
    """Computed metrics for an episode."""

    session_id: str
    episode_id: int
    topology_id: str

    # Raw metrics
    C: float  # Coupling
    U: float  # Holonomy uncertainty
    D: float  # Goldilocks disagreement
    P: float  # Persistence
    N: float  # Complexity

    # Normalized metrics (z-scores)
    C_z: float
    U_z: float
    D_z: float
    P_z: float
    N_z: float

    # Final F_AI score
    F_AI: float

    # Additional info
    n_messages: int = 0
    n_tokens: int = 0
    roles_present: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "topology_id": self.topology_id,
            "F_AI": self.F_AI,
            "C": self.C,
            "U": self.U,
            "D": self.D,
            "P": self.P,
            "N": self.N,
            "C_z": self.C_z,
            "U_z": self.U_z,
            "D_z": self.D_z,
            "P_z": self.P_z,
            "N_z": self.N_z,
            "n_messages": self.n_messages,
            "n_tokens": self.n_tokens,
            "roles_present": self.roles_present,
        }


@dataclass
class NormalizationParams:
    """Parameters for robust z-score normalization."""

    median: float
    iqr: float
    epsilon: float = 1e-8

    def normalize(self, x: float) -> float:
        """Apply robust z-score normalization."""
        return (x - self.median) / (self.iqr + self.epsilon)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"median": self.median, "iqr": self.iqr}


@dataclass
class CalibrationData:
    """Normalization parameters from calibration split."""

    topology_id: str
    C_params: NormalizationParams
    U_params: NormalizationParams
    D_params: NormalizationParams
    P_params: NormalizationParams
    N_params: NormalizationParams
    n_calibration_episodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topology_id": self.topology_id,
            "n_calibration_episodes": self.n_calibration_episodes,
            "C": self.C_params.to_dict(),
            "U": self.U_params.to_dict(),
            "D": self.D_params.to_dict(),
            "P": self.P_params.to_dict(),
            "N": self.N_params.to_dict(),
        }
