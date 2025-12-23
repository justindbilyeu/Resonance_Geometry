"""
F_AI Intrinsic Metric v1.0

Geometric and information-theoretic evaluation of AI system dynamics.
"""

__version__ = "1.0.0"
__author__ = "Justin Bilyeu"

from .types import Message, Episode
from .functional import compute_f_ai_episode, compute_f_ai_batch

__all__ = [
    "Message",
    "Episode",
    "compute_f_ai_episode",
    "compute_f_ai_batch",
]
