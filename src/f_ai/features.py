"""
Feature extraction and Jensen-Shannon Divergence for Goldilocks disagreement.
"""

from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon


def extract_feature_distribution(
    texts: List[str],
    ngram_range: tuple = (1, 2),
    lowercase: bool = True,
    max_features: Optional[int] = None,
) -> np.ndarray:
    """
    Extract feature distribution from texts using CountVectorizer.

    Args:
        texts: List of text strings
        ngram_range: N-gram range (default: (1, 2) for unigrams and bigrams)
        lowercase: Whether to lowercase text
        max_features: Maximum number of features (None for unlimited)

    Returns:
        Probability distribution (normalized counts)
    """
    if not texts:
        return np.array([])

    # Combine all texts
    combined_text = " ".join(texts)

    # Count vectorizer
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        lowercase=lowercase,
        max_features=max_features,
        dtype=np.float64,
    )

    # Get counts
    try:
        counts = vectorizer.fit_transform([combined_text]).toarray()[0]
    except ValueError:
        # Empty vocabulary (e.g., all stop words)
        return np.array([])

    # Normalize to probability distribution
    total = counts.sum()
    if total == 0:
        return np.array([])

    distribution = counts / total

    return distribution


def compute_jsd(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Jensen-Shannon Divergence between two distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small value for numerical stability

    Returns:
        JSD value in [0, 1] (for base 2 logarithm)
    """
    # Handle empty distributions
    if len(p) == 0 or len(q) == 0:
        return 0.0

    # Align distributions to same length
    max_len = max(len(p), len(q))

    p_aligned = np.zeros(max_len)
    q_aligned = np.zeros(max_len)

    p_aligned[: len(p)] = p
    q_aligned[: len(q)] = q

    # Renormalize (in case of padding)
    p_sum = p_aligned.sum()
    q_sum = q_aligned.sum()

    if p_sum > 0:
        p_aligned = p_aligned / p_sum
    if q_sum > 0:
        q_aligned = q_aligned / q_sum

    # Add epsilon for numerical stability
    p_aligned = p_aligned + epsilon
    q_aligned = q_aligned + epsilon

    # Renormalize again
    p_aligned = p_aligned / p_aligned.sum()
    q_aligned = q_aligned / q_aligned.sum()

    # Compute JSD using scipy (returns sqrt of JS divergence by default)
    # We want the divergence itself, so we square it
    jsd = jensenshannon(p_aligned, q_aligned, base=2.0) ** 2

    return float(jsd)


def compute_role_feature_distributions(
    episode,
    roles: List[str],
    ngram_range: tuple = (1, 2),
    lowercase: bool = True,
) -> dict:
    """
    Compute feature distributions for specific roles in an episode.

    Args:
        episode: Episode object
        roles: List of roles to compute distributions for
        ngram_range: N-gram range
        lowercase: Whether to lowercase

    Returns:
        Dict mapping role to probability distribution
    """
    distributions = {}

    for role in roles:
        texts = episode.get_text_by_role(role)
        if texts:
            dist = extract_feature_distribution(
                texts, ngram_range=ngram_range, lowercase=lowercase
            )
            distributions[role] = dist
        else:
            distributions[role] = np.array([])

    return distributions


