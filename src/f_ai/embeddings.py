"""
Embedding backends for F_AI.

Default: TF-IDF (lightweight, no heavy dependencies)
Optional: sentence-transformers (better semantic quality)
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from abc import ABC, abstractmethod


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Embed a list of sentences.

        Args:
            sentences: List of sentence strings

        Returns:
            Array of shape (n_sentences, embedding_dim)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class TfidfEmbeddingBackend(EmbeddingBackend):
    """
    TF-IDF based embedding (default).

    Lightweight, deterministic, no heavy dependencies.
    """

    def __init__(
        self,
        max_features: int = 384,
        ngram_range: tuple = (1, 2),
        lowercase: bool = True,
    ):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features (embedding dimension)
            ngram_range: N-gram range for TF-IDF
            lowercase: Whether to lowercase text
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=lowercase,
            dtype=np.float32,
        )
        self.fitted = False

    def fit(self, sentences: List[str]) -> None:
        """
        Fit vectorizer on sentences.

        Args:
            sentences: List of sentences to fit on
        """
        if sentences:
            self.vectorizer.fit(sentences)
            self.fitted = True

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Embed sentences using TF-IDF.

        Args:
            sentences: List of sentence strings

        Returns:
            Array of shape (n_sentences, max_features)
        """
        if not sentences:
            return np.zeros((0, self.max_features), dtype=np.float32)

        # Fit if not already fitted
        if not self.fitted:
            self.fit(sentences)

        # Transform
        embeddings = self.vectorizer.transform(sentences).toarray()

        # Ensure correct dimension (pad if needed)
        if embeddings.shape[1] < self.max_features:
            padding = np.zeros(
                (embeddings.shape[0], self.max_features - embeddings.shape[1]),
                dtype=np.float32,
            )
            embeddings = np.hstack([embeddings, padding])

        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.max_features


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Sentence-transformers backend (optional).

    Requires: sentence-transformers, torch
    Better semantic quality, but heavier dependencies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence-transformer model.

        Args:
            model_name: Model name (default: all-MiniLM-L6-v2, 384 dims)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Embed sentences using transformer model.

        Args:
            sentences: List of sentence strings

        Returns:
            Array of shape (n_sentences, embedding_dim)
        """
        if not sentences:
            dim = self.get_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        embeddings = self.model.encode(
            sentences, show_progress_bar=False, convert_to_numpy=True
        )

        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


def create_embedding_backend(
    backend: str = "tfidf", **kwargs
) -> EmbeddingBackend:
    """
    Factory function to create embedding backend.

    Args:
        backend: "tfidf" or "sentence_transformer"
        **kwargs: Backend-specific arguments

    Returns:
        EmbeddingBackend instance
    """
    if backend == "tfidf":
        return TfidfEmbeddingBackend(**kwargs)
    elif backend == "sentence_transformer":
        return SentenceTransformerBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'tfidf' or 'sentence_transformer'"
        )


class EpisodeEmbedder:
    """
    Compute embeddings for episodes.

    Handles both sentence-level and episode-level embeddings.
    """

    def __init__(self, backend: EmbeddingBackend):
        """
        Initialize embedder.

        Args:
            backend: Embedding backend to use
        """
        self.backend = backend

    def embed_role_sentences(
        self, sentences: List[str]
    ) -> np.ndarray:
        """
        Embed sentences for a role.

        Args:
            sentences: List of sentences

        Returns:
            Array of shape (n_sentences, d) where d is embedding dimension
            Fix #1: sentence-level embeddings clearly labeled
        """
        if not sentences:
            dim = self.backend.get_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        return self.backend.embed_sentences(sentences)

    def compute_episode_embedding(
        self, sentence_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute episode-level embedding from sentence embeddings.

        Args:
            sentence_embeddings: Array of shape (n_sentences, d)

        Returns:
            Array of shape (d,) - mean of sentence embeddings
            Fix #1: episode-level embedding clearly labeled
        """
        if len(sentence_embeddings) == 0:
            dim = self.backend.get_dimension()
            return np.zeros(dim, dtype=np.float32)

        # Mean pooling
        return sentence_embeddings.mean(axis=0).astype(np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.backend.get_dimension()


def embed_episode_roles(
    episode,
    embedder: EpisodeEmbedder,
    roles: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute embeddings for all roles in an episode.

    Args:
        episode: Episode object
        embedder: EpisodeEmbedder instance
        roles: List of roles to embed

    Returns:
        Dict mapping role to:
            - "sentence_embeddings": array of shape (n_sentences, d)
            - "episode_embedding": array of shape (d,)
            - "n_sentences": int

    Fix #1: Clear separation of sentence-level and episode-level embeddings
    """
    from .segment import get_role_sentences

    result = {}

    for role in roles:
        sentences = get_role_sentences(episode, role)

        # Sentence-level embeddings
        sentence_embs = embedder.embed_role_sentences(sentences)

        # Episode-level embedding (mean of sentences)
        episode_emb = embedder.compute_episode_embedding(sentence_embs)

        result[role] = {
            "sentence_embeddings": sentence_embs,  # (n_sentences, d)
            "episode_embedding": episode_emb,  # (d,)
            "n_sentences": len(sentences),
        }

    return result
