"""
Smoke tests for F_AI intrinsic metric.

These tests verify basic functionality without requiring heavy dependencies.
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.f_ai.types import Message, Episode, EpisodeMetrics
from src.f_ai.io import load_episodes, load_json_config, load_coefficients
from src.f_ai.embeddings import create_embedding_backend, EpisodeEmbedder
from src.f_ai.segment import split_into_sentences, get_role_sentences


def test_message_creation():
    """Test Message dataclass creation."""
    msg = Message(role="U", text="Hello world")
    assert msg.role == "U"
    assert msg.text == "Hello world"
    assert msg.token_count() > 0


def test_episode_creation():
    """Test Episode dataclass creation."""
    messages = [
        Message(role="U", text="Hello"),
        Message(role="B", text="Hi there"),
    ]
    episode = Episode(
        session_id="S001",
        episode_id=1,
        topology_id="unified",
        messages=messages,
    )
    assert episode.session_id == "S001"
    assert episode.episode_id == 1
    assert len(episode.messages) == 2
    assert episode.get_roles() == ["U", "B"]


def test_sentence_splitting():
    """Test sentence splitting utility."""
    text = "This is sentence one. This is sentence two! Is this sentence three?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert "sentence one" in sentences[0]
    assert "sentence two" in sentences[1]
    assert "sentence three" in sentences[2]


def test_tfidf_backend():
    """Test TF-IDF embedding backend."""
    backend = create_embedding_backend("tfidf", max_features=100)
    embedder = EpisodeEmbedder(backend)

    sentences = ["Hello world", "This is a test", "Another sentence"]
    embeddings = embedder.embed_role_sentences(sentences)

    assert embeddings.shape == (3, 100)
    assert embeddings.dtype == np.float32


def test_load_configs():
    """Test loading configuration files."""
    # Test topology config
    topology_config_path = "configs/f_ai/topology_configs.json"
    if Path(topology_config_path).exists():
        config = load_json_config(topology_config_path)
        assert "topologies" in config
        assert "role_split_BSIM" in config["topologies"]

    # Test coefficients
    coeff_path = "configs/f_ai/coefficients.json"
    if Path(coeff_path).exists():
        coeffs = load_coefficients(coeff_path)
        assert "alpha" in coeffs
        assert coeffs["alpha"] == 1.0


def test_load_sample_fixture():
    """Test loading sample fixture JSONL."""
    fixture_path = "tests/fixtures/f_ai_sample.jsonl"

    if not Path(fixture_path).exists():
        pytest.skip("Sample fixture not found")

    episodes = load_episodes(fixture_path, validate=False)

    assert len(episodes) > 0

    # Check first episode
    ep = episodes[0]
    assert ep.session_id == "S001"
    assert ep.episode_id == 1
    assert ep.topology_id == "role_split_BSIM"
    assert len(ep.messages) == 5  # U, B, S, I, M


def test_episode_role_extraction():
    """Test extracting role sentences from episode."""
    messages = [
        Message(role="U", text="First message."),
        Message(role="U", text="Second message."),
        Message(role="B", text="Belief response."),
    ]
    episode = Episode(
        session_id="S001",
        episode_id=1,
        topology_id="unified",
        messages=messages,
    )

    u_sentences = get_role_sentences(episode, "U")
    assert len(u_sentences) == 2
    assert "First message" in u_sentences[0]

    b_sentences = get_role_sentences(episode, "B")
    assert len(b_sentences) == 1


def test_episode_metrics_creation():
    """Test EpisodeMetrics dataclass."""
    metrics = EpisodeMetrics(
        session_id="S001",
        episode_id=1,
        topology_id="role_split_BSIM",
        C=0.8,
        U=0.1,
        D=0.05,
        P=0.7,
        N=2.5,
        C_z=1.2,
        U_z=-0.5,
        D_z=-0.8,
        P_z=0.9,
        N_z=0.3,
        F_AI=1.5,
    )

    assert metrics.F_AI == 1.5
    assert metrics.C == 0.8

    # Test to_dict
    d = metrics.to_dict()
    assert d["F_AI"] == 1.5
    assert d["session_id"] == "S001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
