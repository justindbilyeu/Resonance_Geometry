"""
Test rotation invariance of holonomy (gauge invariance).

Holonomy should be invariant under global rotations of all embeddings.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.f_ai.transport import (
    compute_all_transports,
    compute_all_holonomies,
    compute_average_holonomy,
)


def create_test_role_embeddings(n_roles=4, n_sentences=50, d=10, seed=42):
    """Create synthetic role embeddings for testing."""
    np.random.seed(seed)

    role_names = [f"R{i}" for i in range(n_roles)]
    embeddings = {}

    for role in role_names:
        sentence_embeddings = np.random.randn(n_sentences, d)
        episode_embedding = sentence_embeddings.mean(axis=0)

        embeddings[role] = {
            "sentence_embeddings": sentence_embeddings,
            "episode_embedding": episode_embedding,
            "n_sentences": n_sentences,
        }

    return embeddings, role_names


def create_cycle_topology(role_names):
    """Create a simple cycle topology."""
    edges = [(role_names[i], role_names[(i + 1) % len(role_names)]) for i in range(len(role_names))]

    cycles = [
        {
            "name": "full_cycle",
            "path": edges,
        }
    ]

    return edges, cycles


def test_holonomy_rotation_invariance():
    """Test that holonomy is invariant under global rotation."""
    from scipy.stats import ortho_group

    # Create test embeddings
    n_roles = 4
    d = 10
    embeddings, role_names = create_test_role_embeddings(
        n_roles=n_roles, n_sentences=50, d=d, seed=42
    )

    edges, cycles = create_cycle_topology(role_names)

    # Compute holonomy for original embeddings
    transports_orig = compute_all_transports(embeddings, edges, m_max=100)
    holonomies_orig = compute_all_holonomies(transports_orig, cycles, d)
    U_orig = compute_average_holonomy(holonomies_orig)

    # Apply random rotation to ALL embeddings
    Q = ortho_group.rvs(d)

    embeddings_rotated = {}
    for role in role_names:
        sentence_emb_rotated = embeddings[role]["sentence_embeddings"] @ Q
        episode_emb_rotated = embeddings[role]["episode_embedding"] @ Q

        embeddings_rotated[role] = {
            "sentence_embeddings": sentence_emb_rotated,
            "episode_embedding": episode_emb_rotated,
            "n_sentences": embeddings[role]["n_sentences"],
        }

    # Compute holonomy for rotated embeddings
    transports_rot = compute_all_transports(embeddings_rotated, edges, m_max=100)
    holonomies_rot = compute_all_holonomies(transports_rot, cycles, d)
    U_rot = compute_average_holonomy(holonomies_rot)

    # Holonomy should be approximately the same (rotation invariant)
    np.testing.assert_allclose(U_orig, U_rot, atol=1e-4, rtol=1e-3)

    print(f"Original holonomy: {U_orig:.6f}")
    print(f"Rotated holonomy: {U_rot:.6f}")
    print(f"Difference: {abs(U_orig - U_rot):.8f}")


def test_holonomy_range():
    """Test that holonomy values are in expected range [0, 2]."""
    embeddings, role_names = create_test_role_embeddings()
    edges, cycles = create_cycle_topology(role_names)

    d = 10
    transports = compute_all_transports(embeddings, edges, m_max=100)
    holonomies = compute_all_holonomies(transports, cycles, d)

    for name, H in holonomies.items():
        assert 0.0 <= H <= 2.0, f"Holonomy {name} = {H} out of range [0, 2]"


def test_holonomy_normalized_by_dimension():
    """Test that holonomy is properly normalized by dimension (Fix #3)."""
    # Create identical embeddings (should give low holonomy)
    embeddings, role_names = create_test_role_embeddings(seed=42)
    edges, cycles = create_cycle_topology(role_names)

    # Test with different dimensions
    for d in [5, 10, 20, 50]:
        # Resize embeddings to dimension d
        embeddings_d = {}
        for role in role_names:
            sentence_emb = np.random.randn(50, d)
            episode_emb = sentence_emb.mean(axis=0)

            embeddings_d[role] = {
                "sentence_embeddings": sentence_emb,
                "episode_embedding": episode_emb,
                "n_sentences": 50,
            }

        transports = compute_all_transports(embeddings_d, edges, m_max=100)
        holonomies = compute_all_holonomies(transports, cycles, d)

        # All holonomies should be in [0, 2] regardless of dimension
        for name, H in holonomies.items():
            assert 0.0 <= H <= 2.0, f"Dimension {d}: Holonomy {H} out of range"


def test_zero_holonomy_for_consistent_geometry():
    """Test that geometrically consistent embeddings give low holonomy."""
    # Create embeddings where all roles have very similar geometry
    # (same point cloud up to small noise)

    n_roles = 4
    d = 10
    base_embeddings = np.random.randn(50, d)

    role_names = [f"R{i}" for i in range(n_roles)]
    embeddings = {}

    for role in role_names:
        # Add small noise to base embeddings
        noise = 0.01 * np.random.randn(50, d)
        sentence_emb = base_embeddings + noise
        episode_emb = sentence_emb.mean(axis=0)

        embeddings[role] = {
            "sentence_embeddings": sentence_emb,
            "episode_embedding": episode_emb,
            "n_sentences": 50,
        }

    edges, cycles = create_cycle_topology(role_names)

    transports = compute_all_transports(embeddings, edges, m_max=100)
    holonomies = compute_all_holonomies(transports, cycles, d)
    U = compute_average_holonomy(holonomies)

    # Holonomy should be very small (near zero)
    assert U < 0.1, f"Expected low holonomy for consistent geometry, got {U}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
