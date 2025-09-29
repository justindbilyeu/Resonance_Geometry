"""Topological data analysis helpers."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:  # pragma: no cover - optional dependency branch
    import gudhi as gd
    import persim

    _TDA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency branch
    _TDA_AVAILABLE = False


def compute_tda(embeddings: np.ndarray):
    """Compute persistent homology from embeddings.

    Returns a tuple ``(betti, diagram, landscape)``. When optional dependencies
    are unavailable we return zeros and ``None`` so the pipeline can still run
    in lightweight environments.
    """

    if not _TDA_AVAILABLE:
        return [0, 0, 0], np.zeros((0, 2)), None

    rips = gd.RipsComplex(points=embeddings, max_edge_length=1.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()
    diagram = np.array(
        [
            (birth, death)
            for dim, (birth, death) in persistence
            if death != float("inf")
        ]
    )
    betti = [
        len([item for item in persistence if item[0] == dimension and item[1][1] == float("inf")])
        for dimension in [0, 1, 2]
    ]
    try:
        landscape = persim.landscape(diagram)
    except Exception:  # pragma: no cover - plotting fallbacks
        landscape = None
    if diagram.size:
        plt.scatter(diagram[:, 0], diagram[:, 1])
    return betti, diagram, landscape
