"""Graph construction utilities for resonance mapper datasets."""
from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np

from .loader import MultiFreqResults


def build_graph_from_multi_freq(results: MultiFreqResults) -> nx.Graph:
    """Create a graph from :class:`MultiFreqResults` instances."""
    graph = nx.Graph()
    for band in results.bands:
        graph.add_node(
            band.band_id,
            features=np.array([band.lambda_star, band.hysteresis_area, band.transition_sharpness]),
        )
    for key, metrics in results.cross_band.items():
        try:
            u, v = key.split("_")
        except ValueError:
            # Fallback for unexpected keys
            continue
        graph.add_edge(u, v, weight=metrics.get("mi_peak", 0.0))
    return graph


def build_graph_from_spin_foam(data: Dict) -> nx.Graph:
    """Placeholder construction for spin foam lattices."""
    return nx.grid_2d_graph(10, 10)


def build_graph_from_microtubule(data: Dict) -> nx.Graph:
    """Placeholder construction for microtubule coherence chains."""
    return nx.path_graph(100)
