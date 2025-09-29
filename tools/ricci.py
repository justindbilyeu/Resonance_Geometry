"""Lightweight Ollivier--Ricci curvature utilities for grid-based experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog

__all__ = ["ollivier_ricci"]


@dataclass(slots=True)
class _Measure:
    support: list[object]
    mass: np.ndarray


def _node_measure(graph: nx.Graph, node: object, alpha: float) -> _Measure:
    neighbors = list(graph.neighbors(node))
    support = [node] + neighbors
    mass = np.empty(len(support), dtype=float)
    if neighbors:
        mass[0] = alpha
        mass[1:] = (1.0 - alpha) / len(neighbors)
    else:
        mass[0] = 1.0
    return _Measure(support=support, mass=mass)


def _all_pairs_distances(graph: nx.Graph) -> Mapping[object, Mapping[object, float]]:
    return {node: dict(lengths) for node, lengths in nx.all_pairs_shortest_path_length(graph)}


def _earth_mover_distance(
    support_a: Iterable[object],
    support_b: Iterable[object],
    mass_a: np.ndarray,
    mass_b: np.ndarray,
    distances: Mapping[object, Mapping[object, float]],
) -> float:
    sup_a = list(support_a)
    sup_b = list(support_b)
    m, n = len(sup_a), len(sup_b)
    cost = np.zeros((m, n), dtype=float)
    for i, node_a in enumerate(sup_a):
        da = distances[node_a]
        for j, node_b in enumerate(sup_b):
            cost[i, j] = da.get(node_b, np.inf)
    if not np.isfinite(cost).all():
        finite = cost[np.isfinite(cost)]
        if finite.size == 0:
            raise RuntimeError("Disconnected supports encountered in EMD computation")
        penalty = float(finite.max() * 2.0)
        cost[~np.isfinite(cost)] = penalty
    cvec = cost.reshape(-1)

    a_eq = []
    b_eq = []
    for i in range(m):
        row = np.zeros(m * n)
        row[i * n : (i + 1) * n] = 1.0
        a_eq.append(row)
        b_eq.append(mass_a[i])
    for j in range(n):
        row = np.zeros(m * n)
        row[j::n] = 1.0
        a_eq.append(row)
        b_eq.append(mass_b[j])
    bounds = [(0.0, None)] * (m * n)
    result = linprog(
        cvec,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"linprog failed: {result.message}")
    return float(result.fun)


def ollivier_ricci(
    graph: nx.Graph,
    *,
    alpha: float = 0.5,
    method: str | None = None,
) -> Tuple[Dict[Tuple[object, object], float], float]:
    """Compute Ollivier--Ricci curvature for each edge in ``graph``.

    Parameters
    ----------
    graph:
        NetworkX graph (treated as unweighted) describing the adjacency.
    alpha:
        Idle mass retained at each node. ``alpha=0`` recovers the usual lazy random walk.
    method:
        Currently ignored; present for API compatibility with historical callers.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    distances = _all_pairs_distances(graph)
    measures = {node: _node_measure(graph, node, alpha) for node in graph.nodes}

    ricci: Dict[Tuple[object, object], float] = {}
    values = []
    undirected = not graph.is_directed()
    for u, v in graph.edges:
        mu = measures[u]
        mv = measures[v]
        w1 = _earth_mover_distance(mu.support, mv.support, mu.mass, mv.mass, distances)
        kappa = 1.0 - w1  # edge length assumed to be 1
        if undirected:
            try:
                key = (u, v) if u <= v else (v, u)
            except TypeError:
                ordered = sorted((u, v), key=repr)
                key = (ordered[0], ordered[1])
        else:
            key = (u, v)
        ricci[key] = float(kappa)
        values.append(kappa)
    avg = float(np.mean(values)) if values else float("nan")
    return ricci, avg
