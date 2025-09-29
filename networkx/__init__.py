"""A lightweight subset of the :mod:`networkx` API used for testing."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[object, Dict[str, object]] = {}
        self._adj: Dict[object, Dict[object, Dict[str, object]]] = {}
        self._edge_attrs: Dict[Tuple[object, object], Dict[str, object]] = {}

    def add_node(self, node: object, **attrs: object) -> None:
        data = self._nodes.setdefault(node, {})
        data.update(attrs)
        self._adj.setdefault(node, {})

    def add_edge(self, u: object, v: object, **attrs: object) -> None:
        for node in (u, v):
            self.add_node(node)
        self._adj[u][v] = dict(attrs)
        self._adj[v][u] = dict(attrs)
        self._edge_attrs[(u, v)] = dict(attrs)
        self._edge_attrs[(v, u)] = dict(attrs)

    def nodes(self) -> Iterable[object]:
        return self._nodes.keys()

    @property
    def nodes_data(self) -> Dict[object, Dict[str, object]]:
        return self._nodes

    def __iter__(self) -> Iterator[object]:
        return iter(self._nodes)

    def __contains__(self, item: object) -> bool:
        return item in self._nodes

    def degree(self) -> Dict[object, int]:
        return {node: len(adj) for node, adj in self._adj.items()}

    @property
    def edges(self) -> List[Tuple[object, object]]:
        seen = set()
        edges = []
        for u, adj in self._adj.items():
            for v in adj:
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                edges.append((u, v))
        return edges

    def neighbors(self, node: object) -> Iterable[object]:
        return self._adj[node].keys()

    def get_edge_data(self, u: object, v: object) -> Optional[Dict[str, object]]:
        return self._adj.get(u, {}).get(v)

    def adjacency(self) -> Dict[object, Dict[object, Dict[str, object]]]:
        return self._adj

    @property
    def nodes_view(self) -> Dict[object, Dict[str, object]]:
        return self._nodes


def to_numpy_array(graph: Graph, nodelist: Optional[List[object]] = None, dtype=float) -> np.ndarray:
    if nodelist is None:
        nodelist = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodelist)}
    matrix = np.zeros((len(nodelist), len(nodelist)), dtype=dtype)
    for u in nodelist:
        for v in graph.adjacency().get(u, {}):
            matrix[index[u], index[v]] = 1
    return matrix


def grid_graph(dim: Tuple[int, int]) -> Graph:
    rows, cols = dim
    graph = Graph()
    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            graph.add_node(node)
            if r > 0:
                graph.add_edge(node, (r - 1, c))
            if c > 0:
                graph.add_edge(node, (r, c - 1))
    return graph


def grid_2d_graph(rows: int, cols: int) -> Graph:
    return grid_graph((rows, cols))


def path_graph(length: int) -> Graph:
    graph = Graph()
    prev = None
    for idx in range(length):
        graph.add_node(idx)
        if prev is not None:
            graph.add_edge(prev, idx)
        prev = idx
    return graph
