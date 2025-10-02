"""A lightweight subset of the :mod:`networkx` API used for testing."""
from __future__ import annotations

from collections import deque
from itertools import product
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

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

    def remove_node(self, node: object) -> None:
        """Remove ``node`` and any incident edges if present."""

        if node not in self._nodes:
            return
        for neighbor in list(self._adj.get(node, {})):
            self._adj[neighbor].pop(node, None)
            self._edge_attrs.pop((node, neighbor), None)
            self._edge_attrs.pop((neighbor, node), None)
        self._adj.pop(node, None)
        self._nodes.pop(node, None)

    def remove_nodes_from(self, nodes: Iterable[object]) -> None:
        for node in list(nodes):
            self.remove_node(node)

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
        return self._adj.get(node, {}).keys()

    def get_edge_data(self, u: object, v: object) -> Optional[Dict[str, object]]:
        return self._adj.get(u, {}).get(v)

    def is_directed(self) -> bool:
        return False

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


def grid_graph(dim: Sequence[int] | int) -> Graph:
    """Return an undirected grid graph for an N-dimensional lattice."""

    if isinstance(dim, int):
        dimensions = (dim,)
    else:
        dimensions = tuple(int(d) for d in dim)

    if not dimensions:
        raise ValueError("grid_graph requires at least one dimension")
    if any(d <= 0 for d in dimensions):
        raise ValueError("grid_graph dimensions must be positive")

    graph = Graph()
    ranges = [range(d) for d in dimensions]
    for node in product(*ranges):
        graph.add_node(node)
        for axis, size in enumerate(dimensions):
            if node[axis] == 0:
                continue
            neighbor = list(node)
            neighbor[axis] -= 1
            graph.add_edge(node, tuple(neighbor))
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


def single_source_shortest_path_length(
    graph: Graph, source: object
) -> Mapping[object, int]:
    """Compute shortest path lengths from ``source`` using BFS."""

    if source not in graph:
        raise KeyError(f"{source!r} is not a node in the graph")

    lengths: Dict[object, int] = {source: 0}
    queue: deque[object] = deque([source])
    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v in lengths:
                continue
            lengths[v] = lengths[u] + 1
            queue.append(v)
    return lengths


def all_pairs_shortest_path_length(graph: Graph) -> Iterable[Tuple[object, Mapping[object, int]]]:
    """Yield shortest path lengths for all source nodes in ``graph``."""

    for node in graph.nodes():
        yield node, single_source_shortest_path_length(graph, node)


def connected_components(graph: Graph) -> Iterable[set]:
    """Yield the connected components of ``graph`` as sets of nodes."""

    seen: set = set()
    for node in list(graph.nodes()):
        if node in seen:
            continue
        if node not in graph:
            continue
        component = set()
        queue: deque = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor in seen or neighbor not in graph:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        if component:
            yield component
