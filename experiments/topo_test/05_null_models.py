import numpy as np
import sys
from pathlib import Path
from importlib import import_module

import networkx as nx

if __package__:
    common = import_module(".00_common", __package__)
else:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    common = import_module("experiments.topo_test.00_common")
load_cfg = common.load_cfg
ensure_dir = common.ensure_dir
save_json = common.save_json

def null1_shuffle_info(x): return np.random.permutation(x)
def null2_remove_geometry(x): return x * np.array([1,1,0,1,0,1])  # zero-out geometric coords as a stub
def null3_rewire_topology(x): return x + np.random.normal(0, 0.5, size=x.shape)  # crude disruptor


def phase_shuffle(series: np.ndarray, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(series)
    phases = np.angle(spectrum)
    magnitudes = np.abs(spectrum)
    shuffled = phases.copy()
    if shuffled.size > 2:
        shuffled[1:-1] = rng.permutation(shuffled[1:-1])
    randomized = magnitudes * np.exp(1j * shuffled)
    result = np.fft.irfft(randomized, n=series.shape[0])
    return result.astype(series.dtype)


def _copy_graph(graph: nx.Graph) -> nx.Graph:
    clone = nx.Graph()
    for node in graph.nodes():
        attrs = {}
        try:
            attrs = graph.nodes[node]
        except Exception:
            attrs = getattr(graph, "nodes_data", {}).get(node, {})
        clone.add_node(node, **(attrs or {}))
    for u, v in graph.edges:
        data = graph.get_edge_data(u, v) or {}
        clone.add_edge(u, v, **data)
    return clone


def rewire_graph(graph: nx.Graph, swaps: int = 5, seed: int | None = None) -> nx.Graph:
    rng = np.random.default_rng(seed)
    rewired = _copy_graph(graph)
    edges = list(rewired.edges)
    if len(edges) < 2:
        return rewired
    for _ in range(swaps):
        idx1, idx2 = rng.choice(len(edges), size=2, replace=False)
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]
        if len({u1, v1, u2, v2}) < 4:
            continue
        candidates = [(u1, v2), (u2, v1)]
        if any(((a, b) in edges or (b, a) in edges or a == b) for a, b in candidates):
            continue
        edges[idx1] = candidates[0]
        edges[idx2] = candidates[1]
    rewired = _copy_graph(graph)
    rewired_edges = set()
    for u, v in edges:
        if (u, v) in rewired_edges or (v, u) in rewired_edges or u == v:
            continue
        rewired_edges.add((u, v))
    rewired = nx.Graph()
    for node in graph.nodes():
        attrs = getattr(graph, "nodes_data", {}).get(node, {}) if hasattr(graph, "nodes_data") else {}
        try:
            attrs = graph.nodes[node]
        except Exception:
            pass
        rewired.add_node(node, **(attrs or {}))
    for u, v in rewired_edges:
        rewired.add_edge(u, v)
    return rewired

def main():
    cfg = load_cfg(); out_dir = cfg["io"]["out_dir"]; ensure_dir(out_dir)
    base = np.random.normal(size=(400,6))
    reports = {}
    if cfg["nulls"]["run_null1"]: reports["null1"] = float(np.std(null1_shuffle_info(base)))
    if cfg["nulls"]["run_null2"]: reports["null2"] = float(np.std(null2_remove_geometry(base)))
    if cfg["nulls"]["run_null3"]: reports["null3"] = float(np.std(null3_rewire_topology(base)))
    save_json({"null_reports": reports}, f"{out_dir}/nulls.json")
    print(f"[nulls] reports: {reports}")

if __name__ == "__main__":
    main()
