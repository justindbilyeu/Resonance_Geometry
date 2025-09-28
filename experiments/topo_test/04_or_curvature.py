import numpy as np
import sys
from pathlib import Path
from importlib import import_module

if __package__:
    common = import_module(".00_common", __package__)
else:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    common = import_module("experiments.topo_test.00_common")
load_cfg = common.load_cfg
ensure_dir = common.ensure_dir
save_json = common.save_json

def build_stub_graph():
    nodes = list(range(20))
    edges = []
    for i in range(10):
        for j in range(i + 1, 10):
            edges.append((i, j, 1.0))
    for i in range(10, 20):
        for j in range(i + 1, 20):
            edges.append((i, j, 1.0))
    edges.append((2, 15, 0.2))
    edges.append((4, 18, 0.2))
    return nodes, edges

def ollivier_ricci_stub(edges, alpha=0.5):
    # crude proxy: negative curvature on edges bridging communities
    curv = {}
    comp_id = {n: (0 if n < 10 else 1) for n in range(20)}
    for u, v, w in edges:
        key = (u, v)
        curv[key] = -0.2 if comp_id[u] != comp_id[v] else 0.05
    return curv

def main():
    cfg = load_cfg(); out_dir = cfg["io"]["out_dir"]; ensure_dir(out_dir)
    nodes, edges = build_stub_graph()
    curv = ollivier_ricci_stub(edges, alpha=cfg["curvature"]["alpha"])
    neg_edges = [e for e, k in curv.items() if k < cfg["curvature"]["neg_threshold"]]
    coverage = len(neg_edges) / max(1, len(edges))
    save_json({"avg_kappa": float(np.mean(list(curv.values()))),
               "neg_edge_fraction": coverage}, f"{out_dir}/curvature_report.json")
    print(f"[curvature] neg_edge_fraction={coverage:.2f}")

if __name__ == "__main__":
    main()
