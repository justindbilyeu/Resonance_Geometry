import json, os, math, itertools, random, ast
from pathlib import Path
import numpy as np

def _parse_value(val):
    val = val.strip()
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if not val:
        return {}
    try:
        return ast.literal_eval(val)
    except Exception:
        return val.strip('"')

def load_cfg(path="config/topo_test.yaml"):
    root = {}
    stack = [(-1, root)]
    with open(path, "r") as f:
        for raw in f:
            # remove inline comments
            stripped = raw.split('#', 1)[0].rstrip()
            if not stripped.strip():
                continue
            indent = len(raw) - len(raw.lstrip(' '))
            key, _, remainder = stripped.strip().partition(':')
            if not _:
                continue
            value = remainder.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            container = stack[-1][1]
            if value == "":
                container[key] = {}
                stack.append((indent, container[key]))
            else:
                container[key] = _parse_value(value)
    return root

def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)

def zscore(arr):
    arr = np.asarray(arr)
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-9)

def grid_cells(bounds, res_per_dim):
    """bounds: list of (min,max) for 6 dims; yield centers and spacing."""
    edges = [np.linspace(lo, hi, res_per_dim) for (lo,hi) in bounds]
    spacing = [(hi-lo)/(res_per_dim-1 + 1e-9) for (lo,hi) in bounds]
    for idx in itertools.product(*[range(res_per_dim)]*len(bounds)):
        center = np.array([edges[d][i] for d,i in enumerate(idx)], dtype=float)
        yield center, np.array(spacing)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path)); 
    with open(path, "w") as f: json.dump(obj, f, indent=2)
