cat > experiments/phase1_prediction.py << 'PY'
import numpy as np
from typing import List, Dict
from experiments.forbidden_region_detector import gp_toy_evolve
from tools.geom_predict import finite_diff_grad, angle_between

# Try OR; otherwise we use boundary-density fallback
try:
    from tools.ricci import ollivier_ricci
    import networkx as nx
    HAVE_OR = True
except Exception:
    HAVE_OR = False

# Optional fallback: use visited_4d grid if present to define a scalar field
_V_GRID = None
try:
    import json, pathlib
    summ = json.load(open("results/forbidden_v1/forbidden_summary.json"))
    vpath = summ.get("visited_path","results/forbidden_v1/visited_4d.npy")
    import numpy as _np
    _V_GRID = _np.load(vpath).astype(np.uint8)  # 1=visited, 0=never
except Exception:
    _V_GRID = None

def _local_graph_kappa_or(x, n=8) -> float:
    """
    κ(x) via OR on a tiny 4D L1 star around the nearest gridpoint to x.
    """
    if not HAVE_OR:
        return 0.0
    xi = np.clip((np.asarray(x)*(n-1)).round().astype(int), 0, n-1)
    pts = [tuple(xi)]
    for d in range(4):
        for s in (-1, 1):
            p = xi.copy(); p[d] += s
            if 0 <= p[d] < n:
                pts.append(tuple(p))
    pts = list(dict.fromkeys(pts))
    G = nx.Graph()
    for i,p in enumerate(pts):
        G.add_node(i, coord=np.array(p, dtype=float))
    for i,pi in enumerate(pts):
        for j,pj in enumerate(pts):
            if i < j and sum(abs(a-b) for a,b in zip(pi,pj)) == 1:
                G.add_edge(i,j,weight=1.0)
    kappa_edges, _ = ollivier_ricci(G, alpha=0.5, method="exact")
    if not kappa_edges:
        return 0.0
    center_idx = 0
    vals = []
    for (u,v), kv in kappa_edges.items():
        if kv is None: 
            continue
        if u == center_idx or v == center_idx:
            vals.append(kv)
    if not vals:
        vals = [kv for kv in kappa_edges.values() if kv is not None]
    return float(np.mean(vals)) if vals else 0.0

def _grid_mean_in_cube(V, idx, rad=1):
    n = V.shape[0]
    i,j,k,m = idx
    i0,i1 = max(0,i-rad), min(n-1,i+rad)
    j0,j1 = max(0,j-rad), min(n-1,j+rad)
    k0,k1 = max(0,k-rad), min(n-1,k+rad)
    m0,m1 = max(0,m-rad), min(n-1,m+rad)
    block = V[i0:i1+1, j0:j1+1, k0:k1+1, m0:m1+1]
    return float(block.mean()) if block.size else 0.0

def _kappa_fallback(x, n=8) -> float:
    """
    κ(x) fallback: use boundary-density from visited_4d grid.
    Higher κ near mixed visited/forbidden regions.
    """
    if _V_GRID is None:
        return 0.0
    xi = np.clip((np.asarray(x)*(n-1)).round().astype(int), 0, n-1)
    v_mean = _grid_mean_in_cube(_V_GRID, tuple(xi), rad=1)   # local visited density
    # define κ as "forbidden density" locally; smooth-ish scalar field
    return float(1.0 - v_mean)

def _kappa_field(x, n=8) -> float:
    # prefer OR if available; else fallback to boundary-density
    if HAVE_OR:
        return _local_graph_kappa_or(x, n=n)
    return _kappa_fallback(x, n=n)

def _actual_deflection(traj, tail=100):
    T = len(traj)
    if T < 4:
        return np.zeros(4)
    tail = min(tail, T//2)
    head = min(tail, T - tail)
    v_init  = np.mean(np.diff(traj[:head], axis=0), axis=0) if head>1 else (traj[min(1,T-1)]-traj[0])
    v_final = np.mean(np.diff(traj[-tail:], axis=0), axis=0) if tail>1 else (traj[-1]-traj[-2])
    return v_final - v_init

def run_phase1_analysis(n_runs: int = 1000, seed: int = 42) -> List[Dict]:
    """
    Geometry-only prediction of deflection vs. actual deflection.
    Returns list of {"sign_match":bool, "angular_error":float}.
    """
    rng = np.random.default_rng(seed)
    results: List[Dict] = []
    n = 8

    for _ in range(n_runs):
        # 1) Random start + GP evolution
        x0 = rng.random(4)
        traj = gp_toy_evolve(x0, steps=400, dt=0.02,
                             seed=int(rng.integers(0, 2**31-1)))
        traj = np.asarray(traj, dtype=float)
        traj = np.clip(traj, 0.0, 1.0)  # ensure [0,1]^4

        # 2) Actual deflection from trajectory tail
        d_actual = _actual_deflection(traj, tail=100)

        # 3) Geometric prediction: −∇κ(x0)
        grad_k = finite_diff_grad(lambda z: _kappa_field(z, n=n), traj[0], h=1e-2)
        d_geom = -grad_k

        # 4) Compare
        ang = angle_between(d_geom, d_actual)
        results.append({
            "sign_match": bool(np.sign(d_geom[0]) == np.sign(d_actual[0])),
            "angular_error": float(ang)
        })

    return results
PY
