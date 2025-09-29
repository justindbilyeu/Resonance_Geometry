import json, argparse, numpy as np
from pathlib import Path

# reuse the toy GP stepper from the detector
from experiments.forbidden_region_detector import gp_toy_evolve, param_grid4
from tools.ricci import ollivier_ricci
import networkx as nx

def coupling_to_graph(g, topk=5):
    """Build a simple kNN graph (by absolute weight) from coupling matrix g."""
    n = g.shape[0]
    G = nx.Graph()
    for i in range(n): G.add_node(i)
    for i in range(n):
        # exclude self, take top-k strongest absolute couplings
        idx = np.argsort(-np.abs(g[i]))[:topk+1]  # includes i
        for j in idx:
            if i == j: continue
            w = float(g[i, j])
            if w != 0.0:
                G.add_edge(i, j, weight=abs(w))
    return G

def predict_deflection_from_curvature(G):
    """
    Heuristic: more negative edge curvature => 'repulsive moat'.
    We predict the local flow points in the average direction
    that *decreases* exposure to negative curvature edges.
    Return a scalar 'curvature pressure' and sign (repel/attract).
    """
    # summarize by average edge curvature
    ricci, avg_kappa = ollivier_ricci(G, alpha=0.5, method="sinkhorn", tau=0.02)
    # for a quick scalar predictor, negative avg means 'repelled'
    return avg_kappa  # < 0 predicts outward deflection from high-curvature zone

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/forbidden_v0/forbidden_summary.json")
    ap.add_argument("--samples", type=int, default=64, help="boundary-near initializations")
    ap.add_argument("--steps", type=int, default=8, help="short rollout for deflection estimate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/phase1_prediction")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 1) Load grid extents & visited mask
    summ = json.load(open(args.summary))
    grid_res = summ["grid_res"]  # e.g., 8
    visited = np.load("results/forbidden_v0/visited_4d.npy")  # shape (n,n,n,n) booleans
    n = visited.shape[0]

    # 2) find boundary-near accessible cells (touch at least one forbidden neighbor)
    def neighbors(ix):
        i,j,k,m = ix
        for di in (-1,0,1):
          for dj in (-1,0,1):
            for dk in (-1,0,1):
              for dm in (-1,0,1):
                if (di,dj,dk,dm) == (0,0,0,0): continue
                u,v,w,x = i+di,j+dj,k+dk,m+dm
                if 0 <= u < n and 0 <= v < n and 0 <= w < n and 0 <= x < n:
                    yield (u,v,w,x)

    boundary_cells = []
    it = np.ndindex(visited.shape)
    for idx in it:
        if visited[idx]:  # accessible
            # boundary if any neighbor is forbidden
            if any(not visited[u] for u in neighbors(idx)):
                boundary_cells.append(idx)

    if len(boundary_cells) == 0:
        print("[phase1] no boundary cells found—need a fresh scan output.")
        return

    # 3) sample some boundary cells
    picks = boundary_cells if len(boundary_cells) <= args.samples else [boundary_cells[i] for i in rng.choice(len(boundary_cells), args.samples, replace=False)]

    # 4) map cell index -> param values via the same grid builder
    # we reuse param_grid4 logic but without full construction:
    # assume each axis spans [0,1] for demo; we only need relative deflection vectors
    # if your detector stored true axis ranges, load them here.
    to_param = lambda idx: np.array(idx) / (n - 1)  # crude normalized (λ,β,A,||g||) proxy

    # 5) roll short trajectories & compare deflection to curvature sign
    # metric: sign agreement and angular error between predicted sign and actual step in ||g|| axis
    results = []
    sign_hits = 0
    angles = []

    for cell in picks:
        p0 = to_param(cell)  # 4D param proxy
        # initialize internal GP state from this param proxy
        lam, beta, A, normg = p0
        # small random internal state consistent with ||g|| ~ normg
        d = 32
        g0 = rng.randn(d, d)
        g0 = g0 / np.linalg.norm(g0) * (1e-3 + normg)  # tiny baseline + target norm

        traj = []
        g = g0.copy()
        lam_t, beta_t, A_t = lam, beta, A

        # short rollout
        for t in range(args.steps):
            g = gp_toy_evolve(g, lam_t, beta_t, A_t, dt=0.05, steps=1)
            traj.append(g.copy())

        # actual deflection in parameter proxy = change in ||g||
        norms = np.array([np.linalg.norm(G) for G in traj])
        d_norm = norms[-1] - norms[0]  # positive = moved outward along ||g||

        # curvature prediction (repel if negative)
        G_graph = coupling_to_graph(traj[0])  # evaluate geometry at start
        kappa = predict_deflection_from_curvature(G_graph)
        predicted_sign = -1 if kappa < 0 else +1   # negative curvature predicts outward push

        actual_sign = +1 if d_norm > 0 else -1 if d_norm < 0 else 0
        sign_hits += int(predicted_sign == actual_sign)

        # coarse "angle": compare 1D signs; angle 0 if agree, pi if flip
        angle = 0.0 if predicted_sign == actual_sign else np.pi
        angles.append(angle)

        results.append({
            "cell": list(cell),
            "kappa_avg": float(kappa),
            "pred_sign": int(predicted_sign),
            "d_norm": float(d_norm),
            "actual_sign": int(actual_sign)
        })

    acc = sign_hits / max(1, len(results))
    mean_angle = float(np.mean(angles)) if angles else None

    out = {
        "tested": len(results),
        "sign_accuracy": acc,                # want >> 0.5
        "mean_angular_error_rad": mean_angle,# want << pi/2
        "details": results
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "phase1_prediction_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("[phase1] summary:", out)

if __name__ == "__main__":
    main()
