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

def boxcount(points, eps):
    # minimal 6D grid counting
    pts = np.asarray(points)
    mins = pts.min(0); maxs = pts.max(0)
    bins = np.ceil((maxs - mins)/eps).astype(int) + 1
    keys = set()
    for p in pts:
        idx = tuple(((p - mins)/eps).astype(int))
        keys.add(idx)
    return len(keys)

def estimate_fractal_dimension(boundary_pts, cfg):
    eps_list = np.logspace(np.log10(cfg["fractal"]["eps_min"]), np.log10(cfg["fractal"]["eps_max"]), cfg["fractal"]["n_scales"])
    N = []
    for e in eps_list:
        N.append(boxcount(boundary_pts, e))
    logx = np.log(1.0/eps_list); logy = np.log(np.maximum(1, N))
    slope, intercept = np.polyfit(logx, logy, 1)
    # crude R2
    yhat = slope*logx + intercept
    ss_res = np.sum((logy - yhat)**2)
    ss_tot = np.sum((logy - np.mean(logy))**2) + 1e-9
    r2 = 1.0 - ss_res/ss_tot
    return slope, r2

def main():
    cfg = load_cfg(); out_dir = cfg["io"]["out_dir"]; ensure_dir(out_dir)
    # stub boundary (ring in 6D projected to 2D plane then lifted)
    theta = np.linspace(0, 2*np.pi, 400)
    xy = np.c_[np.cos(theta), np.sin(theta)]
    pts = np.hstack([xy, np.zeros((len(theta), 4))])
    H, r2 = estimate_fractal_dimension(pts, cfg)
    save_json({"H_estimate": float(H), "r2": float(r2)}, f"{out_dir}/fractal_dim.json")
    print(f"[fractal] H≈{H:.2f}, R²={r2:.2f}")

if __name__ == "__main__":
    main()
