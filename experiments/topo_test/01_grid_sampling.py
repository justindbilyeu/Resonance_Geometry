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
grid_cells = common.grid_cells

def synthetic_gp_evolution(seed=0):
    # Stub: returns a 6D state vector ~ N(0,1) to exercise pipeline
    rng = np.random.default_rng(seed)
    return rng.standard_normal(6)

def run_random_exploration(n=400):
    states = [synthetic_gp_evolution(seed=i) for i in range(n)]
    return np.array(states)

def main():
    cfg = load_cfg()
    out_dir = cfg["io"]["out_dir"]
    ensure_dir(out_dir)
    states = run_random_exploration(n=cfg["grid"]["random_starts"])
    save_json({"states": states.tolist()}, f"{out_dir}/random_exploration.json")
    print(f"[grid] random exploration -> {states.shape[0]} samples")

if __name__ == "__main__":
    main()
