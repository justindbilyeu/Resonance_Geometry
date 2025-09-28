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

def null1_shuffle_info(x): return np.random.permutation(x)
def null2_remove_geometry(x): return x * np.array([1,1,0,1,0,1])  # zero-out geometric coords as a stub
def null3_rewire_topology(x): return x + np.random.normal(0, 0.5, size=x.shape)  # crude disruptor

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
