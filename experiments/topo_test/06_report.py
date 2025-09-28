import json, os
import sys
from pathlib import Path
from importlib import import_module

if __package__:
    common = import_module(".00_common", __package__)
else:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    common = import_module("experiments.topo_test.00_common")
load_cfg = common.load_cfg

def load(path):
    with open(path,"r") as f: return json.load(f)

def main():
    cfg = load_cfg(); out = cfg["io"]["out_dir"]
    fdim = load(f"{out}/fractal_dim.json") if os.path.exists(f"{out}/fractal_dim.json") else {}
    curv = load(f"{out}/curvature_report.json") if os.path.exists(f"{out}/curvature_report.json") else {}
    nulls = load(f"{out}/nulls.json") if os.path.exists(f"{out}/nulls.json") else {}
    summary = {
      "fractal_H": fdim.get("H_estimate"),
      "fractal_R2": fdim.get("r2"),
      "curvature_neg_fraction": curv.get("neg_edge_fraction"),
      "nulls": nulls.get("null_reports")
    }
    print("[summary]", summary)

if __name__ == "__main__":
    main()
