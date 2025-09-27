# scripts/sweep_kc.py
import csv, argparse, numpy as np
from simulations.ringing_threshold import GPParams, kc_engineering

parser = argparse.ArgumentParser()
parser.add_argument("--A", type=float, nargs=3, default=[0.1, 1.0, 0.1], help="start stop step")
parser.add_argument("--B", type=float, nargs=3, default=[0.5, 2.0, 0.25])
parser.add_argument("--Delta", type=float, nargs=3, default=[0.0, 0.5, 0.05])
parser.add_argument("--out", type=str, default="results/phase/kc_grid.csv")
args = parser.parse_args()

A_vals = np.arange(*args.A)
B_vals = np.arange(*args.B)
D_vals = np.arange(*args.Delta)

rows=[]
for A in A_vals:
    for B in B_vals:
        for D in D_vals:
            kc = kc_engineering(GPParams(A=A,B=B,Delta=D))
            rows.append(dict(A=A,B=B,Delta=D,Kc_pred=kc))

import os, pathlib
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["A","B","Delta","Kc_pred"])
    w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} rows to {args.out}")
