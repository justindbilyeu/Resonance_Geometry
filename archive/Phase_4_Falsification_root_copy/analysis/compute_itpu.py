#!/usr/bin/env python3
"""
Compute ITPU = lambda * phi * (1 - kappa) from phase4_results.csv
and write a human-readable summary to itpu_summary.md.

Stdlib only. Run from repo root or this folder.
"""

from __future__ import annotations
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "phase4_results.csv"
OUT_PATH = HERE / "itpu_summary.md"

def itpu(phi: float, kappa: float, lam: float) -> float:
    return lam * phi * (1.0 - kappa)

def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield {
                "model": row["model"],
                "branch": row["branch"],
                "decision": row["decision"],
                "phi": float(row["phi"]),
                "kappa": float(row["kappa"]),
                "lambda": float(row["lambda"]),
            }

def main():
    rows = list(load_rows())
    for r in rows:
        r["itpu"] = itpu(r["phi"], r["kappa"], r["lambda"])

    # branch aggregates
    by_branch = {"A": [], "B": []}
    for r in rows:
        by_branch[r["branch"]].append(r["itpu"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    mean_A = mean(by_branch["A"])
    mean_B = mean(by_branch["B"])
    sep = (mean_A / mean_B) if (mean_A and mean_B) else float("inf")

    # write summary
    lines = []
    lines.append("# ITPU Summary (Phase 4)\n")
    lines.append("| Model | Branch | Decision | Φ | κ | λ | ITPU = λ·Φ·(1−κ) |")
    lines.append("|-------|--------|----------|---|---|---|-------------------|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['branch']} | {r['decision']} | "
            f"{r['phi']:.2f} | {r['kappa']:.2f} | {r['lambda']:.2f} | {r['itpu']:.3f} |"
        )
    lines.append("")
    lines.append(f"**Branch A mean ITPU:** {mean_A:.3f}")
    lines.append(f"**Branch B mean ITPU:** {mean_B:.3f}")
    lines.append(f"**Separation (A/B):** {sep:.1f}×")
    lines.append("")
    lines.append("Interpretation: higher ITPU indicates higher theoretical information throughput potential.\n")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
