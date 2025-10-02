#!/usr/bin/env python3
"""
Chunked Phase-1 runner with checkpointing & memory guard.

It calls a user-provided analysis function `run_phase1_analysis(n_runs, seed)`
that returns a list[dict] of per-run metrics. We only store light metrics.

Outputs:
- checkpoints/chunk_XXXX.json   (each chunk’s results)
- checkpoints/summary.json      (aggregated metrics)
"""
import argparse, json, math, os, gc, time, glob, importlib
from pathlib import Path

def _mem_guard(thresh_pct=80):
    try:
        import psutil
        vm = psutil.virtual_memory()
        if vm.percent >= thresh_pct:
            print(f"[mem] high usage {vm.percent:.1f}% → gc.collect()")
            gc.collect()
    except Exception:
        pass

def _save_json(path, obj):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="experiments.phase1_prediction",
                    help="Module that has run_phase1_analysis(n_runs:int, seed:int)->list[dict]")
    ap.add_argument("--func", default="run_phase1_analysis",
                    help="Function name to call inside the module")
    ap.add_argument("--total_runs", type=int, default=30000)
    ap.add_argument("--chunk_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint_dir", default="checkpoints/phase1")
    ap.add_argument("--summary_out", default="checkpoints/phase1/summary.json")
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    run_fn = getattr(mod, args.func)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(args.total_runs / args.chunk_size)

    # Resume: figure out which chunks exist
    done = {os.path.basename(p) for p in glob.glob(os.path.join(args.checkpoint_dir, "chunk_*.json"))}
    print(f"[chunked] total_runs={args.total_runs}, chunk_size={args.chunk_size}, n_chunks={n_chunks}")
    if done:
        print(f"[chunked] found {len(done)} completed chunks → resuming")

    total_sign = 0
    total_ang = 0.0
    total_cnt = 0

    for ci in range(n_chunks):
        ck = f"chunk_{ci:04d}.json"
        ck_path = os.path.join(args.checkpoint_dir, ck)
        if ck in done:
            print(f"[chunk {ci}] already done → skip")
            # Load to update running summary
            with open(ck_path, "r") as f:
                rows = json.load(f)
            total_cnt += len(rows)
            total_sign += sum(1 for r in rows if r.get("sign_match"))
            total_ang += sum(float(r.get("angular_error", 0.0)) for r in rows)
            continue

        start = ci * args.chunk_size
        end   = min((ci+1)*args.chunk_size, args.total_runs)
        runs  = end - start
        seed  = args.seed + ci

        print(f"[chunk {ci}] runs {start}:{end} (n={runs}) seed={seed}")
        t0 = time.time()
        try:
            rows = run_fn(n_runs=runs, seed=seed)
        except Exception as e:
            print(f"[chunk {ci}] ERROR: {e}")
            continue
        dt = time.time() - t0
        print(f"[chunk {ci}] done in {dt:.2f}s ({len(rows)} rows)")

        _save_json(ck_path, rows)

        total_cnt += len(rows)
        total_sign += sum(1 for r in rows if r.get("sign_match"))
        total_ang += sum(float(r.get("angular_error", 0.0)) for r in rows)

        _mem_guard()

        # live summary
        if total_cnt > 0:
            _save_json(args.summary_out, {
                "completed": total_cnt,
                "sign_acc": total_sign / total_cnt,
                "mean_angular_error": total_ang / total_cnt,
                "total_runs": args.total_runs,
                "chunk_size": args.chunk_size
            })

    if total_cnt > 0:
        _save_json(args.summary_out, {
            "completed": total_cnt,
            "sign_acc": total_sign / total_cnt,
            "mean_angular_error": total_ang / total_cnt,
            "total_runs": args.total_runs,
            "chunk_size": args.chunk_size
        })
        print(f"[summary] completed={total_cnt} sign_acc≈{total_sign/total_cnt:.3f} "
              f"mean_ang≈{total_ang/total_cnt:.3f}")
    print("[chunked] ALL DONE")

if __name__ == "__main__":
    main()
