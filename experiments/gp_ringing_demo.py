# experiments/gp_ringing_demo.py
"""
Lightweight, CI-safe ringing demo.

- Accepts CLI flags: --steps, --runs, --seeds, --out
- Honors RG_CI=1 to clamp heavy workloads
- Generates a tiny synthetic damped/forced oscillator to emulate "ringing"
- Writes results/<out>/ringing_demo_summary.json with simple metrics

This file avoids heavy plotting and large arrays to keep CI fast.
"""
from __future__ import annotations
import os, json, math, pathlib, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

@dataclass
class RunMetrics:
    peaks: int
    overshoot_rate: float
    rms: float
    ringing_detected: bool

@dataclass
class Summary:
    steps: int
    runs: int
    seeds: int
    beta: float
    alpha: float
    tau: float
    Kc_proxy: float
    ringing_fraction: float
    avg_peaks: float
    avg_overshoot: float
    avg_rms: float
    notes: List[str]

def _oscillator_series(steps: int, alpha: float, beta: float, tau: float, seed: int) -> List[float]:
    """
    Toy discrete-time damped/forced oscillator:
        x_{t+1} = (1 - alpha)*x_t + beta * sin(2π t / tau) + η_t
    """
    rng = random.Random(seed)
    x = 0.0
    series = []
    for t in range(steps):
        drive = beta * math.sin(2.0 * math.pi * (t / max(tau, 1.0)))
        noise = 0.02 * (rng.random() - 0.5)  # tiny noise
        x = (1.0 - alpha) * x + drive + noise
        series.append(x)
    return series

def _count_peaks(xs: List[float]) -> int:
    c = 0
    for i in range(1, len(xs) - 1):
        if xs[i] > xs[i-1] and xs[i] > xs[i+1]:
            c += 1
    return c

def _metrics(xs: List[float]) -> RunMetrics:
    peaks = _count_peaks(xs)
    envelope = max(abs(min(xs)), abs(max(xs))) + 1e-12
    overshoot_count = sum(1 for v in xs if abs(v) > 0.8 * envelope)
    overshoot_rate = overshoot_count / max(len(xs), 1)
    rms = math.sqrt(sum(v*v for v in xs) / max(len(xs), 1))

    # Simple ringing heuristic: multiple peaks + nontrivial overshoot
    ringing = (peaks >= 3) and (overshoot_rate > 0.1)
    return RunMetrics(peaks=peaks, overshoot_rate=overshoot_rate, rms=rms, ringing_detected=ringing)

def run_demo(steps: int, runs: int, seeds: int, out_dir: str,
             alpha: float = 0.08, beta: float = 0.9, tau: float = 40.0) -> Summary:
    """
    Run a small batch of toy oscillators; summarize ringing behavior.
    """
    # In RG you sometimes compare against a stability proxy like Kc ~ (1+τ)/(2α)
    Kc_proxy = (1.0 + tau) / (2.0 * max(alpha, 1e-9))

    metrics: List[RunMetrics] = []
    base_seed = 12345
    for s in range(seeds):
        for r in range(runs):
            xs = _oscillator_series(steps=steps, alpha=alpha, beta=beta, tau=tau,
                                    seed=base_seed + s * 10000 + r)
            m = _metrics(xs)
            metrics.append(m)

    if metrics:
        ringing_fraction = sum(1 for m in metrics if m.ringing_detected) / len(metrics)
        avg_peaks = sum(m.peaks for m in metrics) / len(metrics)
        avg_overshoot = sum(m.overshoot_rate for m in metrics) / len(metrics)
        avg_rms = sum(m.rms for m in metrics) / len(metrics)
    else:
        ringing_fraction = 0.0
        avg_peaks = 0.0
        avg_overshoot = 0.0
        avg_rms = 0.0

    notes = [
        "CI-safe demo; synthetic oscillator approximates ringing behavior.",
        "Tune alpha (damping), beta (drive), tau (period) to shape ringing.",
        "Kc_proxy is a rough stability threshold proxy for dashboards."
    ]

    return Summary(
        steps=steps, runs=runs, seeds=seeds,
        beta=beta, alpha=alpha, tau=tau, Kc_proxy=Kc_proxy,
        ringing_fraction=ringing_fraction,
        avg_peaks=avg_peaks, avg_overshoot=avg_overshoot, avg_rms=avg_rms,
        notes=notes
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CI-safe GP ringing demo")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="results/gp_ringing_demo")
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=40.0)
    args = parser.parse_args()

    # CI guard: clamp heavy workloads when RG_CI is set
    if os.getenv("RG_CI", "").strip() == "1":
        args.steps = min(args.steps, 200)
        args.runs = min(args.runs, 1500)
        args.seeds = min(args.seeds, 1)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_demo(
        steps=args.steps, runs=args.runs, seeds=args.seeds,
        out_dir=str(out_dir),
        alpha=args.alpha, beta=args.beta, tau=args.tau
    )

    out_json = out_dir / "ringing_demo_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary.__dict__, f, indent=2)

    print(f"[ringing_demo] wrote: {out_json} "
          f"(ringing_fraction={summary.ringing_fraction:.3f}, avg_peaks={summary.avg_peaks:.2f})")

if __name__ == "__main__":
    main()
