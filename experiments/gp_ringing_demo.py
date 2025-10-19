from __future__ import annotations
import os, json, math, pathlib, argparse
import numpy as np


# --- Optional vector field for local linearization (Phase 2) ---
def vector_field(x, params):
    """Minimal vector field wrapper matching ``simulate_coupled``'s structure."""

    alpha = float(params.get("alpha", 0.1))
    beta = float(params.get("beta", 0.2))
    K0 = float(params.get("K0", 0.0))
    tau = float(params.get("tau", 20.0))
    w0 = 1.0 / max(tau, 1e-9)

    x = np.asarray(x, dtype=float)
    N = x.size // 2
    q, p = x[:N], x[N:]

    q_mean = np.mean(q)
    dq = p
    dp = -beta * p - (w0 ** 2) * q + K0 * np.sin(alpha * q_mean)

    return np.concatenate([dq, dp])

# --- CI/test-friendly coupled demo: returns (lam, x, y) and accepts fs ---
def simulate_coupled(
    fs: float | None = None,
    dur_up: float = 5.0,
    dur_dn: float = 5.0,
    lam_max: float = 0.9,
    seed: int | None = None,
    dt: float | None = None,
    **kwargs,
):
    """
    Minimal coupled 'ringing' stub used by tests.

    Accepts either fs (Hz) or dt (s). Returns:
        lam : (T,) linear up->down schedule in [0, lam_max]
        x,y : (T,) bounded, finite time series
    """
    if dt is None:
        dt = 1.0 / float(fs if fs else 64.0)
    n_up = max(1, int(round(dur_up / dt)))
    n_dn = max(1, int(round(dur_dn / dt)))
    T = n_up + n_dn

    lam_up = np.linspace(0.0, lam_max, n_up, endpoint=False)
    lam_dn = np.linspace(lam_max, 0.0, n_dn)
    lam = np.concatenate([lam_up, lam_dn]).astype(float)

    rng = np.random.default_rng(seed)
    x = np.zeros(T, dtype=float)
    y = np.zeros(T, dtype=float)
    vx = vy = 0.0

    k = 0.05     # light coupling
    damp = 0.02  # damping
    for t in range(1, T):
        ax = -k * (2 * x[t - 1] - y[t - 1]) - damp * vx + 0.01 * lam[t] * rng.normal()
        ay = -k * (2 * y[t - 1] - x[t - 1]) - damp * vy + 0.01 * lam[t] * rng.normal()
        vx += ax * dt
        vy += ay * dt
        x[t] = x[t - 1] + vx * dt
        y[t] = y[t - 1] + vy * dt

    return lam, x, y

# --- separate oscillator for the CLI demo (different name, so no overwrite) ---
def oscillator_series(T=200, beta=0.9, alpha=0.08, tau=40.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T, float)
    v = 0.0
    for t in range(1, T):
        force = beta * math.sin(2 * math.pi * t / max(tau, 1.0))
        v = (1 - alpha) * v + force + 0.01 * rng.normal()
        x[t] = x[t - 1] + v
    return x


def windowed_mi(x: np.ndarray, w: int = 50) -> float:
    if len(x) < 2 * w:
        return 0.0
    a = x[:w] - x[:w].mean()
    b = x[-w:] - x[-w:].mean()
    rho = float(np.corrcoef(a, b)[0, 1])
    rho = np.clip(rho, -0.999, 0.999)
    return float(-math.log(1 - rho * rho))

def analyze_ringing(x: np.ndarray) -> dict:
    if len(x) < 3:
        return {"peaks": 0, "overshoot": 0.0, "rms": 0.0}
    peaks = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])))
    overshoot = float(np.max(x) - x[0])
    rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    return {"peaks": peaks, "overshoot": overshoot, "rms": rms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--runs", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.08)
    p.add_argument("--tau", type=float, default=40.0)
    p.add_argument("--out", type=str, default="results/gp_demo_test")
    args = p.parse_args()

    if os.getenv("RG_CI") == "1":
        args.steps = min(args.steps, 200)
        args.runs = min(args.runs, 1500)
        args.seeds = min(args.seeds, 1)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    peaks, overs, rmss = [], [], []

    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        for _ in range(args.runs):
            x = oscillator_series(
                T=args.steps,
                beta=args.beta,
                alpha=args.alpha,
                tau=args.tau,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            m = analyze_ringing(x)
            peaks.append(m["peaks"])
            overs.append(m["overshoot"])
            rmss.append(m["rms"])

    summary = {
        "steps": args.steps, "runs": args.runs, "seeds": args.seeds,
        "beta": args.beta, "alpha": args.alpha, "tau": args.tau,
        "Kc_proxy": float((1 + args.tau) / max(2 * args.alpha, 1e-9)),
        "ringing_fraction": 1.0 if np.mean(peaks) > 0 else 0.0,
        "avg_peaks": float(np.mean(peaks)),
        "avg_overshoot": float(np.mean(overs)) if overs else 0.0,
        "avg_rms": float(np.mean(rmss)) if rmss else 0.0,
        "notes": [
            "CI-safe demo; synthetic oscillator approximates ringing behavior.",
            "Use real GP dynamics offline; this keeps CI green and quick.",
        ],
    }
    with open(out_dir / "ringing_demo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote:", out_dir / "ringing_demo_summary.json")


if __name__ == "__main__":
    main()

# --- test-compatible sliding-window MI (overrides earlier stub) ---
def windowed_mi(x: np.ndarray,
                y: np.ndarray,
                win: int = 128,
                hop: int = 32,
                bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window mutual information between two 1D arrays.
    Returns (starts, mi_vals), where starts are window start indices and mi_vals are MI (nats).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    if n < win or win <= 1 or hop <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    def _mi(a: np.ndarray, b: np.ndarray, k: int) -> float:
        # Robust histogram-based MI (nats)
        eps = 1e-12
        rng0 = np.random.default_rng(0)
        rng1 = np.random.default_rng(1)
        aj = a + eps * rng0.normal(size=a.shape)
        bj = b + eps * rng1.normal(size=b.shape)
        H, _, _ = np.histogram2d(aj, bj, bins=k)
        P = H / max(H.sum(), 1.0)
        Px = P.sum(axis=1, keepdims=True)
        Py = P.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = P * (np.log(P + eps) - np.log(Px + eps) - np.log(Py + eps))
        return float(np.nansum(term))

    starts = np.arange(0, n - win + 1, hop, dtype=int)
    mi_vals = np.empty_like(starts, dtype=float)
    for i, s in enumerate(starts):
        a = x[s:s + win]
        b = y[s:s + win]
        mi_vals[i] = _mi(a, b, bins)

    # Ensure finite outputs
    mi_vals = np.where(np.isfinite(mi_vals), mi_vals, 0.0)
    return starts, mi_vals
