# GP ringing demo: coupled stochastic oscillators → windowed MI → alpha-band MI power & hysteresis
# Deps: numpy, scipy, matplotlib

import json, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ---------- MI (histogram) ----------
def hist_mi(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    H, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = H / H.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    logpx = np.log(px, where=px > 0, out=np.full_like(px, -np.inf))
    logpy = np.log(py, where=py > 0, out=np.full_like(py, -np.inf))
    return float(np.sum(pxy[nz] * (np.log(pxy[nz]) - (logpx + logpy)[nz])))

def windowed_mi(x, y, win=256, hop=64, bins=64):
    n = min(len(x), len(y))
    starts = np.arange(0, max(0, n - win + 1), hop, dtype=int)
    vals = np.empty_like(starts, dtype=float)
    for i, s in enumerate(starts):
        segx = x[s:s+win]
        segy = y[s:s+win]
        vals[i] = hist_mi(segx, segy, bins=bins)
    return starts, vals

# ---------- Synthetic generator ----------
def simulate_coupled(fs=128, dur_up=60, dur_dn=60, lam_max=0.9, seed=7):
    rng = np.random.default_rng(seed)
    n_up = fs * dur_up
    n_dn = fs * dur_dn
    lam_up = np.linspace(0.0, lam_max, n_up)
    lam_dn = np.linspace(lam_max, 0.0, n_dn)
    lam = np.concatenate([lam_up, lam_dn])

    # AR(2)-like oscillators around ~10 Hz with mutual coupling
    f0 = 10.0
    r = 0.98
    theta = 2*np.pi*f0/fs
    a1, a2 = 2*r*np.cos(theta), -r**2

    x = np.zeros_like(lam)
    y = np.zeros_like(lam)
    nx = rng.normal(scale=0.5, size=len(lam))
    ny = rng.normal(scale=0.5, size=len(lam))

    for t in range(2, len(lam)):
        c = lam[t]
        x[t] = a1*x[t-1] + a2*x[t-2] + c*y[t-1] + nx[t]
        y[t] = a1*y[t-1] + a2*y[t-2] + c*x[t-1] + ny[t]
    return lam, x, y

# ---------- Analysis ----------
def alpha_power(signal, fs=128, band=(8,12)):
    f, Pxx = welch(signal, fs=fs, nperseg=128, noverlap=64)
    idx = (f >= band[0]) & (f <= band[1])
    return float(np.trapz(Pxx[idx], f[idx]))

def main(outdir="results/gp_demo", show=False):
    os.makedirs(outdir, exist_ok=True)
    fs = 128
    lam, x, y = simulate_coupled(fs=fs, dur_up=60, dur_dn=60, lam_max=0.9, seed=7)

    # Windowed MI
    starts, mi_vals = windowed_mi(x, y, win=256, hop=64, bins=64)
    t = starts / fs  # seconds
    # Alpha-band power of MI(t)
    mi_alpha = alpha_power(mi_vals - np.mean(mi_vals), fs=fs/0.5, band=(8,12))  # hop=0.5s → fs_MI = 2 Hz

    # Up vs down segmentation for simple hysteresis proxy
    mid = np.argmax(lam)
    lam_win = lam[starts + 256//2]  # map each MI point to center λ
    up_mask = np.arange(len(starts)) < np.searchsorted(starts, mid)
    dn_mask = ~up_mask

    # Aggregate by λ bins
    nb = 20
    bins = np.linspace(0, 0.9, nb+1)
    up_means, dn_means, centers = [], [], []
    for i in range(nb):
        m_up = (lam_win >= bins[i]) & (lam_win < bins[i+1]) & up_mask
        m_dn = (lam_win >= bins[i]) & (lam_win < bins[i+1]) & dn_mask
        up_means.append(np.nanmean(mi_vals[m_up]) if np.any(m_up) else np.nan)
        dn_means.append(np.nanmean(mi_vals[m_dn]) if np.any(m_dn) else np.nan)
        centers.append(0.5*(bins[i]+bins[i+1]))

    # ---- Plots ----
    plt.figure(figsize=(10,4))
    plt.plot(t, mi_vals, lw=1.5)
    plt.xlabel("Time (s)"); plt.ylabel("MI (nats)")
    plt.title("Windowed MI(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mi_timeseries.png"), dpi=140)
    if show: plt.show(); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(lam))/fs, lam, lw=1.0, color='gray')
    plt.xlabel("Time (s)"); plt.ylabel("λ (coupling)")
    plt.title("Coupling schedule (up then down)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lambda_schedule.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(centers, up_means, '-o', label='up-sweep')
    plt.plot(centers, dn_means, '-o', label='down-sweep')
    plt.xlabel("λ"); plt.ylabel("Mean MI (nats)")
    plt.title("Hysteresis proxy (synthetic)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hysteresis_curve.png"), dpi=140)
    plt.close()

    # ---- Summary ----
    summary = {
        "alpha_power_MI": mi_alpha,
        "mi_mean_up": np.nanmean(up_means),
        "mi_mean_down": np.nanmean(dn_means),
        "files": ["mi_timeseries.png","lambda_schedule.png","hysteresis_curve.png"]
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote plots + summary to {outdir}")
    print(f"Alpha-band MI power (synthetic): {mi_alpha:.4f}")

if __name__ == "__main__":
    main()
