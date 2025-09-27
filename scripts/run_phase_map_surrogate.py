#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfftfreq, rfft
from scipy.signal import find_peaks, hilbert, lfilter
from scipy.stats import linregress

import pandas as pd

def psd_peak(signal, fs=1.0):
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return {"peak_freq": 0.0, "gain_db": 0.0}
    signal /= std
    # crop to 99% energy (avoid long tails)
    cum = np.cumsum(signal**2) / np.sum(signal**2)
    idxs = np.argwhere(cum > 0.99)
    last = (idxs[0][0] + 1) if len(idxs) else len(signal)
    signal = signal[:last]
    if len(signal) < 32:
        return {"peak_freq": 0.0, "gain_db": 0.0}
    freqs = rfftfreq(len(signal), 1/fs)
    fft_val = rfft(signal)
    Pxx = np.abs(fft_val)**2 / len(signal)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    # exclude DC
    pk = np.argmax(Pxx_db[1:]) + 1
    # use median baseline for robustness
    baseline = np.median(Pxx_db[1:])
    gain_db = Pxx_db[pk] - baseline
    return {"peak_freq": freqs[pk], "gain_db": gain_db}

def count_overshoots(series, thresh=0.5):
    mean, std = np.mean(series), np.std(series)
    if std == 0: return 0
    z = (series - mean) / std
    pos = len(find_peaks(z, height=thresh)[0])
    neg = len(find_peaks(-z, height=thresh)[0])
    return pos + neg

def estimate_damping_ratio(flux, peak_freq):
    if peak_freq <= 0: return None, None
    cum = np.cumsum(flux**2) / np.sum(flux**2)
    idxs = np.argwhere(cum > 0.99)
    last = (idxs[0][0] + 1) if len(idxs) else len(flux)
    flux = flux[:last]
    env = np.abs(hilbert(flux))
    t = np.arange(len(flux))
    mask = env > 1e-6
    if np.sum(mask) < 5: return None, None
    slope = linregress(t[mask], np.log(env[mask])).slope
    sigma = -slope
    omega_d = 2 * np.pi * peak_freq
    zeta = sigma / np.sqrt(sigma**2 + omega_d**2)
    if not (0 < zeta < 1): return None, None
    return zeta, sigma

def ar2_impulse(alpha, eta, T=150):
    """AR(2) surrogate: r = 1-eta, theta = alpha*pi (calibrates ring boundary)"""
    r = max(0.0, min(0.999, 1.0 - eta))
    theta = np.clip(alpha * np.pi, 1e-3, np.pi-1e-3)
    phi1 = 2 * r * np.cos(theta)
    phi2 = - r**2
    a = [1, -phi1, -phi2]
    b = [1]
    imp = np.zeros(T); imp[0] = 1.0
    return lfilter(b, a, imp)

def main():
    p = argparse.ArgumentParser(description="Surrogate phase map (AR2) for ringing")
    p.add_argument("--alphas", default="0.1,0.4,0.8")
    p.add_argument("--etas",   default="0.02,0.05,0.08")
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--out_dir", default="results/phase_map_surrogate")
    p.add_argument("--psd_db", type=float, default=6.0)
    p.add_argument("--overs", type=int, default=2)
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    etas   = [float(x) for x in args.etas.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(42)

    grid_ring = np.zeros((len(alphas), len(etas)))
    grid_k    = np.zeros((len(alphas), len(etas)))
    rows = []

    for i, alpha in enumerate(alphas):
        for j, eta in enumerate(etas):
            s = ar2_impulse(alpha, eta, T=args.T)
            psd = psd_peak(s)
            gain_db = psd["gain_db"]; fpk = psd["peak_freq"]
            overs = count_overshoots(s)
            zeta, sigma = estimate_damping_ratio(s, fpk)
            is_ring = (gain_db >= args.psd_db) and (overs >= args.overs)

            grid_ring[i, j] = 1 if is_ring else 0
            # Use ζ as a proxy “K_est_surrogate” (monotone with underdamping)
            K_est = zeta if zeta is not None else 0.0
            grid_k[i, j] = K_est

            rows.append({
                "alpha": alpha, "eta": eta,
                "is_ringing": int(is_ring),
                "gain_db": float(gain_db),
                "overshoots": int(overs),
                "K_est_surrogate": float(K_est),
                "tau_geom_surrogate": (1/float(sigma)) if (sigma and sigma>0) else 0.0
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "phase_map.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(
        grid_ring, cmap="RdBu", origin="lower",
        extent=[min(etas), max(etas), min(alphas), max(alphas)], aspect="auto"
    )
    X, Y = np.meshgrid(etas, alphas)
    cs = ax.contour(X, Y, grid_k, levels=[0.3, 0.5, 0.7, 0.9], colors="k", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="ζ=%.1f")
    ax.set_xlabel("η (learning rate)")
    ax.set_ylabel("α (EMA memory)")
    ax.set_title("Surrogate Ringing Map (AR(2))")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Ringing (1) vs Smooth (0)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "phase_map.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
