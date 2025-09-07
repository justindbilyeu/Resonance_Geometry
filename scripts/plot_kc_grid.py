#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pivot_err(df, err_col):
    df = df.copy()
    df['A_over_B'] = df['A'] / df['B']
    piv = df.pivot_table(values=err_col, index='A_over_B', columns='Δ', aggfunc=np.mean)
    piv = piv.sort_index().sort_index(axis=1)
    return piv

def heatmap(ax, data, title, cmap='magma'):
    im = ax.imshow(data.values, origin='lower', aspect='auto',
                   extent=[data.columns.min(), data.columns.max(),
                           data.index.min(), data.index.max()],
                   cmap=cmap)
    ax.set_xlabel('Δ')
    ax.set_ylabel('A / B')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='relative error vs Kc_ER')

def parity(ax, x, y, label_x, label_y, title):
    ax.scatter(x, y, s=20, alpha=0.8)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(title)
    ax.grid(alpha=0.3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/kc_rule/Kc_comparison_grid.csv')
    ap.add_argument('--outdir', default='results/kc_rule')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    err_rh = pivot_err(df, 'err_RH')
    err_ds = pivot_err(df, 'err_DS')
    heatmap(axes[0], err_rh, 'RH vs ER — relative error')
    heatmap(axes[1], err_ds, 'DS vs ER — relative error')
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'kc_error_heatmaps.png'), dpi=200)

    # Parity plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    parity(axes[0], df['Kc_ER'].values, df['Kc_RH'].values, 'Kc_ER', 'Kc_RH',
           'Parity: RH vs ER')
    parity(axes[1], df['Kc_ER'].values, df['Kc_DS'].values, 'Kc_ER', 'Kc_DS',
           'Parity: DS vs ER')
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'kc_parity_plots.png'), dpi=200)

    # Summary print
    print('== Summary vs Kc_ER ==')
    for name, col in [('RH', 'err_RH'), ('DS', 'err_DS')]:
        e = df[col].values
        print(f'{name}: mean={np.mean(e):.4f}, median={np.median(e):.4f}, max={np.max(e):.4f}')

if __name__ == '__main__':
    main()
