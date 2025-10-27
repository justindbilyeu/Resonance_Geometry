#!/usr/bin/env python3
"""
Compute Curvature Metrics from Activations

Computes κ(Σ), λ_max, and related spectral stability metrics from
extracted activations.

Usage:
    python rg_empirical/compute_curvature_metrics.py --activations results/activations/activations_gpt2_truthfulqa.npz

Output:
    - curvature_metrics_{model}_{dataset}.csv: Per-sample metrics
    - summary_stats_{model}_{dataset}.json: Aggregate statistics
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import csv


def compute_covariance_metrics(activations: np.ndarray, epsilon: float = 1e-12) -> dict:
    """
    Compute spectral metrics from activation matrix.

    Args:
        activations: (n_samples, hidden_dim) activation matrix
        epsilon: Regularization for numerical stability

    Returns:
        Dict with kappa, lambda_max, lambda_min, trace, log_det
    """
    # Compute covariance matrix
    # Center the data first
    activations_centered = activations - np.mean(activations, axis=0, keepdims=True)
    cov = np.cov(activations_centered, rowvar=False, bias=True)

    # Regularize
    cov_reg = cov + epsilon * np.eye(cov.shape[0])

    # Eigenvalues
    try:
        evals = np.linalg.eigvalsh(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback: use eig (slower but more robust)
        evals, _ = np.linalg.eig(cov_reg)
        evals = np.real(evals)

    evals = np.sort(evals)[::-1]  # Descending order

    lambda_max = float(evals[0])
    lambda_min = float(max(evals[-1], epsilon))
    kappa = lambda_max / lambda_min

    trace_cov = float(np.trace(cov_reg))

    # Log determinant (for stability)
    sign, log_det = np.linalg.slogdet(cov_reg)
    log_det = float(log_det) if sign > 0 else -np.inf

    # Effective rank (participation ratio)
    evals_normalized = evals / (np.sum(evals) + 1e-12)
    eff_rank = float(1.0 / np.sum(evals_normalized**2))

    return {
        'kappa': kappa,
        'lambda_max': lambda_max,
        'lambda_min': lambda_min,
        'trace': trace_cov,
        'log_det': log_det,
        'eff_rank': eff_rank,
        'dim': cov.shape[0]
    }


def compute_metrics_per_layer(activations_dict: dict, epsilon: float = 1e-12) -> dict:
    """
    Compute metrics for all layers.

    Args:
        activations_dict: Dict mapping layer_idx -> (n_samples, hidden_dim)
        epsilon: Regularization

    Returns:
        Dict mapping layer_idx -> metrics dict
    """
    layer_metrics = {}

    for layer_name, acts in activations_dict.items():
        if layer_name.startswith('layer_'):
            layer_idx = int(layer_name.split('_')[1])
            metrics = compute_covariance_metrics(acts, epsilon=epsilon)
            layer_metrics[layer_idx] = metrics

    return layer_metrics


def compute_per_sample_metrics(
    activations: np.ndarray,
    window_size: int = 10,
    epsilon: float = 1e-12
) -> list:
    """
    Compute rolling metrics per sample (for sequential data).

    Args:
        activations: (n_samples, hidden_dim)
        window_size: Rolling window size
        epsilon: Regularization

    Returns:
        List of metric dicts per sample
    """
    n_samples = activations.shape[0]
    per_sample_metrics = []

    for i in range(n_samples):
        # Use rolling window ending at current sample
        start_idx = max(0, i - window_size + 1)
        window_acts = activations[start_idx:i+1, :]

        if window_acts.shape[0] < 3:
            # Not enough samples yet
            per_sample_metrics.append({
                'sample_idx': i,
                'kappa': np.nan,
                'lambda_max': np.nan,
                'lambda_min': np.nan
            })
        else:
            metrics = compute_covariance_metrics(window_acts, epsilon=epsilon)
            metrics['sample_idx'] = i
            per_sample_metrics.append(metrics)

    return per_sample_metrics


def save_metrics_csv(metrics_list: list, output_path: str):
    """Save per-sample metrics to CSV."""
    if not metrics_list:
        return

    with open(output_path, 'w', newline='') as f:
        fieldnames = metrics_list[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)


def main():
    parser = argparse.ArgumentParser(description="Compute curvature metrics from activations")
    parser.add_argument('--activations', required=True, help='Path to activations NPZ file')
    parser.add_argument('--output', default='results/curvature/', help='Output directory')
    parser.add_argument('--epsilon', type=float, default=1e-12, help='Regularization epsilon')
    parser.add_argument('--window_size', type=int, default=10, help='Rolling window for per-sample metrics')
    args = parser.parse_args()

    print("=" * 60)
    print("Curvature Metrics Computation")
    print("=" * 60)
    print(f"Activations: {args.activations}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Window size: {args.window_size}")
    print()

    # Load activations
    print("Loading activations...")
    activations_dict = dict(np.load(args.activations))
    print(f"Loaded {len(activations_dict)} layers")

    # Compute per-layer metrics (global)
    print()
    print("Computing per-layer metrics...")
    layer_metrics = compute_metrics_per_layer(activations_dict, epsilon=args.epsilon)

    for layer_idx in sorted(layer_metrics.keys()):
        metrics = layer_metrics[layer_idx]
        print(f"  Layer {layer_idx:2d}: κ={metrics['kappa']:.2f}, λ_max={metrics['lambda_max']:.3f}, "
              f"eff_rank={metrics['eff_rank']:.1f}/{metrics['dim']}")

    # Compute per-sample metrics for middle layer
    print()
    print("Computing per-sample metrics (middle layer)...")
    middle_layer_idx = sorted(layer_metrics.keys())[len(layer_metrics)//2]
    middle_layer_name = f"layer_{middle_layer_idx}"
    middle_acts = activations_dict[middle_layer_name]

    per_sample_metrics = compute_per_sample_metrics(
        middle_acts,
        window_size=args.window_size,
        epsilon=args.epsilon
    )

    print(f"  Computed metrics for {len(per_sample_metrics)} samples")

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract base name
    base_name = Path(args.activations).stem.replace('activations_', '')

    # Save per-layer summary
    summary_path = output_dir / f"summary_stats_{base_name}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'per_layer': {
                f'layer_{idx}': metrics
                for idx, metrics in layer_metrics.items()
            },
            'global': {
                'mean_kappa': float(np.mean([m['kappa'] for m in layer_metrics.values()])),
                'max_kappa': float(np.max([m['kappa'] for m in layer_metrics.values()])),
                'mean_lambda_max': float(np.mean([m['lambda_max'] for m in layer_metrics.values()]))
            }
        }, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Save per-sample CSV
    csv_path = output_dir / f"curvature_metrics_{base_name}.csv"
    save_metrics_csv(per_sample_metrics, csv_path)
    print(f"Per-sample metrics saved: {csv_path}")

    print()
    print("=" * 60)
    print("Computation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
