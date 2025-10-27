#!/usr/bin/env python3
"""
Evaluate λ_max as Hallucination Detector on TruthfulQA

Tests the hypothesis that λ_max > threshold correlates with hallucination
via ROC-AUC analysis.

Usage:
    python rg_empirical/eval_truthfulqa_lambda.py \
        --metrics results/curvature/curvature_metrics_gpt2_truthfulqa.csv \
        --labels results/truthfulqa_labels.json

Output:
    - roc_curve_lambda_max.svg: ROC curve visualization
    - eval_results.json: AUC, threshold, confusion matrix
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import csv


def load_metrics_csv(csv_path: str) -> list:
    """Load per-sample curvature metrics from CSV."""
    metrics = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({
                'sample_idx': int(row['sample_idx']),
                'lambda_max': float(row['lambda_max']) if row['lambda_max'] != 'nan' else np.nan,
                'kappa': float(row['kappa']) if row['kappa'] != 'nan' else np.nan
            })
    return metrics


def load_hallucination_labels(labels_path: str) -> dict:
    """
    Load ground truth hallucination labels.

    Returns:
        Dict mapping sample_idx -> label (1=hallucination, 0=truthful)
    """
    with open(labels_path, 'r') as f:
        data = json.load(f)

    # Convert to dict for easy lookup
    return {item['sample_idx']: item['is_hallucination'] for item in data}


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute ROC-AUC and optimal threshold.

    Args:
        scores: Predicted scores (higher = more likely hallucination)
        labels: Ground truth binary labels (1=hallucination, 0=truthful)

    Returns:
        (auc, fpr_array, tpr_array, thresholds, optimal_threshold)
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Compute TPR and FPR at different thresholds
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)

    if n_positive == 0 or n_negative == 0:
        # Degenerate case
        return 0.5, np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), 0.0

    tpr_list = []
    fpr_list = []
    threshold_list = []

    for i, threshold in enumerate(sorted_scores):
        # Predict positive for all scores >= threshold
        predictions = (scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        threshold_list.append(threshold)

    # Add endpoints
    tpr_array = np.array([0.0] + tpr_list + [1.0])
    fpr_array = np.array([0.0] + fpr_list + [1.0])
    thresholds = np.array([np.inf] + threshold_list + [-np.inf])

    # Compute AUC via trapezoidal rule
    auc = np.trapz(tpr_array, fpr_array)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr_array - fpr_array
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return auc, fpr_array, tpr_array, thresholds, optimal_threshold


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, output_path: str):
    """Generate ROC curve as SVG."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ROC
    ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'λ_max (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve: λ_max as Hallucination Detector', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate λ_max as hallucination detector")
    parser.add_argument('--metrics', required=True, help='Path to curvature metrics CSV')
    parser.add_argument('--labels', help='Path to hallucination labels JSON (if available)')
    parser.add_argument('--output', default='results/eval/', help='Output directory')
    parser.add_argument('--mock_labels', action='store_true', help='Generate mock labels for demo')
    args = parser.parse_args()

    print("=" * 60)
    print("TruthfulQA λ_max Evaluation")
    print("=" * 60)
    print(f"Metrics: {args.metrics}")
    print(f"Labels: {args.labels or 'MOCK'}")
    print()

    # Load metrics
    print("Loading metrics...")
    metrics = load_metrics_csv(args.metrics)
    print(f"Loaded metrics for {len(metrics)} samples")

    # Filter out NaN
    valid_metrics = [m for m in metrics if not np.isnan(m['lambda_max'])]
    print(f"Valid (non-NaN) samples: {len(valid_metrics)}")

    # Load or generate labels
    if args.labels and not args.mock_labels:
        print("Loading hallucination labels...")
        labels_dict = load_hallucination_labels(args.labels)
    else:
        print("Generating mock hallucination labels...")
        # Mock: high λ_max → more likely hallucination
        labels_dict = {}
        lambda_values = [m['lambda_max'] for m in valid_metrics]
        median_lambda = np.median(lambda_values)
        for m in valid_metrics:
            # Stochastic labeling based on λ_max
            prob_hallucination = 0.3 + 0.4 * (m['lambda_max'] > median_lambda)
            labels_dict[m['sample_idx']] = int(np.random.rand() < prob_hallucination)

    # Align metrics and labels
    aligned_data = []
    for m in valid_metrics:
        if m['sample_idx'] in labels_dict:
            aligned_data.append({
                'lambda_max': m['lambda_max'],
                'kappa': m['kappa'],
                'label': labels_dict[m['sample_idx']]
            })

    print(f"Aligned samples: {len(aligned_data)}")

    if len(aligned_data) < 10:
        print("Error: Not enough aligned samples for ROC analysis")
        return

    # Extract arrays
    lambda_max_scores = np.array([d['lambda_max'] for d in aligned_data])
    labels = np.array([d['label'] for d in aligned_data])

    print(f"Hallucinations: {np.sum(labels)} / {len(labels)} ({np.mean(labels)*100:.1f}%)")

    # Compute ROC-AUC
    print()
    print("Computing ROC-AUC...")
    auc, fpr, tpr, thresholds, optimal_threshold = compute_roc_auc(lambda_max_scores, labels)

    print(f"  AUC: {auc:.3f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")

    # Confusion matrix at optimal threshold
    predictions = (lambda_max_scores >= optimal_threshold).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    print()
    print("Confusion Matrix (at optimal threshold):")
    print(f"  TP: {tp:3d}  FP: {fp:3d}")
    print(f"  FN: {fn:3d}  TN: {tn:3d}")
    print(f"  Precision: {tp/(tp+fp):.3f}" if (tp+fp) > 0 else "  Precision: N/A")
    print(f"  Recall:    {tp/(tp+fn):.3f}" if (tp+fn) > 0 else "  Recall: N/A")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation results
    results = {
        'auc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'n_samples': len(aligned_data),
        'n_hallucinations': int(np.sum(labels)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else None,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else None
    }

    results_path = output_dir / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # Plot ROC curve
    plot_roc_curve(fpr, tpr, auc, output_dir / 'roc_curve_lambda_max.svg')

    print()
    print("=" * 60)
    print("Evaluation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
