#!/usr/bin/env python3
"""
RAIC (Resonance-Aware Inference Controller) Demo

Demonstrates the RAIC controller on synthetic covariance matrices
with varying conditioning and instability patterns.

Usage:
    python controllers/demo_raic.py

Output:
    - Console log of alarm events and temperature adjustments
    - Summary statistics
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from resonance_geometry.controllers.resonance_aware import (
    ResonanceAwareController,
    BatchRAICHarness
)


def generate_synthetic_covariances(
    n_steps: int = 100,
    dim: int = 6,
    instability_interval: tuple = (30, 50),
    seed: int = 42
) -> list:
    """
    Generate synthetic covariance matrices with embedded instability.

    Args:
        n_steps: Number of time steps
        dim: Dimensionality of covariance matrices
        instability_interval: (start, end) indices for high λ_max region
        seed: Random seed

    Returns:
        List of (dim, dim) covariance matrices
    """
    rng = np.random.default_rng(seed)
    covariances = []

    for step in range(n_steps):
        # Base: well-conditioned isotropic covariance
        base_scale = 1.0
        cov = base_scale * np.eye(dim)

        # Add random correlation structure
        A = rng.standard_normal((dim, dim))
        corr_matrix = A @ A.T
        corr_matrix = corr_matrix / np.max(np.linalg.eigvalsh(corr_matrix))
        cov += 0.2 * corr_matrix

        # Inject instability in specified interval
        if instability_interval[0] <= step < instability_interval[1]:
            # Amplify leading eigenvalue (ill-conditioning)
            evals, evecs = np.linalg.eigh(cov)
            evals[-1] *= 3.0 + 0.5 * (step - instability_interval[0])
            cov = evecs @ np.diag(evals) @ evecs.T

        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)
        covariances.append(cov)

    return covariances


def main():
    print("=" * 60)
    print("RAIC (Resonance-Aware Inference Controller) Demo")
    print("=" * 60)
    print()

    # Initialize controller
    controller = ResonanceAwareController(
        window_size=20,
        threshold_sigma=2.0,
        intervention_mode="temperature",
        baseline_temperature=0.7,
        min_temperature=0.3
    )

    print("Controller Configuration:")
    print(f"  Window size: {controller.window_size}")
    print(f"  Threshold: mean + {controller.threshold_sigma}σ")
    print(f"  Baseline temperature: {controller.baseline_temperature}")
    print(f"  Min temperature: {controller.min_temperature}")
    print()

    # Generate synthetic covariance sequence with embedded instability
    print("Generating synthetic covariance sequence...")
    print("  Steps: 100")
    print("  Dimension: 6")
    print("  Instability injected: steps 30-50")
    print()

    covariances = generate_synthetic_covariances(
        n_steps=100,
        dim=6,
        instability_interval=(30, 50),
        seed=42
    )

    # Run RAIC harness
    harness = BatchRAICHarness(controller)
    summary = harness.run_sequence(covariances, reset_between=False)

    # Display results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total steps: {summary['total_steps']}")
    print(f"Alarms triggered: {summary['alarm_count']} ({summary['alarm_rate']*100:.1f}%)")
    print(f"Average temperature: {summary['avg_temperature']:.3f}")
    print(f"Max λ_max observed: {summary['max_lambda_max']:.3f}")
    print()

    # Show sample alarm events
    print("Sample Alarm Events:")
    print("-" * 60)
    log = harness.get_log()
    alarm_events = [entry for entry in log if entry['is_alarmed']]

    if alarm_events:
        for i, event in enumerate(alarm_events[:5]):  # Show first 5 alarms
            print(f"  Step {event['step']:3d}: λ_max={event['lambda_max']:.3f} "
                  f"(threshold={event['alarm_threshold']:.3f}), "
                  f"T={event['temperature']:.3f}")
        if len(alarm_events) > 5:
            print(f"  ... and {len(alarm_events) - 5} more alarms")
    else:
        print("  No alarms triggered")

    print()

    # Stability analysis
    print("Stability Analysis:")
    print("-" * 60)
    stable_steps = [e for e in log if not e['is_alarmed']]
    unstable_steps = [e for e in log if e['is_alarmed']]

    if stable_steps:
        stable_lambdas = [e['lambda_max'] for e in stable_steps]
        print(f"  Stable region (n={len(stable_steps)}):")
        print(f"    λ_max: {np.mean(stable_lambdas):.3f} ± {np.std(stable_lambdas):.3f}")
        print(f"    Temperature: {controller.baseline_temperature:.3f}")

    if unstable_steps:
        unstable_lambdas = [e['lambda_max'] for e in unstable_steps]
        unstable_temps = [e['temperature'] for e in unstable_steps]
        print(f"  Unstable region (n={len(unstable_steps)}):")
        print(f"    λ_max: {np.mean(unstable_lambdas):.3f} ± {np.std(unstable_lambdas):.3f}")
        print(f"    Temperature: {np.mean(unstable_temps):.3f} ± {np.std(unstable_temps):.3f}")

    print()
    print("=" * 60)
    print("Demo complete. See experiments/raic/ for more advanced usage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
