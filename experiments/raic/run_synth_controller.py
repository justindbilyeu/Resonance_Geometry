#!/usr/bin/env python3
"""
RAIC Synthetic Controller Experiment

Runs batch experiments with RAIC on synthetic covariance matrices
with configurable instability patterns. Generates CSV logs and
SVG visualizations.

Usage:
    python experiments/raic/run_synth_controller.py [--config CONFIG_PATH]

Output:
    - results/raic/raic_demo_log.csv
    - results/raic/raic_demo_summary.json
    - figures/*.svg (if visualization enabled)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from resonance_geometry.controllers.resonance_aware import (
    ResonanceAwareController,
    BatchRAICHarness
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_covariances_with_instability(
    n_steps: int,
    dim: int,
    instability_interval: tuple,
    lambda_amplification: float = 3.0,
    seed: int = 42
) -> list:
    """
    Generate synthetic covariance matrices with controllable instability.

    Args:
        n_steps: Number of time steps
        dim: Matrix dimensionality
        instability_interval: (start, end) for λ_max amplification
        lambda_amplification: Factor to amplify leading eigenvalue
        seed: Random seed

    Returns:
        List of (dim, dim) covariance matrices
    """
    rng = np.random.default_rng(seed)
    covariances = []

    for step in range(n_steps):
        # Base isotropic covariance
        cov = np.eye(dim)

        # Add random correlation
        A = rng.standard_normal((dim, dim))
        corr = A @ A.T
        corr /= np.max(np.linalg.eigvalsh(corr))
        cov += 0.2 * corr

        # Inject instability
        if instability_interval and instability_interval[0] <= step < instability_interval[1]:
            # Amplify leading eigenvalue
            evals, evecs = np.linalg.eigh(cov)
            progress = (step - instability_interval[0]) / (instability_interval[1] - instability_interval[0])
            # Ramp up then down (parabolic)
            amplification = lambda_amplification * (1 - (2*progress - 1)**2)
            evals[-1] *= (1 + amplification)
            cov = evecs @ np.diag(evals) @ evecs.T

        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)
        covariances.append(cov)

    return covariances


def run_scenario(scenario: dict, controller_cfg: dict, synth_cfg: dict) -> dict:
    """
    Run single RAIC scenario.

    Args:
        scenario: Scenario configuration dict
        controller_cfg: Controller parameters
        synth_cfg: Synthetic data parameters

    Returns:
        Summary dict with logs and statistics
    """
    # Initialize controller
    controller = ResonanceAwareController(**controller_cfg)

    # Generate covariances
    instability_interval = scenario.get('instability_interval')
    if instability_interval:
        instability_interval = tuple(instability_interval)

    lambda_amplification = scenario.get('lambda_amplification', 3.0)

    covariances = generate_covariances_with_instability(
        n_steps=synth_cfg['n_steps'],
        dim=synth_cfg['dim'],
        instability_interval=instability_interval,
        lambda_amplification=lambda_amplification,
        seed=synth_cfg['seed']
    )

    # Run batch harness
    harness = BatchRAICHarness(controller)
    summary = harness.run_sequence(covariances, reset_between=False)

    # Add scenario metadata
    summary['scenario_name'] = scenario['name']
    summary['instability_interval'] = instability_interval
    summary['lambda_amplification'] = lambda_amplification

    return summary


def save_log_csv(log: list, output_path: str):
    """Save step-by-step log to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        if not log:
            return

        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)


def save_summary_json(summary: dict, output_path: str):
    """Save summary statistics to JSON."""
    # Remove log (too verbose for JSON summary)
    summary_clean = {k: v for k, v in summary.items() if k != 'log'}

    with open(output_path, 'w') as f:
        json.dump(summary_clean, f, indent=2)


def visualize_results(log: list, output_dir: str, scenario_name: str):
    """
    Generate SVG visualizations of RAIC behavior.

    Args:
        log: Step-by-step log from harness
        output_dir: Directory for output SVGs
        scenario_name: Scenario name for filename
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    # Extract time series
    steps = [entry['step'] for entry in log]
    lambdas = [entry['lambda_max'] for entry in log]
    thresholds = [entry['alarm_threshold'] for entry in log]
    temperatures = [entry['temperature'] for entry in log]
    alarms = [entry['is_alarmed'] for entry in log]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: λ_max and threshold
    ax1.plot(steps, lambdas, label=r'$\lambda_{\mathrm{max}}$', color='blue', linewidth=1.5)
    ax1.plot(steps, thresholds, label='Alarm threshold', color='red', linestyle='--', linewidth=1)
    ax1.fill_between(steps, 0, 1, where=alarms, alpha=0.2, color='red', label='Alarm region')
    ax1.set_ylabel(r'$\lambda_{\mathrm{max}}$', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title(f'RAIC Controller: {scenario_name}', fontsize=12, fontweight='bold')

    # Bottom: Temperature
    ax2.plot(steps, temperatures, label='Temperature', color='green', linewidth=1.5)
    ax2.axhline(0.7, color='gray', linestyle=':', linewidth=1, label='Baseline')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Temperature', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.2, 1.1)

    plt.tight_layout()

    # Save as SVG
    output_path = Path(output_dir) / f'raic_{scenario_name}_trace.svg'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"  Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAIC Synthetic Controller Experiment")
    parser.add_argument(
        '--config',
        default='experiments/raic/configs/raic_demo.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    controller_cfg = config['controller']
    synth_cfg = config['synthetic']
    scenarios = config['scenarios']
    output_cfg = config['output']
    viz_cfg = config.get('visualization', {})

    print(f"Configuration loaded: {len(scenarios)} scenarios")
    print()

    # Create output directory
    results_dir = Path(output_cfg['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all scenarios
    all_summaries = []

    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        print("-" * 60)

        summary = run_scenario(scenario, controller_cfg, synth_cfg)

        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Alarms: {summary['alarm_count']} ({summary['alarm_rate']*100:.1f}%)")
        print(f"  Avg temperature: {summary['avg_temperature']:.3f}")
        print(f"  Max λ_max: {summary['max_lambda_max']:.3f}")

        # Save logs
        log_path = results_dir / f"raic_{scenario['name']}_log.csv"
        save_log_csv(summary['log'], log_path)
        print(f"  Log saved: {log_path}")

        # Save summary
        summary_path = results_dir / f"raic_{scenario['name']}_summary.json"
        save_summary_json(summary, summary_path)
        print(f"  Summary saved: {summary_path}")

        # Visualize
        if viz_cfg.get('plot_lambda_trace'):
            visualize_results(
                summary['log'],
                output_cfg['figures_dir'],
                scenario['name']
            )

        all_summaries.append(summary)
        print()

    # Save combined summary
    combined_path = results_dir / output_cfg['summary_json']
    with open(combined_path, 'w') as f:
        json.dump({
            'scenarios': [
                {k: v for k, v in s.items() if k != 'log'}
                for s in all_summaries
            ]
        }, f, indent=2)

    print("=" * 60)
    print("All scenarios complete.")
    print(f"Combined summary: {combined_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
