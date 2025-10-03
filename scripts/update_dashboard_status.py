#!/usr/bin/env python3
“””
Update dashboard status summary from experimental results.
Reads results from results/ and generates docs/data/status/summary.json
“””

import json
import os
from datetime import datetime
from pathlib import Path

def load_experiment_results():
“”“Load all available experiment results.”””
results = {}

```
# Phase sweep
phase_sweep_path = Path("results/phase/phase_sweep.json")
if phase_sweep_path.exists():
    with open(phase_sweep_path) as f:
        results['phase_sweep'] = json.load(f)

# Forbidden region
forbidden_path = Path("results/forbidden/forbidden_summary.json")
if forbidden_path.exists():
    with open(forbidden_path) as f:
        results['forbidden'] = json.load(f)

# Multi-frequency (when available)
multi_freq_path = Path("results/multi_freq/multi_frequency_summary.json")
if multi_freq_path.exists():
    with open(multi_freq_path) as f:
        results['multi_freq'] = json.load(f)

return results
```

def compute_evidence_grade(results):
“””
Compute evidence grade based on completed experiments.

```
Grading:
A: Multiple experiments + null models + replication
B: Multiple experiments + partial validation
C: Single experiment completed
D: Preliminary/incomplete
"""
n_experiments = len(results)

if n_experiments >= 3:
    return "B: Multiple experiments, validation in progress"
elif n_experiments >= 2:
    return "C: Two experiments completed, expanding validation"
elif n_experiments >= 1:
    return "C: Initial experiment completed"
else:
    return "D: Preliminary setup"
```

def compute_validation_metrics(results):
“”“Extract validation metrics from experiment results.”””
metrics = {
“surrogate_controls”: “—”,
“null_models”: “—”,
“replication”: “All experiments use seed 42 (deterministic)”
}

```
# Check for forbidden region results
if 'forbidden' in results:
    forbidden_pct = results['forbidden'].get('forbidden_percent', 0)
    largest_component = results['forbidden'].get('largest_forbidden_component', 0)
    metrics["forbidden_region"] = f"{forbidden_pct:.1f}% forbidden, largest component: {largest_component} cells"

# Check for phase sweep
if 'phase_sweep' in results:
    n_points = len(results['phase_sweep'].get('points', []))
    metrics["phase_map"] = f"{n_points} parameter points sampled"

# Multi-frequency
if 'multi_freq' in results:
    metrics["surrogate_controls"] = "Multi-frequency analysis complete"

return metrics
```

def generate_status_summary(results):
“”“Generate complete status summary.”””

```
# Current timestamp
timestamp = datetime.utcnow().isoformat() + "Z"

# Evidence grade
evidence_grade = compute_evidence_grade(results)

# Validation metrics
validation = compute_validation_metrics(results)

# Count experiments
experiments_completed = list(results.keys())

# Build summary
summary = {
    "last_updated": timestamp,
    "evidence_grade": evidence_grade,
    "experiments_completed": experiments_completed,
    "n_experiments": len(experiments_completed),
    "validation": validation,
    "status": "testing" if len(experiments_completed) > 0 else "setup"
}

return summary
```

def main():
“”“Main entry point.”””

```
# Load results
print("Loading experiment results...")
results = load_experiment_results()

if not results:
    print("WARNING: No experiment results found!")
    print("Run experiments first:")
    print("  python scripts/run_phase_sweep.py --seed 42")
    print("  python experiments/forbidden_region_detector.py --seed 42")
    return

print(f"Found {len(results)} completed experiments:")
for exp in results.keys():
    print(f"  ✓ {exp}")

# Generate summary
summary = generate_status_summary(results)

# Write to docs/data/status/
output_dir = Path("docs/data/status")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "summary.json"
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Status summary written to: {output_path}")
print(f"  Latest run: {summary['last_updated']}")
print(f"  Evidence grade: {summary['evidence_grade']}")
print(f"  Experiments: {', '.join(summary['experiments_completed'])}")
print("\nDeploy with:")
print("  git add docs/data/status/")
print("  git commit -m 'Update dashboard status'")
print("  git push origin main")
```

if **name** == “**main**”:
main()
