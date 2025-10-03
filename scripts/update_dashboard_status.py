#!/usr/bin/env python3
"""
Update dashboard status summary from experimental results.
"""

import json
from datetime import datetime
from pathlib import Path

def load_experiment_results():
    results = {}
    phase_sweep_path = Path("results/phase/phase_sweep.json")
    if phase_sweep_path.exists():
        with open(phase_sweep_path) as f:
            results["phase_sweep"] = json.load(f)

    forbidden_path = Path("results/forbidden/forbidden_summary.json")
    if forbidden_path.exists():
        with open(forbidden_path) as f:
            results["forbidden"] = json.load(f)

    return results

def compute_evidence_grade(results):
    n = len(results)
    if n >= 2:
        return "C: Two experiments completed, expanding validation"
    elif n >= 1:
        return "C: Initial experiment completed"
    else:
        return "D: Preliminary setup"

def generate_status_summary(results):
    timestamp = datetime.utcnow().isoformat() + "Z"
    return {
        "last_updated": timestamp,
        "evidence_grade": compute_evidence_grade(results),
        "experiments_completed": list(results.keys()),
        "n_experiments": len(results),
        "status": "testing" if results else "setup"
    }

def main():
    print("Loading experiment results...")
    results = load_experiment_results()

    if not results:
        print("WARNING: No results found!")
        return

    print(f"Found {len(results)} completed experiments")
    summary = generate_status_summary(results)

    output_dir = Path("docs/data/status")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Status written to: {output_path}")
    print(f"Evidence grade: {summary['evidence_grade']}")

if __name__ == "__main__":
    main()