#!/usr/bin/env python3
"""
Pre-registration tool for experiments

Generates a pre-registration document with hypotheses, methods,
and analysis plan before running experiments.
"""

import argparse
from datetime import datetime
import json
import os

def create_preregistration(
    experiment_name,
    hypotheses,
    methods,
    analysis_plan,
    output_dir="docs/notes/preregistrations"
):
    """Create a pre-registration markdown file"""
    
    timestamp = datetime.now().isoformat()
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    template = f"""# Pre-Registration: {experiment_name}

**Date**: {date_str}
**Committed before**: Analyzing any results
**Status**: Pre-registered (locked)

---

## Experiment Overview

{experiment_name}

---

## Hypotheses

{hypotheses}

---

## Methods

{methods}

---

## Analysis Plan

{analysis_plan}

---

## Outcomes Reporting Commitment

We commit to reporting all results (positive, negative, null) and will not
cherry-pick favorable outcomes. This pre-registration is locked and will not
be modified after data collection begins.

**Timestamp**: {timestamp}

---

## Results

*(To be filled after experiment completion)*

---

*Pre-registration tool: tools/preregister_experiment.py*
"""
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{date_str}_{experiment_name.lower().replace(' ', '_')}.md"
    
    with open(filename, 'w') as f:
        f.write(template)
    
    print(f"✓ Pre-registration created: {filename}")
    print(f"✓ Commit this file before running experiments")
    
    return filename

def main():
    parser = argparse.ArgumentParser(description="Create experiment pre-registration")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--hypotheses", required=True, help="Hypotheses (comma-separated or file path)")
    parser.add_argument("--methods", required=True, help="Methods description or file path")
    parser.add_argument("--analysis", required=True, help="Analysis plan or file path")
    parser.add_argument("--output_dir", default="docs/notes/preregistrations", help="Output directory")
    
    args = parser.parse_args()
    
    # Read from files if paths provided
    def read_if_file(text):
        if os.path.exists(text):
            with open(text, 'r') as f:
                return f.read()
        return text
    
    hypotheses = read_if_file(args.hypotheses)
    methods = read_if_file(args.methods)
    analysis = read_if_file(args.analysis)
    
    create_preregistration(
        args.name,
        hypotheses,
        methods,
        analysis,
        args.output_dir
    )

if __name__ == "__main__":
    main()
