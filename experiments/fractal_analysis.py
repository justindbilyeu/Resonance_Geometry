#!/usr/bin/env python3
“””
Fractal dimension analysis of phase boundaries.
Uses box-counting on stable/unstable transition regions.
“””

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Tuple, List, Dict, Any

def load_phase_data(phase_file: Path) -> np.ndarray:
“”“Load phase sweep results and extract boundary points.”””
with open(phase_file) as f:
data = json.load(f)

```
points = data.get('points', [])
if not points:
    raise ValueError(f"No points found in {phase_file}")

# Extract (alpha, eta, status) tuples
boundary_points = []
for pt in points:
    alpha = pt.get('alpha', pt.get('params', {}).get('alpha'))
    eta = pt.get('eta', pt.get('params', {}).get('eta'))
    
    # Classify as unstable if ringing detected
    is_ringing = pt.get('ringing', False) or pt.get('psd_peak', 0) > 6
    
    if alpha is not None and eta is not None:
        boundary_points.append([alpha, eta, 1 if is_ringing else 0])

return np.array(boundary_points)
```

def extract_boundary(points: np.ndarray, threshold: float = 0.5) -> np.ndarray:
“”“Extract points near the stable/unstable boundary.”””
# Grid the space and find transition points
alpha_range = np.unique(points[:, 0])
eta_range = np.unique(points[:, 1])

```
boundary = []

# For each alpha, find eta where transition occurs
for alpha in alpha_range:
    alpha_points = points[points[:, 0] == alpha]
    if len(alpha_points) < 2:
        continue
        
    # Sort by eta
    alpha_points = alpha_points[alpha_points[:, 1].argsort()]
    
    # Find transition
    for i in range(len(alpha_points) - 1):
        if alpha_points[i, 2] != alpha_points[i+1, 2]:
            # Boundary point between i and i+1
            boundary.append([
                alpha,
                (alpha_points[i, 1] + alpha_points[i+1, 1]) / 2
            ])

return np.array(boundary) if boundary else points[:, :2]
```

def box_counting(points: np.ndarray, epsilon: float) -> int:
“”“Count number of boxes of size epsilon that contain boundary points.”””
if len(points) == 0:
return 0

```
# Normalize to [0, 1]
min_vals = points.min(axis=0)
max_vals = points.max(axis=0)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1  # Avoid division by zero

normalized = (points - min_vals) / range_vals

# Count boxes
boxes = set()
for point in normalized:
    box_idx = tuple((point / epsilon).astype(int))
    boxes.add(box_idx)

return len(boxes)
```

def compute_fractal_dimension(
boundary: np.ndarray,
epsilon_range: Tuple[float, float] = (0.01, 0.5),
n_scales: int = 20
) -> Dict[str, Any]:
“”“Compute fractal dimension via box-counting.”””

```
# Generate epsilon values (box sizes)
epsilons = np.logspace(
    np.log10(epsilon_range[0]),
    np.log10(epsilon_range[1]),
    n_scales
)

# Count boxes at each scale
counts = [box_counting(boundary, eps) for eps in epsilons]

# Filter out zeros
valid = [(e, c) for e, c in zip(epsilons, counts) if c > 0]
if len(valid) < 3:
    return {
        "H": 0.0,
        "R2": 0.0,
        "log_inv_eps": [],
        "log_counts": [],
        "fit": {"slope": 0.0, "intercept": 0.0},
        "CI": [0.0, 0.0]
    }

epsilons, counts = zip(*valid)

# Log-log regression: log(N) = -H * log(eps) + c
# So we fit against log(1/eps)
x = np.log(1 / np.array(epsilons))
y = np.log(counts)

slope, intercept, r_value, _, std_err = stats.linregress(x, y)

# Confidence interval (95%)
ci_half = 1.96 * std_err

return {
    "H": float(slope),  # Hausdorff dimension estimate
    "R2": float(r_value**2),
    "log_inv_eps": x.tolist(),
    "log_counts": y.tolist(),
    "fit": {
        "slope": float(slope),
        "intercept": float(intercept)
    },
    "CI": [float(slope - ci_half), float(slope + ci_half)]
}
```

def main():
parser = argparse.ArgumentParser(
description=“Fractal dimension analysis of phase boundaries”
)
parser.add_argument(
“–phase-file”,
type=Path,
default=Path(“results/phase/phase_sweep.json”),
help=“Phase sweep results JSON”
)
parser.add_argument(
“–epsilon-min”,
type=float,
default=0.01,
help=“Minimum box size”
)
parser.add_argument(
“–epsilon-max”,
type=float,
default=0.5,
help=“Maximum box size”
)
parser.add_argument(
“–n-scales”,
type=int,
default=20,
help=“Number of scales to sample”
)
parser.add_argument(
“–out-dir”,
type=Path,
default=Path(“results/fractal”),
help=“Output directory”
)

```
args = parser.parse_args()

# Load and process
print(f"Loading phase data from {args.phase_file}...")
points = load_phase_data(args.phase_file)
print(f"Loaded {len(points)} phase points")

print("Extracting boundary...")
boundary = extract_boundary(points)
print(f"Found {len(boundary)} boundary points")

if len(boundary) < 3:
    print("WARNING: Not enough boundary points for fractal analysis")
    result = {
        "H": 0.0,
        "R2": 0.0,
        "log_inv_eps": [],
        "log_counts": [],
        "fit": {"slope": 0.0, "intercept": 0.0},
        "CI": [0.0, 0.0],
        "warning": "Insufficient boundary points"
    }
else:
    print("Computing fractal dimension...")
    result = compute_fractal_dimension(
        boundary,
        epsilon_range=(args.epsilon_min, args.epsilon_max),
        n_scales=args.n_scales
    )
    
    print(f"\nResults:")
    print(f"  Hausdorff dimension: {result['H']:.3f}")
    print(f"  95% CI: [{result['CI'][0]:.3f}, {result['CI'][1]:.3f}]")
    print(f"  R²: {result['R2']:.3f}")

# Save results
args.out_dir.mkdir(parents=True, exist_ok=True)

output_file = args.out_dir / "fractal_summary.json"
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {output_file}")

# Also save to docs/data/latest for dashboard
latest_dir = Path("docs/data/latest")
latest_dir.mkdir(parents=True, exist_ok=True)

latest_file = latest_dir / "fractal.json"
with open(latest_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Dashboard data saved to {latest_file}")
```

if **name** == “**main**”:
main()
