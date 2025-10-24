"""Unit tests asserting eigenvalue sweep claims."""
import json
from pathlib import Path


def _load_max_real(path_json):
    """Load alpha grid and max real eigenvalues from JSON summary."""
    with open(path_json) as f:
        data = json.load(f)
    # supports either {"alpha_grid":[...], "max_real":[...]}
    # or {"alpha":[...], "max_real":[...]}
    alphas = data.get("alpha_grid") or data.get("alpha")
    max_real = data["max_real"]
    # Filter out None values
    pairs = [(a, mr) for a, mr in zip(alphas, max_real) if mr is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_narrow_range_stable():
    """Assert all eigenvalues negative in narrow range (0.25-0.55)."""
    path = Path("docs/analysis/eigs_scan_summary_narrow.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'make sweep-narrow' first."
        )

    alphas, max_real = _load_max_real(path)

    # Check alpha range
    assert all(a >= 0.25 and a <= 0.55 for a in alphas), \
        "Alpha values should be in range [0.25, 0.55]"

    # Check all eigenvalues are negative (stable)
    assert all(mr < 0.0 for mr in max_real), \
        f"All eigenvalues should be negative in narrow range, but max was {max(max_real)}"

    print(f"✓ Narrow range: all {len(max_real)} points stable (max Re(λ) < 0)")


def test_wide_range_crossing():
    """Assert eigenvalue crossing detected in wide range."""
    path = Path("docs/analysis/eigs_scan_summary.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'make sweep-wide' first."
        )

    alphas, max_real = _load_max_real(path)

    # Find positive eigenvalues above α ≈ 0.85
    positive_pairs = [(a, mr) for a, mr in zip(alphas, max_real)
                      if a >= 0.82 and mr > 0.0]

    assert len(positive_pairs) > 0, \
        "Expected positive eigenvalues above α ≈ 0.82, but none found"

    # Find the crossing region
    stable = [(a, mr) for a, mr in zip(alphas, max_real) if mr < 0]
    unstable = [(a, mr) for a, mr in zip(alphas, max_real) if mr > 0]

    if stable and unstable:
        alpha_star_lower = max(a for a, _ in stable if a < min(a for a, _ in unstable))
        alpha_star_upper = min(a for a, _ in unstable if a > max(a for a, _ in stable))
        alpha_star = (alpha_star_lower + alpha_star_upper) / 2
        print(f"✓ Wide range: crossing detected at α* ≈ {alpha_star:.4f}")
        print(f"  Last stable: α = {alpha_star_lower:.4f}")
        print(f"  First unstable: α = {alpha_star_upper:.4f}")

        # Assert crossing is in expected region (0.80 - 0.85)
        assert 0.80 <= alpha_star <= 0.85, \
            f"Expected crossing near 0.832, but found at {alpha_star:.4f}"


def test_zoom_crossing_detail():
    """Assert high-resolution crossing detected in zoom range."""
    path = Path("docs/analysis/zoom/eigs_scan_summary.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'make sweep-zoom' first."
        )

    alphas, max_real = _load_max_real(path)

    # Check alpha range
    assert min(alphas) >= 0.79 and max(alphas) <= 0.87, \
        "Zoom range should cover approximately [0.80, 0.86]"

    # Find crossing
    stable = [(a, mr) for a, mr in zip(alphas, max_real) if mr < 0]
    unstable = [(a, mr) for a, mr in zip(alphas, max_real) if mr > 0]

    assert len(stable) > 0, "Expected stable points in zoom range"
    assert len(unstable) > 0, "Expected unstable points in zoom range"

    alpha_star_lower = max(a for a, _ in stable if a < min(a for a, _ in unstable))
    alpha_star_upper = min(a for a, _ in unstable if a > max(a for a, _ in stable))
    alpha_star = (alpha_star_lower + alpha_star_upper) / 2
    precision = alpha_star_upper - alpha_star_lower

    print(f"✓ Zoom range: high-resolution crossing at α* ≈ {alpha_star:.6f}")
    print(f"  Precision: ±{precision/2:.6f}")

    # Assert precision is better than 0.01 (step size in wide sweep)
    assert precision < 0.01, \
        f"Expected zoom precision better than 0.01, got {precision:.6f}"
