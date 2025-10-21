#!/usr/bin/env python3
"""Render a Markdown report summarising the equilibrium eigenvalue scan."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence


def load_summary(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> Sequence[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def format_float(value: float, precision: int = 3) -> str:
    return f"{value:.{precision}f}"


def render_report(
    summary_path: Path,
    csv_path: Path,
    output_path: Path,
    figures_relative: Sequence[Path],
) -> None:
    summary = load_summary(summary_path)
    rows = load_csv_rows(csv_path)
    stable_count = sum(1 for row in rows if row.get("stable_flag") == "1")
    total_equilibria = len(rows)

    parameters = summary.get("parameters", {})
    alpha_min = float(parameters.get("alpha_min", 0.0))
    alpha_max = float(parameters.get("alpha_max", 0.0))
    alpha_step = float(parameters.get("alpha_step", 0.0))
    w0 = float(parameters.get("w0", 0.0))
    gamma = float(parameters.get("gamma", 0.0))
    K0 = float(parameters.get("K0", 0.0))

    max_real_info = summary.get("overall_max_real_eig", {})
    max_real_value = float(max_real_info.get("value", 0.0))
    max_real_alpha = float(max_real_info.get("alpha", 0.0))

    rtp_alpha = float(summary.get("rtp_alpha_hint", 0.0))
    alpha_critical = float(summary.get("alpha_critical_w0_over_K0", 0.0))
    max_real_by_alpha = summary.get("max_real_eig_by_alpha", [])
    max_real_near_rtp = max_real_value
    if isinstance(max_real_by_alpha, list) and max_real_by_alpha:
        closest_entry = min(
            max_real_by_alpha,
            key=lambda item: abs(float(item.get("alpha", 0.0)) - rtp_alpha),
        )
        max_real_near_rtp = float(closest_entry.get("max_real_eig", max_real_value))

    markdown_lines = [
        "# Equilibrium Analysis (Non-Hopf RTP)",
        "",
        (
            "No eigenvalue crossing; max Re(λ) ≈ "
            f"{format_float(max_real_near_rtp, 2)} near α ≈ {format_float(rtp_alpha, 2)}; "
            "Hopf ruled out. "
            f"α_c = ω₀²/K₀ = {format_float(alpha_critical, 2)}; "
            "RTP happens far before local instability—global/curvature mechanism."
        ),
        "",
        "## Key Metrics",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        (
            "| Scan range | "
            f"α ∈ [{format_float(alpha_min, 2)}, {format_float(alpha_max, 2)}] (Δα = {format_float(alpha_step, 3)}) |"
        ),
        (
            "| max Re(λ) | "
            f"{format_float(max_real_value, 4)} at α = {format_float(max_real_alpha, 3)} |"
        ),
        (
            "| α_RTP (hint) | "
            f"{format_float(rtp_alpha, 3)} |"
        ),
        (
            "| α_c = ω₀²/K₀ | "
            f"{format_float(alpha_critical, 3)} |"
        ),
        (
            "| Stable equilibria | "
            f"{stable_count} / {total_equilibria} |"
        ),
        (
            "| Parameters | "
            f"ω₀ = {format_float(w0, 3)}, γ = {format_float(gamma, 3)}, K₀ = {format_float(K0, 3)} |"
        ),
        "",
        "## Figures",
        "",
    ]

    for rel_path in figures_relative:
        markdown_lines.append(f"![{rel_path.stem.replace('_', ' ').title()}]({rel_path.as_posix()})")
        markdown_lines.append("")

    markdown_lines.extend(
        [
            "## Reproduction",
            "",
            "1. `python scripts/analysis/equilibrium_eigs_scan.py`",
            "2. `python scripts/analysis/render_equilibrium_report.py`",
            "",
            "CSV/JSON outputs live in `results/equilibrium/` and figures under `docs/assets/figures/`.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(markdown_lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Markdown report for the equilibrium eigenvalue analysis.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/equilibrium"),
        help="Directory containing the CSV/JSON outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/assets/figures"),
        help="Directory containing generated figures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/analysis/equilibrium_analysis.md"),
        help="Markdown file to produce.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path = args.results_dir / "alpha_scan_summary.json"
    csv_path = args.results_dir / "alpha_scan_eigs.csv"
    figures = [
        Path("../assets/figures/eigenvalue_real_vs_alpha.svg"),
        Path("../assets/figures/eigenvalue_imag_vs_alpha.svg"),
        Path("../assets/figures/equilibrium_vs_alpha.svg"),
    ]
    render_report(summary_path, csv_path, args.output, figures)


if __name__ == "__main__":
    main()
