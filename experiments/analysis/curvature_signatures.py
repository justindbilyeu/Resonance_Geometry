"""Compute Ollivier--Ricci curvature signatures near forbidden zones.

This script aggregates curvature statistics for graph couplings extracted
from simulation snapshots.  It reads one or more coupling matrices ``g``
(with symmetric weights) and builds graphs by thresholding entries at the
matrix median.  For each resulting graph it computes the Ollivier--Ricci
curvature for every edge using a lightweight Earth Mover's Distance solver.
Curvature values are then aggregated to per-cell (node) means and gradient
magnitudes across neighbors.  A JSON summary report and histogram figure are
written to ``outputs/topo/curvature_report.json`` and
``figures/topo/curvature_hist.png`` respectively.

Examples
--------
::

    python experiments/analysis/curvature_signatures.py \
        --input results/couplings --boundary-mask results/boundary.npy

The script is intentionally defensive: if insufficient data is supplied,
meaningful defaults are used and the generated report will indicate the
coverage achieved.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute curvature statistics for coupling matrices",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to a .npy/.npz file or a directory containing coupling"
            " matrices.  All files named '*.npy' or '*.npz' inside the"
            " directory will be processed."
        ),
    )
    parser.add_argument(
        "--boundary-mask",
        help=(
            "Optional path to a boolean .npy/.npz file with shape (N,) or (N,1)"
            " marking cells within 2x grid spacing of forbidden regions."
        ),
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=1.0,
        help="Grid spacing of the lattice (used for logging only).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/topo",
        help="Directory where JSON summaries will be written.",
    )
    parser.add_argument(
        "--figure-dir",
        default="figures/topo",
        help="Directory where figures will be stored.",
    )
    parser.add_argument(
        "--median-symmetrize",
        action="store_true",
        help="If set, symmetrize coupling matrices before thresholding.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "Stay probability used for the Ollivier--Ricci measures."
            " 0 corresponds to uniform distribution over neighbors only,"
            " while values >0 allocate that probability mass to the node"
            " itself."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def collect_coupling_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            f
            for f in path.iterdir()
            if f.suffix.lower() in {".npy", ".npz"} and f.is_file()
        )
        return files
    raise FileNotFoundError(f"Input path '{path}' does not exist")


def load_matrix(file_path: Path, key: str = "g") -> np.ndarray:
    if file_path.suffix.lower() == ".npy":
        arr = np.load(file_path)
    else:
        with np.load(file_path) as data:
            if key in data:
                arr = data[key]
            else:
                if len(data.files) != 1:
                    raise KeyError(
                        f"Unable to determine data array in {file_path}; "
                        "provide an archive with a single array or include 'g'."
                    )
                arr = data[data.files[0]]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Coupling matrix from {file_path} must be square; got {arr.shape}")
    return arr.astype(float)


def load_boundary_mask(path: Path, size: int | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        with np.load(path) as data:
            if "mask" in data:
                mask = data["mask"]
            else:
                mask = data[data.files[0]]
    mask = np.asarray(mask).astype(bool).reshape(-1)
    if size is not None and mask.size != size:
        raise ValueError(
            f"Boundary mask size {mask.size} does not match coupling dimension {size}."
        )
    return mask


def build_graph_from_coupling(g: np.ndarray, symmetrize: bool = False) -> Dict[int, set[int]]:
    if symmetrize:
        g = 0.5 * (g + g.T)
    threshold = float(np.median(g))
    n = g.shape[0]
    adjacency: Dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if g[i, j] > threshold or g[j, i] > threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
    LOGGER.debug("Constructed graph with %d nodes and %d edges", n, sum(len(v) for v in adjacency.values()) // 2)
    return adjacency


def compute_shortest_paths(adjacency: Mapping[int, set[int]]) -> Dict[int, Dict[int, int]]:
    distances: Dict[int, Dict[int, int]] = {}
    for src in adjacency:
        dist: Dict[int, int] = {src: 0}
        frontier = [src]
        while frontier:
            current = frontier.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    frontier.append(neighbor)
        distances[src] = dist
    return distances


def node_distribution(
    node: int,
    adjacency: Mapping[int, set[int]],
    alpha: float = 0.0,
) -> Tuple[List[int], np.ndarray]:
    neighbors = sorted(adjacency[node])
    support = [node] + neighbors
    if not neighbors and alpha <= 0.0:
        # Degenerate isolated node; assign all mass to itself
        return [node], np.array([1.0])
    mass = np.ones(len(support))
    if alpha > 0.0:
        mass[0] = alpha
        if len(support) > 1:
            mass[1:] = (1.0 - alpha) / (len(support) - 1)
        else:
            mass[0] = 1.0
    else:
        mass = mass / mass.sum()
    return support, mass


def earth_mover_distance(
    support_a: Sequence[int],
    support_b: Sequence[int],
    mass_a: np.ndarray,
    mass_b: np.ndarray,
    distances: Mapping[int, Mapping[int, int]],
) -> float:
    m, n = len(support_a), len(support_b)
    cost = np.zeros((m, n), dtype=float)
    for i, node_a in enumerate(support_a):
        for j, node_b in enumerate(support_b):
            if node_b in distances.get(node_a, {}):
                cost[i, j] = distances[node_a][node_b]
            elif node_a in distances.get(node_b, {}):
                cost[i, j] = distances[node_b][node_a]
            else:
                cost[i, j] = math.inf
    if not np.isfinite(cost).all():
        # If supports are disconnected, fallback to large penalty
        cost[~np.isfinite(cost)] = cost[np.isfinite(cost)].max(initial=1.0) * 2
    cvec = cost.reshape(-1)
    a_eq = []
    b_eq = []
    for i in range(m):
        row = np.zeros(m * n)
        row[i * n : (i + 1) * n] = 1.0
        a_eq.append(row)
        b_eq.append(mass_a[i])
    for j in range(n):
        row = np.zeros(m * n)
        row[j::n] = 1.0
        a_eq.append(row)
        b_eq.append(mass_b[j])
    bounds = [(0.0, None)] * (m * n)
    result = linprog(
        cvec,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        LOGGER.warning("linprog failed to converge: %s", result.message)
        return float("nan")
    return float(result.fun)


def ollivier_ricci_edge_curvature(
    node_u: int,
    node_v: int,
    adjacency: Mapping[int, set[int]],
    distances: Mapping[int, Mapping[int, int]],
    alpha: float = 0.0,
) -> float:
    support_u, mass_u = node_distribution(node_u, adjacency, alpha)
    support_v, mass_v = node_distribution(node_v, adjacency, alpha)
    w1 = earth_mover_distance(support_u, support_v, mass_u, mass_v, distances)
    if not np.isfinite(w1):
        return float("nan")
    return 1.0 - w1  # edge length is 1 in the unweighted graph


def aggregate_curvatures(
    adjacency: Mapping[int, set[int]],
    distances: Mapping[int, Mapping[int, int]],
    alpha: float = 0.0,
) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, np.ndarray, np.ndarray]:
    edge_curvature: Dict[Tuple[int, int], float] = {}
    n = len(adjacency)
    node_curvatures = np.full(n, np.nan)
    for u in adjacency:
        curvatures: List[float] = []
        for v in adjacency[u]:
            key = (min(u, v), max(u, v))
            if key not in edge_curvature:
                edge_curvature[key] = ollivier_ricci_edge_curvature(u, v, adjacency, distances, alpha)
            curv = edge_curvature[key]
            if np.isfinite(curv):
                curvatures.append(curv)
        if curvatures:
            node_curvatures[u] = float(np.mean(curvatures))
    gradient_magnitude = np.full(n, np.nan)
    gradient_direction = np.full(n, np.nan)
    for u in adjacency:
        if not np.isfinite(node_curvatures[u]):
            continue
        diffs: List[float] = []
        for v in adjacency[u]:
            v_curv = node_curvatures[v]
            if np.isfinite(v_curv):
                diffs.append(v_curv - node_curvatures[u])
        if diffs:
            gradient_magnitude[u] = float(np.mean(np.abs(diffs)))
            gradient_direction[u] = float(np.mean(diffs))
    return edge_curvature, node_curvatures, gradient_magnitude, gradient_direction


def summarize_statistics(
    node_curvatures: np.ndarray,
    gradient_magnitude: np.ndarray,
    gradient_direction: np.ndarray,
    boundary_mask: np.ndarray,
    kappa_threshold: float = -0.1,
    gradient_threshold: float = 0.05,
) -> Dict[str, object]:
    valid_nodes = np.isfinite(node_curvatures)
    boundary_mask = boundary_mask & valid_nodes
    boundary_count = int(np.count_nonzero(boundary_mask))
    if boundary_count == 0:
        kappa_pass_fraction = 0.0
        gradient_pass_fraction = 0.0
        mean_gradient = float(np.nanmean(gradient_magnitude)) if np.any(np.isfinite(gradient_magnitude)) else float("nan")
        mean_direction = float(np.nanmean(gradient_direction)) if np.any(np.isfinite(gradient_direction)) else float("nan")
        return {
            "boundary_cell_count": boundary_count,
            "kappa_pass_fraction": kappa_pass_fraction,
            "gradient_pass_fraction": gradient_pass_fraction,
            "kappa_threshold": kappa_threshold,
            "gradient_threshold": gradient_threshold,
            "pass": False,
            "mean_gradient_magnitude": mean_gradient,
            "mean_gradient_direction": mean_direction,
        }
    boundary_kappa = node_curvatures[boundary_mask]
    boundary_gradient = gradient_magnitude[boundary_mask]
    boundary_direction = gradient_direction[boundary_mask]
    kappa_pass_fraction = float(np.mean(boundary_kappa < kappa_threshold))
    gradient_pass_fraction = float(np.mean(boundary_gradient >= gradient_threshold))
    mean_gradient = float(np.nanmean(boundary_gradient)) if np.any(np.isfinite(boundary_gradient)) else float("nan")
    mean_direction = float(np.nanmean(boundary_direction)) if np.any(np.isfinite(boundary_direction)) else float("nan")
    overall_pass = kappa_pass_fraction >= 0.8 and gradient_pass_fraction >= 0.8
    return {
        "boundary_cell_count": boundary_count,
        "kappa_pass_fraction": kappa_pass_fraction,
        "gradient_pass_fraction": gradient_pass_fraction,
        "kappa_threshold": kappa_threshold,
        "gradient_threshold": gradient_threshold,
        "pass": overall_pass,
        "mean_gradient_magnitude": mean_gradient,
        "mean_gradient_direction": mean_direction,
        "boundary_kappa_mean": float(np.nanmean(boundary_kappa)),
        "boundary_kappa_std": float(np.nanstd(boundary_kappa)),
    }


def render_histogram(
    node_curvatures: np.ndarray,
    boundary_mask: np.ndarray,
    output_path: Path,
    kappa_threshold: float,
) -> None:
    valid = np.isfinite(node_curvatures)
    boundary_values = node_curvatures[boundary_mask & valid]
    plt.figure(figsize=(8, 4))
    if boundary_values.size:
        plt.hist(boundary_values, bins=20, color="tab:blue", alpha=0.7, label="Boundary κ")
    else:
        plt.text(0.5, 0.5, "No boundary data", ha="center", va="center", transform=plt.gca().transAxes)
    plt.axvline(kappa_threshold, color="tab:red", linestyle="--", label=f"κ = {kappa_threshold}")
    plt.xlabel("Ollivier–Ricci curvature κ")
    plt.ylabel("Count")
    plt.title("Boundary curvature distribution")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    input_path = Path(args.input)
    coupling_files = collect_coupling_files(input_path)
    if not coupling_files:
        LOGGER.warning("No coupling files found at %s", input_path)
    base_size: int | None = None
    combined_curvatures = None
    combined_gradient = None
    combined_direction = None
    curvature_counts = None
    gradient_counts = None
    for file_path in coupling_files:
        g = load_matrix(file_path)
        adjacency = build_graph_from_coupling(g, symmetrize=args.median_symmetrize)
        if base_size is None:
            base_size = len(adjacency)
            combined_curvatures = np.zeros(base_size, dtype=float)
            combined_gradient = np.zeros(base_size, dtype=float)
            combined_direction = np.zeros(base_size, dtype=float)
            curvature_counts = np.zeros(base_size, dtype=float)
            gradient_counts = np.zeros(base_size, dtype=float)
        elif len(adjacency) != base_size:
            raise ValueError(
                "All coupling matrices must have the same dimension;"
                f" got {len(adjacency)} vs {base_size}"
            )
        distances = compute_shortest_paths(adjacency)
        _, node_curvatures, gradient_magnitude, gradient_direction = aggregate_curvatures(
            adjacency, distances, alpha=args.alpha
        )
        curvature_mask = np.isfinite(node_curvatures)
        gradient_mask = np.isfinite(gradient_magnitude)
        if np.any(curvature_mask):
            combined_curvatures[curvature_mask] += node_curvatures[curvature_mask]
            curvature_counts[curvature_mask] += 1
        if np.any(gradient_mask):
            combined_gradient[gradient_mask] += gradient_magnitude[gradient_mask]
            gradient_counts[gradient_mask] += 1
            combined_direction[gradient_mask] += gradient_direction[gradient_mask]
    summary_path = Path(args.output_dir) / "curvature_report.json"
    figure_path = Path(args.figure_dir) / "curvature_hist.png"
    if base_size is None or combined_curvatures is None:
        LOGGER.error("No valid coupling data processed; generating empty summary.")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        empty_summary = {
            "boundary_cell_count": 0,
            "kappa_pass_fraction": 0.0,
            "gradient_pass_fraction": 0.0,
            "kappa_threshold": -0.1,
            "gradient_threshold": 0.05,
            "pass": False,
            "mean_gradient_magnitude": float("nan"),
            "mean_gradient_direction": float("nan"),
            "grid_spacing": args.grid_spacing,
            "coupling_file_count": 0,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(empty_summary, f, indent=2)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()
        return
    averaged_curvature = np.divide(
        combined_curvatures,
        curvature_counts,
        out=np.full_like(combined_curvatures, np.nan),
        where=curvature_counts > 0,
    )
    averaged_gradient = np.divide(
        combined_gradient,
        gradient_counts,
        out=np.full_like(combined_gradient, np.nan),
        where=gradient_counts > 0,
    )
    averaged_direction = np.divide(
        combined_direction,
        gradient_counts,
        out=np.full_like(combined_direction, np.nan),
        where=gradient_counts > 0,
    )
    if args.boundary_mask:
        boundary_mask = load_boundary_mask(Path(args.boundary_mask), size=len(averaged_curvature))
    else:
        boundary_mask = np.ones(len(averaged_curvature), dtype=bool)
    summary = summarize_statistics(
        averaged_curvature,
        averaged_gradient,
        averaged_direction,
        boundary_mask,
    )
    summary["grid_spacing"] = args.grid_spacing
    summary["coupling_file_count"] = len(coupling_files)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    render_histogram(averaged_curvature, boundary_mask, figure_path, summary["kappa_threshold"])
    LOGGER.info("Curvature analysis complete. Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
