"""
Command-line interface for F_AI intrinsic metric.

Usage:
    python -m src.f_ai.cli --input data.jsonl --out results/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .io import (
    load_episodes,
    load_json_config,
    load_loop_cycles,
    load_coefficients,
)
from .segment import get_calibration_split, get_evaluation_split
from .embeddings import create_embedding_backend, EpisodeEmbedder
from .functional import (
    compute_calibration_from_episodes,
    compute_f_ai_batch,
)
from .report import write_all_reports


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="F_AI Intrinsic Metric v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with TF-IDF backend
  python -m src.f_ai.cli --input data/episodes.jsonl --out results/

  # With custom configs
  python -m src.f_ai.cli \\
      --input data/episodes.jsonl \\
      --topology-config configs/f_ai/topology_configs.json \\
      --loops configs/f_ai/loop_cycles.json \\
      --coeff configs/f_ai/coefficients.json \\
      --out results/

  # With sentence-transformer backend (requires optional dependencies)
  python -m src.f_ai.cli \\
      --input data/episodes.jsonl \\
      --embedding-backend sentence_transformer \\
      --out results/

  # Custom calibration split
  python -m src.f_ai.cli \\
      --input data/episodes.jsonl \\
      --calibration-split first_n=100 \\
      --out results/
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with episodes",
    )

    parser.add_argument(
        "--topology-config",
        type=str,
        default="configs/f_ai/topology_configs.json",
        help="Path to topology configurations (default: configs/f_ai/topology_configs.json)",
    )

    parser.add_argument(
        "--loops",
        type=str,
        default="configs/f_ai/loop_cycles.json",
        help="Path to loop cycle definitions (default: configs/f_ai/loop_cycles.json)",
    )

    parser.add_argument(
        "--coeff",
        type=str,
        default="configs/f_ai/coefficients.json",
        help="Path to coefficient definitions (default: configs/f_ai/coefficients.json)",
    )

    parser.add_argument(
        "--calibration-split",
        type=str,
        default="first_n=50",
        help="Calibration split method: 'first_n=N' or 'file=PATH' (default: first_n=50)",
    )

    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to calibration episode IDs file (for file-based split)",
    )

    parser.add_argument(
        "--embedding-backend",
        type=str,
        choices=["tfidf", "sentence_transformer"],
        default="tfidf",
        help="Embedding backend (default: tfidf)",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/f_ai/",
        help="Output directory for results (default: results/f_ai/)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate episodes against schema (default: True)",
    )

    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip schema validation",
    )

    return parser.parse_args(args)


def main(args=None):
    """Main entry point."""
    args = parse_args(args)

    print("=" * 60)
    print("F_AI Intrinsic Metric v1.0")
    print("=" * 60)

    # 1. Load configurations
    print("\n[1/6] Loading configurations...")
    print(f"  Topology config: {args.topology_config}")
    print(f"  Loop cycles: {args.loops}")
    print(f"  Coefficients: {args.coeff}")

    topology_config_full = load_json_config(args.topology_config)
    topology_configs = topology_config_full["topologies"]

    coefficients = load_coefficients(args.coeff)

    # Load cycles for each topology
    cycles_by_topology = {}
    for topology_id in topology_configs.keys():
        cycles_by_topology[topology_id] = load_loop_cycles(args.loops, topology_id)

    print(f"  Loaded {len(topology_configs)} topologies")

    # 2. Load episodes
    print(f"\n[2/6] Loading episodes from {args.input}...")

    episodes = load_episodes(
        args.input,
        topology_config_path=args.topology_config if args.validate else None,
        validate=args.validate,
    )

    print(f"  Loaded {len(episodes)} episodes")

    # Count by topology
    from collections import Counter

    topology_counts = Counter(ep.topology_id for ep in episodes)
    for topo, count in topology_counts.items():
        print(f"    - {topo}: {count} episodes")

    # 3. Calibration split
    print("\n[3/6] Creating calibration split...")

    # Parse calibration split argument
    if args.calibration_split.startswith("first_n="):
        method = "first_n"
        n = int(args.calibration_split.split("=")[1])
        print(f"  Method: first_n (n={n})")
        calibration_episodes = get_calibration_split(
            episodes, method="first_n", n=n
        )
    elif args.calibration_split.startswith("file=") or args.calibration_file:
        method = "from_file"
        file_path = (
            args.calibration_file
            if args.calibration_file
            else args.calibration_split.split("=")[1]
        )
        print(f"  Method: from_file ({file_path})")
        calibration_episodes = get_calibration_split(
            episodes, method="from_file", calibration_file=file_path
        )
    else:
        print(f"  Unknown calibration split format: {args.calibration_split}")
        print("  Using default: first_n=50")
        method = "first_n"
        n = 50
        calibration_episodes = get_calibration_split(
            episodes, method="first_n", n=n
        )

    evaluation_episodes = get_evaluation_split(episodes, calibration_episodes)

    print(f"  Calibration: {len(calibration_episodes)} episodes")
    print(f"  Evaluation: {len(evaluation_episodes)} episodes")

    # 4. Create embedding backend
    print(f"\n[4/6] Initializing embedding backend: {args.embedding_backend}...")

    if args.embedding_backend == "tfidf":
        backend = create_embedding_backend("tfidf", max_features=384)
        print("  Backend: TF-IDF (384 dimensions)")
    elif args.embedding_backend == "sentence_transformer":
        try:
            backend = create_embedding_backend(
                "sentence_transformer", model_name="all-MiniLM-L6-v2"
            )
            print("  Backend: sentence-transformers (all-MiniLM-L6-v2, 384 dims)")
        except ImportError as e:
            print(f"  Error: {e}")
            print("  Falling back to TF-IDF backend")
            backend = create_embedding_backend("tfidf", max_features=384)
    else:
        print(f"  Unknown backend: {args.embedding_backend}")
        sys.exit(1)

    embedder = EpisodeEmbedder(backend)

    # 5. Compute calibration
    print("\n[5/6] Computing calibration parameters...")

    calibration_by_topology = compute_calibration_from_episodes(
        calibration_episodes,
        embedder,
        topology_configs,
        cycles_by_topology,
        coefficients=coefficients,
    )

    for topology_id, calib in calibration_by_topology.items():
        print(f"  {topology_id}: {calib.n_calibration_episodes} episodes")

    # 6. Compute F_AI for all episodes
    print("\n[6/6] Computing F_AI metrics...")

    all_metrics = compute_f_ai_batch(
        episodes,
        embedder,
        topology_configs,
        cycles_by_topology,
        calibration_by_topology,
        coefficients=coefficients,
    )

    print(f"  Computed metrics for {len(all_metrics)} episodes")

    # 7. Write reports
    print(f"\nWriting reports to {args.out}...")

    # Determine calibration method and n for reporting
    calib_method = method
    calib_n = n if method == "first_n" else len(calibration_episodes)

    write_all_reports(
        all_metrics,
        calibration_by_topology,
        args.out,
        calibration_method=calib_method,
        calibration_n=calib_n,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
