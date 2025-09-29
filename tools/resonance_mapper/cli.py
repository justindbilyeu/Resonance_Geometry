"""Command line interface for the resonance mapper pipeline."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - optional dependency branch
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

from .gnn_encoder import train_gnn
from .graphify import build_graph_from_multi_freq
from .loader import load_multi_freq
from .tda import compute_tda


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resonance Mapper Pipeline")
    parser.add_argument("inputs", help="Path to multi-frequency JSON results")
    parser.add_argument("out", help="Output directory for embeddings and reports")
    parser.add_argument("--tda", action="store_true", help="Compute persistent homology outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    results = load_multi_freq(args.inputs)
    graph = build_graph_from_multi_freq(results)
    if args.device != "auto":
        device = args.device
    elif _TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    embeddings = train_gnn(graph, epochs=args.epochs, device=device)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "embeddings.npy"), embeddings)

    if args.tda:
        betti, diagram, _ = compute_tda(embeddings)
        np.save(os.path.join(args.out, "tda_diagram.npy"), diagram)
        os.makedirs("figures", exist_ok=True)
        if diagram.size:
            plt.scatter(diagram[:, 0], diagram[:, 1])
        plt.savefig("figures/mapper_persistence.png")
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
            json.dump({"betti": betti}, fh)


if __name__ == "__main__":
    main()
