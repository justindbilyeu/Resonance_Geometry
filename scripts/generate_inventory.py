#!/usr/bin/env python3
"""Generate a JSON inventory of docs, simulations, figures, and archive contents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ["docs", "simulations", "figures", "archive"]


def _scan_directory(directory: Path) -> Dict[str, List[str]]:
    """Return a mapping of file suffix -> sorted list of relative paths."""
    results: Dict[str, List[str]] = defaultdict(list)
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix or "<noext>"
        rel_path = path.relative_to(ROOT).as_posix()
        results[suffix].append(rel_path)
    return {suffix: sorted(paths) for suffix, paths in sorted(results.items())}


def build_inventory() -> Dict[str, object]:
    inventory: Dict[str, object] = {}
    for name in DEFAULT_TARGETS:
        directory = ROOT / name
        if not directory.exists():
            continue
        if name == "archive":
            per_repo: Dict[str, Dict[str, List[str]]] = {}
            for repo_dir in sorted(directory.iterdir()):
                if not repo_dir.is_dir():
                    continue
                per_repo[repo_dir.name] = _scan_directory(repo_dir)
            inventory[name] = per_repo
        else:
            inventory[name] = _scan_directory(directory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON inventory. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Number of spaces to use for JSON indentation (default: 2).",
    )
    args = parser.parse_args()

    inventory = build_inventory()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, indent=args.indent) + "\n", encoding="utf-8")
    else:
        print(json.dumps(inventory, indent=args.indent))

if __name__ == "__main__":
    main()
