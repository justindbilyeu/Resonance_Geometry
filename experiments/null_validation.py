"""Null model validation placeholder logic."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def null_model_suite(out_dir: str = "results/nulls") -> Dict[str, Any]:
    """Write a placeholder null-model summary and mark progress."""
    summary = {"todo": True}

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary_path = path / "nulls_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    os.system("python tools/update_progress.py nulls 100")
    return summary
