#!/usr/bin/env python3
"""Update docs/data/status/summary.json with theory + experiment metadata."""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.load_phase_surface import load_phase_surfaces

STATUS = Path("docs/data/status/summary.json")
RINGING_SUMMARY = Path("results/ringing_sweep/summary.json")
JACOBIAN_DIR = Path("results/jacobian_sweep")
FLUENCY_PROBE_SUMMARY = Path("results/fluency_probe/summary.json")
FLUENCY_SWEEP_SUMMARY = Path("results/fluency_sweep/summary.json")


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_status() -> Dict[str, Any]:
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text())
        except Exception:
            return {}
    return {}


def _ensure(status: Dict[str, Any], key: str, value: Optional[Any]) -> None:
    if value is None:
        return
    status[key] = value


def update_theory(status: Dict[str, Any]) -> None:
    theory = load_phase_surfaces()
    status.setdefault("theory", {})
    status["theory"]["phase_surface"] = theory

    theory_dir = ROOT / "docs/data/theory"
    if theory_dir.is_dir():
        jsons = list(theory_dir.glob("*.json"))
        status["theory_exports"] = {
            "count_json": len(jsons),
            "has_eig_traj": (theory_dir / "jacobian_eig_trajectories.json").exists(),
            "has_error_curve": (theory_dir / "jacobian_error_curve.json").exists(),
            "has_phase_surface": (theory_dir / "phase_surface_all.json").exists(),
        }


def update_experiments(status: Dict[str, Any]) -> None:
    _ensure(status, "ringing_sweep", _load_json(RINGING_SUMMARY))

    if JACOBIAN_DIR.exists():
        _ensure(
            status,
            "jacobian",
            {
                "meta": _load_json(JACOBIAN_DIR / "meta.json"),
                "n_boundary": len(_load_json(JACOBIAN_DIR / "boundary.json") or []),
            },
        )

    _ensure(status, "fluency_probe", _load_json(FLUENCY_PROBE_SUMMARY))
    _ensure(status, "fluency_sweep", _load_json(FLUENCY_SWEEP_SUMMARY))


def main() -> None:
    status = _load_status()
    update_theory(status)
    update_experiments(status)
    status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2) + "\n")

    print("[status] updated", STATUS)


if __name__ == "__main__":
    main()
