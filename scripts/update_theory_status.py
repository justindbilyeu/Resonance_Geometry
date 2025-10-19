#!/usr/bin/env python3
"""
Update docs/data/status/summary.json with theory/phase_surface status and optional ringing sweep summaries.
Safe to run even if no theory JSONs or ringing artifacts have been added yet.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.load_phase_surface import load_phase_surfaces

STATUS = Path("docs/data/status/summary.json")
SWEEP_SUMMARY = Path("results/ringing_sweep/summary.json")
FLUENCY_SUMMARY = Path("results/fluency_probe/summary.json")
FLUENCY_SWEEP_SUMMARY = Path("results/fluency_sweep/summary.json")

def load_status() -> dict:
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text())
        except Exception:
            return {}
    return {}

def update_theory(status: dict) -> None:
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

def update_ringing(status: dict) -> None:
    if SWEEP_SUMMARY.exists():
        status["ringing_sweep"] = json.loads(SWEEP_SUMMARY.read_text())


def update_fluency(status: dict) -> None:
    if FLUENCY_SUMMARY.exists():
        try:
            status["fluency_probe"] = json.loads(FLUENCY_SUMMARY.read_text())
        except Exception:
            pass

    if FLUENCY_SWEEP_SUMMARY.exists():
        try:
            status["fluency_sweep"] = json.loads(FLUENCY_SWEEP_SUMMARY.read_text())
        except Exception:
            pass

def main() -> None:
    status = load_status()
    update_theory(status)
    update_ringing(status)
    update_fluency(status)
    status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2))
    print("[status] updated theory.phase_surface; files:", status.get("theory", {}).get("phase_surface", {}).get("files_present", []))
    if "ringing_sweep" in status:
        print("[status] incorporated ringing sweep summary")
    if "fluency_probe" in status:
        print("[status] incorporated fluency probe summary")
    if "fluency_sweep" in status:
        print("[status] incorporated fluency sweep summary")

if __name__ == "__main__":
    main()
