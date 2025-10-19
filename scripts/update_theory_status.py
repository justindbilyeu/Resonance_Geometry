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

def update_ringing(status: dict) -> None:
    if SWEEP_SUMMARY.exists():
        status["ringing_sweep"] = json.loads(SWEEP_SUMMARY.read_text())

def main() -> None:
    status = load_status()
    update_theory(status)
    update_ringing(status)
    status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2))
    print("[status] updated theory.phase_surface; files:", status.get("theory", {}).get("phase_surface", {}).get("files_present", []))
    if "ringing_sweep" in status:
        print("[status] incorporated ringing sweep summary")

if __name__ == "__main__":
    main()
