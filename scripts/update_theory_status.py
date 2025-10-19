#!/usr/bin/env python3
"""
Update docs/data/status/summary.json with theory/phase_surface status.
Safe to run even if no theory JSONs have been added yet.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from tools.load_phase_surface import load_phase_surfaces

STATUS = Path("docs/data/status/summary.json")

def main() -> None:
    status = {}
    if STATUS.exists():
        try:
            status = json.loads(STATUS.read_text())
        except Exception:
            status = {}

    theory = load_phase_surfaces()
    status.setdefault("theory", {})
    status["theory"]["phase_surface"] = theory
    status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2))
    print("[status] updated theory.phase_surface; files:", theory.get("files_present", []))

if __name__ == "__main__":
    main()
