"""Test configuration for Resonance Geometry."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --- Known-failure quarantine -------------------------------------------
# Reads tests/known_failures.txt and marks each listed node ID
# xfail(strict=True), so CI can gate on everything else. Strict means a
# quarantined test that starts PASSING turns CI red, forcing the list to
# shrink. A listed node ID that matches nothing also turns CI red, so the
# list cannot rot as tests are renamed or deleted.

import pytest

KNOWN_FAILURES_FILE = Path(__file__).parent / "known_failures.txt"


def _load_known_failures():
    if not KNOWN_FAILURES_FILE.exists():
        return []
    entries = []
    for line in KNOWN_FAILURES_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def pytest_collection_modifyitems(config, items):
    known = _load_known_failures()
    if not known:
        return
    matched = set()
    for item in items:
        for entry in known:
            if item.nodeid == entry:
                item.add_marker(
                    pytest.mark.xfail(
                        strict=True,
                        reason=f"quarantined in {KNOWN_FAILURES_FILE.name}",
                    )
                )
                matched.add(entry)
    # Only judge an entry stale if its FILE was collected. Otherwise a subset
    # run (pytest tests/test_one.py, or -k) would flag every other entry.
    collected_files = {item.nodeid.split("::", 1)[0] for item in items}
    stale = [
        e
        for e in known
        if e not in matched and e.split("::", 1)[0] in collected_files
    ]
    if stale:
        raise pytest.UsageError(
            "tests/known_failures.txt lists node IDs that no longer exist:\n  "
            + "\n  ".join(stale)
            + "\nRemove them, or fix the node ID."
        )
