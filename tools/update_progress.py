import json
import pathlib
import sys
import time


def update_progress(stage, value):
    """Update docs/data/status/summary.json with progress % per stage."""
    path = pathlib.Path("docs/data/status/summary.json")
    if not path.exists():
        status = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "progress": {}}
    else:
        status = json.load(open(path))

    status.setdefault("progress", {})
    status["progress"][stage] = value
    status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(status, open(path, "w"), indent=2)
    print(f"[progress] {stage} → {value}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/update_progress.py <stage> <percent>")
        sys.exit(1)
    update_progress(sys.argv[1], int(sys.argv[2]))
