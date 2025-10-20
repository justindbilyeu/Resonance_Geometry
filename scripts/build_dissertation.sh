#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/docs/build"
mkdir -p "${OUT_DIR}"

# Default pandoc args
# Use tectonic by default for LaTeX (auto-fetches packages on CI)
PDF_ENGINE="${PDF_ENGINE:-tectonic}"
META="${ROOT_DIR}/docs/dissertation/pandoc/defaults.yaml"

echo "[dissertation] building from defaults: ${META}"

# PDF
echo "[dissertation] -> PDF"
pandoc --pdf-engine="${PDF_ENGINE}" --defaults="${META}" \
  -o "${OUT_DIR}/dissertation.pdf"

# HTML
echo "[dissertation] -> HTML"
pandoc --defaults="${META}" -t html5 \
  -o "${OUT_DIR}/dissertation.html"

echo "[dissertation] outputs:"
ls -lh "${OUT_DIR}"
