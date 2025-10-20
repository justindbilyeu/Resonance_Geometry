#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "[build] Pandoc version:"
pandoc -v | head -2

ENGINE="xelatex"
if command -v xelatex >/dev/null 2>&1; then
  ENGINE="xelatex"
elif command -v tectonic >/dev/null 2>&1; then
  ENGINE="tectonic"
elif command -v pdflatex >/dev/null 2>&1; then
  ENGINE="pdflatex"
else
  echo "[build] ERROR: No TeX engine found (xelatex/tectonic/pdflatex)."
  exit 47
fi
echo "[build] Using TeX engine: ${ENGINE}"

OUT_DIR="docs/dissertation/build"
OUT_PDF="${OUT_DIR}/resonance_geometry_dissertation.pdf"
OUT_HTML="${OUT_DIR}/resonance_geometry_dissertation.html"
mkdir -p "${OUT_DIR}"

echo ""
echo "[build] Checking chapter inputs..."
ls -1 docs/dissertation/*.md

echo ""
echo "[build] Compiling PDF -> ${OUT_PDF}"
pandoc --defaults=docs/dissertation/pandoc/defaults.yaml \
       --pdf-engine="${ENGINE}" \
       -o "${OUT_PDF}"

echo ""
echo "[build] Compiling HTML -> ${OUT_HTML}"
pandoc --defaults=docs/dissertation/pandoc/defaults.yaml \
       -o "${OUT_HTML}"

echo ""
echo "[build] Build dir contents:"
ls -lah "${OUT_DIR}" || true
