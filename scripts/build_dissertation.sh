#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIS_DIR="$ROOT_DIR/docs/dissertation"
BUILD_DIR="$ROOT_DIR/docs/build"

mkdir -p "$BUILD_DIR"

# Optional override: RG_TEX_ENGINE=tectonic|xelatex (default to auto-detect)
: "${RG_TEX_ENGINE:=}"
if [[ -z "$RG_TEX_ENGINE" ]]; then
  if command -v tectonic >/dev/null 2>&1; then
    RG_TEX_ENGINE=tectonic
  else
    RG_TEX_ENGINE=xelatex
  fi
fi

echo "[build] Pandoc version:"
pandoc -v | head -n1
echo "[build] Using TeX engine: $RG_TEX_ENGINE"

echo "[build] Checking chapter inputs..."
for f in 00_prologue.md 01_introduction.md 02_foundations.md 03_general_theory.md 04_retrospective.md; do
  test -f "$DIS_DIR/$f" || { echo "Missing: $DIS_DIR/$f"; exit 1; }
done

PDF_OUT="$BUILD_DIR/dissertation.pdf"
HTML_OUT="$BUILD_DIR/dissertation.html"

echo "[build] Compiling PDF -> $PDF_OUT"
pandoc -d "$DIS_DIR/dissertation.yml" --pdf-engine="$RG_TEX_ENGINE" -o "$PDF_OUT"
test -f "$PDF_OUT" && echo "[build] OK -> $PDF_OUT" || { echo "[build] FAILED"; exit 1; }

echo "[build] Rendering HTML -> $HTML_OUT"
pandoc -d "$DIS_DIR/dissertation.yml" -t html5 -o "$HTML_OUT"
test -f "$HTML_OUT" && echo "[build] OK -> $HTML_OUT" || echo "[build] ⚠️ HTML output missing"
