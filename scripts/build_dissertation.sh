#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIS_DIR="$ROOT_DIR/docs/dissertation"
BUILD_DIR="$DIS_DIR/build"

mkdir -p "$BUILD_DIR"

echo "[build] Pandoc version:"
pandoc -v | head -n1

echo "[build] Checking chapter inputs..."
for f in 00_prologue.md 01_introduction.md 02_foundations.md 03_general_theory.md 04_retrospective.md; do
  test -f "$DIS_DIR/$f" || { echo "Missing: $DIS_DIR/$f"; exit 1; }
done

echo "[build] Compiling PDF…"
pandoc -d "$DIS_DIR/dissertation.yml"

OUT_PDF="$BUILD_DIR/resonance_geometry_dissertation.pdf"
test -f "$OUT_PDF" && echo "[build] OK -> $OUT_PDF" || { echo "[build] FAILED"; exit 1; }

# Optional HTML build for quick review
if command -v pandoc >/dev/null 2>&1; then
  pandoc \
    --metadata-file="$DIS_DIR/dissertation.yml" \
    -t html5 \
    -o "$BUILD_DIR/resonance_geometry_dissertation.html" \
    "$DIS_DIR/00_prologue.md" \
    "$DIS_DIR/01_introduction.md" \
    "$DIS_DIR/02_foundations.md" \
    "$DIS_DIR/03_general_theory.md" \
    "$DIS_DIR/04_retrospective.md"
  echo "[build] HTML -> $BUILD_DIR/resonance_geometry_dissertation.html"
fi
