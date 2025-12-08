#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

PANDOC_DEFAULTS="docs/dissertation/pandoc_defaults_dissertation.yaml"
OUTDIR="docs/dissertation/build"

mkdir -p "$OUTDIR"

pandoc \
  --defaults="$PANDOC_DEFAULTS" \
  -o "$OUTDIR/resonance_geometry_dissertation.pdf"
