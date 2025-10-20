#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULTS_PATH="${PANDOC_DEFAULTS:-}"
if [[ -z "$DEFAULTS_PATH" ]]; then
  if [[ -f docs/dissertation/pandoc/defaults.yml ]]; then
    DEFAULTS_PATH="docs/dissertation/pandoc/defaults.yml"
  else
    DEFAULTS_PATH="docs/dissertation/pandoc/defaults.yaml"
  fi
fi

if [[ ! -f "$DEFAULTS_PATH" ]]; then
  echo "[build] ERROR: Pandoc defaults file not found: $DEFAULTS_PATH" >&2
  exit 1
fi

OUT_DIR="docs/dissertation/build"
mkdir -p "$OUT_DIR"

override_engine="${PANDOC_PDF_ENGINE:-}"
if [[ -z "$override_engine" && -n "${PDF_ENGINE:-}" ]]; then
  override_engine="${PDF_ENGINE}"
fi

PDF_ENGINE_ARG=()
if [[ -n "$override_engine" ]]; then
  echo "[build] Using PDF engine override: ${override_engine}"
  PDF_ENGINE_ARG=(--pdf-engine="${override_engine}")
fi

echo "[build] Defaults: $DEFAULTS_PATH"

echo "[build] Building PDF…"
pandoc \
  --defaults "$DEFAULTS_PATH" \
  "${PDF_ENGINE_ARG[@]}" \
  -o "$OUT_DIR/resonance_geometry_dissertation.pdf"

echo "[build] Building HTML…"
pandoc \
  --defaults "$DEFAULTS_PATH" \
  -t html5 \
  "${PDF_ENGINE_ARG[@]}" \
  -o "$OUT_DIR/resonance_geometry_dissertation.html"

echo "[build] Outputs:"
ls -lh "$OUT_DIR"
