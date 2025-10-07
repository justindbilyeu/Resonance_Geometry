#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar nullglob
for md in docs/papers/**/*.md; do
  pdf="${md%.md}.pdf"
  echo "Building $pdf"
  pandoc "$md" \
    --from markdown+tex_math_single_backslash \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    --resource-path=.:"docs/papers/neurips/figures":"docs/papers/neurips" \
    --output "$pdf"
done
