#!/usr/bin/env bash
set -euo pipefail
pandoc A_Geometric_Theory_of_AI_Hallucination.md \
  --from gfm \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -o paper.pdf
echo "Built paper.pdf"
