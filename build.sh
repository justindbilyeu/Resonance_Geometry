#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:-paper}

case "$TARGET" in
  paper)
    pandoc A_Geometric_Theory_of_AI_Hallucination.md \
      --from gfm \
      --pdf-engine=xelatex \
      -V geometry:margin=1in \
      -V fontsize=11pt \
      -o paper.pdf
    echo "Built paper.pdf"
    ;;
  dissertation)
    mkdir -p docs/dissertation/build
    pandoc -d docs/dissertation/dissertation.yml
    echo "Built docs/dissertation/build/resonance_geometry_dissertation.pdf"
    ;;
  *)
    echo "Usage: $0 [paper|dissertation]" 1>&2
    exit 1
    ;;
esac
