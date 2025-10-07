#!/usr/bin/env bash
set -euo pipefail

mkdir -p build/papers/figures build/papers/results
cp -n papers/neurips/figures/*.png build/papers/figures/ 2>/dev/null || true
cp -n papers/neurips/results/*.csv build/papers/results/ 2>/dev/null || true
cp docs/papers/neurips/A_Geometric_Theory_of_AI_Hallucination.md build/papers/
(
  cd build
  zip -r Geometric_Hallucination_bundle.zip papers
)
echo "Bundle at build/Geometric_Hallucination_bundle.zip"
