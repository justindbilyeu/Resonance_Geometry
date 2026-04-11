#!/bin/bash
# Open all three PRs for session 011CUWtfEK1atiYdPZXEGALp
# Usage: bash open_prs.sh

set -e

echo "=========================================="
echo "Opening 3 PRs for Session 011CUWtfEK1atiYdPZXEGALp"
echo "=========================================="
echo ""

# PR #1: Adaptive η_eff
echo "Creating PR #1: Adaptive η_eff..."
gh pr create \
  --head claude/hallu-v2-adaptive-eta-011CUWtfEK1atiYdPZXEGALp \
  --base main \
  --title "🔁 Integrate Adaptive η_eff (DeepSeek Whitening Gain) and v2 Phase Boundary Updates" \
  --body-file <(cat <<'EOF'
## Summary

This PR integrates **adaptive whitening gain** into the SU(2) hallucination simulator, enabling dynamic adjustment of coupling strength η based on covariance conditioning. The formula η_eff = η × (1 + tanh(log(κ)/d) × cap/15) improves stability for ill-conditioned activation covariances while preserving baseline behavior for well-conditioned cases.

**Key contributions:**
- Simplified `adaptive_gain.py` module with cap-based tanh stabilization
- Parameter sweep with conditioning injection (`run_phase_cond_sweep.py`)
- SVG-based visualization overlay comparing base vs adaptive boundaries
- Mathematical derivation (Appendix D) and independent replication validation (Appendix E)

---

## Testing

**Unit Tests** (22 test cases in `tests/hallucination/test_adaptive_eta.py`)

**Run targets:**
```bash
make phase-cond-sweep
make phase-adaptive-overlay
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)

echo "✓ PR #1 created"
echo ""

# PR #2: RAIC Sandbox
echo "Creating PR #2: RAIC Sandbox..."
gh pr create \
  --head claude/raic-sandbox-v3-011CUWtfEK1atiYdPZXEGALp \
  --base main \
  --title "🎛️ RAIC v3 Sandbox — Resonance-Aware Inference Controller Prototype" \
  --body-file <(cat <<'EOF'
## Summary

This PR introduces the **Resonance-Aware Inference Controller (RAIC)**, a runtime controller that monitors spectral stability (λ_max) during inference and applies adaptive interventions to mitigate hallucination risk.

**Key contributions:**
- `ResonanceAwareController` class with rolling σ-threshold detection
- `BatchRAICHarness` for multi-scenario evaluation
- Synthetic covariance generator with controllable instability
- Demo script and full experiment harness with SVG visualization

---

## Demo

```bash
make raic-demo     # Quick demonstration
make raic-synth    # Full batch experiment
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)

echo "✓ PR #2 created"
echo ""

# PR #3: Empirical Validation
echo "Creating PR #3: Empirical Validation..."
gh pr create \
  --head claude/empirical-kappa-lmax-v1-011CUWtfEK1atiYdPZXEGALp \
  --base main \
  --title "🔬 Empirical Validation — κ(Σ) & λ_max Correlations in TruthfulQA/HaluEval" \
  --body-file <(cat <<'EOF'
## Summary

This PR provides the **empirical validation framework** for testing the hypothesis that λ_max (spectral stability) correlates with hallucination in real LLMs. Includes activation extraction, curvature metrics computation, and ROC-AUC evaluation.

**Key contributions:**
- Activation extraction from HuggingFace transformers (GPT-2, GPT-J, LLaMA)
- Spectral curvature metrics: κ(Σ), λ_max, λ_min, effective rank
- ROC-AUC evaluation for hallucination detection
- Hypothesis testing framework (H1-H4 from paper)
- Mock mode for testing without model loading

---

## Pipeline

```bash
make empirical-extract    # Extract activations (mock mode)
make empirical-curvature  # Compute κ, λ_max
make empirical-eval       # ROC-AUC evaluation
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)

echo "✓ PR #3 created"
echo ""

echo "=========================================="
echo "All 3 PRs created successfully!"
echo "=========================================="
echo ""
echo "View PRs at:"
echo "  https://github.com/justindbilyeu/Resonance_Geometry/pulls"
