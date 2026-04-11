# Phase 4C — Run Plan (Gemini / Grok / DeepSeek)

**Date:** 2025-10-26
**Experimenter:** Justin
**Protocol:** Near-Miss + Paraphrase robustness (standardized prompt; randomized order)
**Docs:** A′ = branch_A_paraphrase.md, B′ = branch_B_paraphrase.md, C = branch_C_near_miss.md

## Randomization (pre-registration)
Flip a fair coin for each model to pick one of the 6 permutations of {A′, B′, C}.
Record assignments below **before** running.

| Model     | Order (1→3) | Timestamp (start) | temp | top_p | top_k |
|-----------|--------------|-------------------|------|-------|-------|
| Gemini    |              |                   | 0.7  | 1.0   | 50    |
| Grok      |              |                   | 0.7  | 1.0   | 50    |
| DeepSeek  |              |                   | 0.7  | 1.0   | 50    |

## Expected outcomes (pre-registered)
- A′ (coherent): **Integrate with reservations**; Φ≈0.75–0.85, κ≈0.45–0.60, λ≈0.70–0.80, ITPU≈0.25–0.35
- B′ (falsifier): **Reject**; Φ≈0.20–0.40, κ≈0.65–0.95, λ≈0.05–0.25, ITPU<0.10
- C (near-miss): **Integrate with reservations**; single contradiction flagged; metrics close to A′ but κ slightly higher, ITPU slightly lower.

## Logging checklist (per doc)
- [ ] Paste `deployment/standardized_eval_prompt.md`
- [ ] Paste doc content (`../branch_*`)
- [ ] Save verbatim response under `../responses/<model>_<doc>_<YYYYMMDD>.md`
- [ ] Fill model scorecard and append to `../analysis/metrics_rolling.csv`
