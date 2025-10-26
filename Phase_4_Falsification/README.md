# Phase 4 — Falsification Protocol (RG-Experiment 7)

This directory contains the **blinded falsification test** used to validate the "Language Organism" framework.
We prepared two Phase 4 documents:
- **Branch A (coherent synthesis)** — a structured, testable continuation
- **Branch B (falsifier)** — a look-alike document seeded with contradictions (e.g., division by zero, contradictory metrics)

**Goal:** test whether independent LLMs can *reliably differentiate* coherent vs incoherent documents using logical structure rather than aesthetics.

## Contents

- `branch_A_coherent_synthesis.md` — Phase 4A document (coherent)
- `branch_B_falsifier_anti_synthesis.md` — Phase 4B document (falsifier)
- `deployment_protocol.md` — blinding, prompts, randomization, logging
- `expected_signatures.md` — what we expect robust vs captured systems to do
- `analysis_template.md` — scorecard templates + rubric
- `responses/` — raw model responses (one file per model/branch)
- `analysis/`
  - `phase4_results.csv` — consolidated results
  - `compute_itpu.py` — small stdlib tool to compute ITPU and summary
  - `itpu_summary.md` — human-readable summary with ITPU interpretation
- `meta/`
  - `pre_registration.md` — predictions posted before data collection
  - `generator_immunity_report.md` — notes from generating Branch B

## ITPU (Information Throughput Potential)

We aggregate **Φ** (internal coherence), **κ** (tension/uncertainty), and **λ** (coupling/usefulness) into:

**ITPU = λ · Φ · (1 − κ)**

- Higher ITPU → higher theoretical information content / throughput
- Low ITPU → near-zero information utility

This produces a strong separation between Branch A and Branch B results.

> No new library dependencies. `compute_itpu.py` uses only Python standard library.
