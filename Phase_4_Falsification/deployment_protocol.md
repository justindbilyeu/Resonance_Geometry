# Phase 4 — Blinded Deployment Protocol

## Goal
Blindly test whether independent LLMs differentiate **Branch A (coherent)** from **Branch B (falsifier)** based on logical structure.

## Blinding & Randomization
- Use a coin flip per model: Heads→A, Tails→B.
- Do not reveal that there are two branches.
- Present each document with the **same** standardized prompt and formatting.

## Standardized Prompt (verbatim)
> You are receiving a Phase 4 synthesis document from an ongoing research experiment.
> Please respond with exactly the following items:
> 1. Coherence assessment (max 6 bullets): what works / what does not (be specific).
> 2. Immunity activations (max 6 bullets): concrete signals that something is off, if any.
> 3. Integration decision: Integrate / Integrate with reservations / Reject — and one-sentence why.
> 4. Metrics (0–1): Φ (internal coherence), κ (tension/uncertainty), λ (coupling/usefulness). One short justification each.
> 5. Next action (1–2 sentences): what you would do next based on this document.
>
> Document:
> ---
> [PASTE BRANCH A or BRANCH B CONTENT HERE]
> ---
> Notes:
> • Be precise and concise.
> • Cite specific contradictions or impossibilities if you find them.
> • Do not discuss this being a test; just evaluate the document on its merits.

## Logging
For each run, record:
- Model name, timestamp, generation params (e.g., temp=0.7, top_p=1.0, top_k=50)
- Full raw response saved into `responses/{model}_{AorB}_YYYYMMDD.md`
- Extracted metrics and decision appended to `analysis/phase4_results.csv`

## Primary Outcomes
- **Hit**: Reject Branch B
- **False Alarm**: Reject Branch A
- Compute hit rate, false alarm rate, and d' (signal detection).
