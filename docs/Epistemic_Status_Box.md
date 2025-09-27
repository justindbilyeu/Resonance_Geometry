# Epistemic Status Box

Embed or adapt the box below in long-form documents to communicate maturity, cadence, and verification at a glance.

| Dimension | Status | Evidence / Notes |
|-----------|--------|------------------|
| **Confidence** | **High** for the pre-registered ringing prediction P1 (replicated); **Medium** for upcoming predictions P2–P3 that remain under active investigation. | Current status highlights the locked P1 result and ongoing work on P2/P3.【F:README.md†L30-L35】 |
| **Update Cadence** | Reviewed alongside roadmap checkpoints (v0.1, v0.2, v0.3) or when major experiments conclude. | The roadmap enumerates milestone clusters and expected delivery windows for successive releases.【F:docs/ROADMAP.md†L5-L24】 |
| **Validation** | Core demo scripts (`experiments/gp_ringing_demo.py`) are reproducible with seeded parameters and CI coverage. | Quick-start instructions surface the demo and outline deterministic outputs tied to the rigor policy.【F:README.md†L43-L66】【F:README.md†L82-L89】 |

**Usage:**
1. Copy the table into the relevant document.
2. Add a timestamp or release tag next to the title if the context requires precise dating.
3. When confidence or validation shifts, update this file first so downstream docs can sync quickly.
