# Documentation Bundle Navigation Guide

The Resonance Geometry documentation has been reorganized into focused collections. Use the sections below to jump to the right part of the tree and understand what each folder contains.

## Philosophy
- **Path:** `docs/philosophy/`
- **Start here:** [`RG_GP_Preface.md`](philosophy/RG_GP_Preface.md) captures the guiding stance and core theses behind Resonance Geometry and Geometric Plasticity, framing the philosophical commitments that motivate the technical work.【F:docs/philosophy/RG_GP_Preface.md†L1-L28】
- **Tips:** Treat this folder as long-form essays and framing documents. Each draft keeps explicit labels about authorship and status so you can distinguish published language from works in progress.

## White Papers
- **Path:** `docs/whitepaper/`
- **Start here:** The structured white paper lives in [`docs/whitepaper/sections/`](whitepaper/sections/) for modular editing, with compiled exports stored alongside the source when needed.【F:docs/whitepaper/sections/00_preface.md†L1-L6】
- **Tips:** Edit Markdown chapters inside `sections/` and rebuild the LaTeX or PDF exports from the repository root when you need a polished release. Keep figure assets and appendices alongside the corresponding section files for easier cross-referencing.

## Codex (Axioms & Anchors)
- **Path:** `docs/index.html`
- **Start here:** The Codex is rendered as part of the static site; jump to “The Ten Axioms of Structured Resonance” for the canonical axioms with their Cosmos and Bio anchors.【F:docs/index.html†L501-L563】
- **Tips:** Use your browser’s outline/TOC or search for `#the-ten-axioms` to navigate directly. When updating axioms, modify the source Markdown/LaTeX and re-export `index.html` to keep formatting consistent.

## Simulations & Analysis
- **Paths:** `docs/experiments/`, `docs/itpu-sim/`, and `docs/src/`
- **Start here:**
  - [`experiments/phase_surrogate.md`](experiments/phase_surrogate.md) documents how surrogate experiments are structured.【F:docs/experiments/phase_surrogate.md†L1-L15】
  - [`itpu-sim/README.md`](itpu-sim/README.md) explains the software simulator that underpins reference runs.【F:docs/itpu-sim/README.md†L1-L15】
  - [`src/analysis/ringing_threshold.py`](src/analysis/ringing_threshold.py) contains the reference implementation for the ringing-threshold checks cited throughout the appendices.【F:docs/papers/appendix_ringing_threshold.md†L35-L40】
- **Tips:** Use these folders when you need reproducible code or protocol details. Each README spells out how to run the associated scripts, and experiment notes point back to the validated metrics referenced in the white papers.

## History & Roadmaps
- **Path:** `docs/ROADMAP.md`
- **Start here:** The roadmap tracks planned milestones and their completion state, providing a lightweight history of the project’s progression.【F:docs/ROADMAP.md†L1-L18】
- **Tips:** Review this file for release cadence, outstanding tasks, and links to reproducible artifacts. Pair it with repository tags and `results/` archives when you need a fuller historical audit.

---

### Quick Orientation Tips
- Prefer relative links (as above) when editing to keep the bundle portable between GitHub and local previews.
- When unsure where content belongs, drop a short stub in the closest collection and link to it here; the bundle is meant to stay current as the tree evolves.
- Use the shared [`Epistemic_Status_Box.md`](Epistemic_Status_Box.md) to communicate confidence, cadence, and validation whenever you publish a long-form update.
- Keep status labels (“Draft”, “Released”, etc.) up to date within each collection so readers can quickly gauge maturity without leaving the folder.
