# Resonance Geometry Documentation Bundle

This bundle is organized so newcomers can navigate the merged Resonance Geometry
materials without losing track of provenance or epistemic status. Use the
sections below as a guided map.

## 1. Philosophy (`docs/philosophy/`)
- Conceptual framing, prefaces, and phenomenological essays.
- Start with [`RG_GP_Preface.md`](philosophy/RG_GP_Preface.md) for a concise
  description of why adaptive geometry matters.

## 2. White Papers (`docs/white-papers/`)
- Formal write-ups, appendices, and LaTeX sources for the Resonance Geometry and
  Geometric Plasticity programs.
- PDFs and LaTeX sources share filenames; variants or addenda are grouped in
  subdirectories such as `draft/` and `appendices/`.

## 3. Resonance Codex (`docs/codex/`)
- Policies, predictions, experiment protocols, and hardware notes.
- Highlights include [`policies/predictions.md`](codex/policies/predictions.md)
  and the pre-registration dossier in
  [`policies/prereg_P1.md`](codex/policies/prereg_P1.md).

## 4. History & Provenance (`docs/history/`)
- `HISTORY.md` tracks every imported artifact, its source repository, and
  deduplication notes.
- `archive_inventory.json` lists the current file layout to accelerate future
  imports.
- `import_log.md` records outstanding tasks (e.g., pending subtree pulls when
  network access is restored).
- Use `scripts/generate_inventory.py` to regenerate the JSON snapshot after
  importing or reorganizing files.

## 5. Simulations (`simulations/`)
- Headless Python modules that reproduce the figures referenced in the docs.
- CI executes the lightweight suite via `.github/workflows/sims.yml` and uploads
  artifacts from `figures/`.

## 6. Archive (`archive/`)
- Preserves legacy repository trees (`REAL`, `Geometric-Plasticity-`,
  `ResonanceGeometry`). When a full import is blocked (e.g., due to sandbox
  networking limits) the folder contains an `IMPORT_NOTES.md` with the current
  status and TODOs.

## Epistemic Status
- Review [`docs/Epistemic_Status_Box.md`](Epistemic_Status_Box.md) for the
  project-wide confidence assessment, validation checkpoints, and replication
  roadmap.

## How to Contribute
1. Read the Epistemic Status box to understand which claims are stable versus
   exploratory.
2. Use the provenance log to check whether your proposed changes supersede an
   imported variant.
3. Update `HISTORY.md` and the archive inventory whenever new legacy content is
   added or deduplicated.
4. Run the simulations workflow locally (`pytest simulations`) before opening a
   PR so CI remains green.
