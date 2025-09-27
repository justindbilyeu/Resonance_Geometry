# Upstream archive import log (2025-09-27)

Attempts were made to import historical repositories using `git subtree add`:

- `git subtree add --prefix archive/REAL <url> <branch>`
- `git subtree add --prefix archive/Geometric-Plasticity- <url> <branch>`
- `git subtree add --prefix archive/ResonanceGeometry <url> <branch>`

Each command requires outbound HTTPS access to GitHub. The sandbox returned `CONNECT tunnel failed, response 403` (see shell log), so the upstream histories could not be fetched automatically. Placeholder directories with `IMPORT_NOTES.md` files were created to record the status and to make a future subtree add straightforward when network access is available.

## Follow-up checklist once network access is restored

1. Run the `git subtree add` commands above, targeting the default branch of each source repository.
2. Regenerate [`docs/history/archive_inventory.json`](archive_inventory.json) using `python scripts/generate_inventory.py` (see below).
3. Append per-file entries to [`HISTORY.md`](HISTORY.md) with commit hashes and import notes.
4. Reconcile any filename conflicts by placing alternates under `docs/history/variants/` and noting the rationale.
5. Update the README bundle and Epistemic Status box if new canonical documents are promoted.

## Local helper script

A small helper script (`scripts/generate_inventory.py`) can be run to rebuild the JSON inventory after imports:

```bash
python scripts/generate_inventory.py --output docs/history/archive_inventory.json
```

The script walks `docs/`, `simulations/`, `figures/`, and `archive/`, normalizes extensions, and produces a deterministic JSON map grouped by directory and filetype.
