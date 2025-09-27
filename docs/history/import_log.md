# Upstream archive import log (2025-09-27)

Attempts were made to import historical repositories using `git subtree add`:

- `git subtree add --prefix archive/REAL <url> <branch>`
- `git subtree add --prefix archive/Geometric-Plasticity- <url> <branch>`
- `git subtree add --prefix archive/ResonanceGeometry <url> <branch>`

Each command requires outbound HTTPS access to GitHub. The sandbox returned `CONNECT tunnel failed, response 403` (see shell log), so the upstream histories could not be fetched automatically. Placeholder directories with `IMPORT_NOTES.md` files were created to record the status and to make a future subtree add straightforward when network access is available.
