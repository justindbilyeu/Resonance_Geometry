# Archived Workflows

This directory contains historical GitHub Actions workflows that are no longer active in the repository's Actions UI.

## Contents

- **pandoc-build.yml** - Legacy Pandoc build workflow, superseded by `build-papers.yml`
- **build-papers.yml** - Legacy paper build workflow, superseded by `dissertation-build.yml`
- **dissertation-build.yml** - Archived dissertation PDF pipeline (paused during analysis integration)
- **ci-core.yml** - Deprecated core CI workflow, replaced by `ci.yml` and `RWP-ci.yml`

## Why Archive?

These workflows were moved out of `.github/workflows/` to:
1. Reduce confusion in the GitHub Actions UI
2. Prevent accidental triggering of deprecated workflows
3. Preserve historical workflow definitions for reference

## Active Workflows

The following workflows remain active under `.github/workflows/`:

- **ci.yml** - Main CI (full checks)
- **RWP-ci.yml** - Library / RWP core CI
- **docs-only.yml** - Docs/data-only PR fast path
- **pages.yml** - GitHub Pages deployment
- **gp-demo.yml** - GP ringing demo
- **sims.yml** - Simulation smoke tests
- **sim-validate.yml** - Manual simulation validation
- **theory.yml** - Theory exports + dashboard update
- **paper-figs.yml** - Non-Hopf paper figures + PDF build
- **manual-analysis.yml** - Manual analysis runs
- **poison-detection-test.yml** - Poison detection tests (gated by env var)

## Restoration

If you need to restore any of these workflows:
1. Move the YAML file back to `.github/workflows/`
2. Review and update any deprecated GitHub Actions syntax
3. Test thoroughly before enabling on main branch
