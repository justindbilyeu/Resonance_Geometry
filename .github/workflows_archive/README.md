# Archived Workflows

This directory contains historical GitHub Actions workflows that are no longer active in the repository's Actions UI.

## Current Status: Dissertation-Focused Workflow Set

**December 2025**: Workflows have been streamlined to focus exclusively on dissertation work. All CI, testing, analysis, and deployment workflows have been archived.

## Active Workflows

The following workflows remain active under `.github/workflows/`:

- **dissertation-build.yml** - Builds dissertation PDF using Pandoc + XeLaTeX
- **docs-only.yml** - Fast-path CI for documentation/data-only changes

## Archived Workflows

### December 2025 - Dissertation Focus Archive (Batch 2)

Archived to focus CI resources solely on dissertation PDF generation:

- **ci.yml** - Main CI with full test suite
- **RWP-ci.yml** - Library / RWP core CI
- **pages.yml** - GitHub Pages deployment
- **theory.yml** - Theory exports + dashboard update
- **paper-figs.yml** - Non-Hopf paper figures + PDF build
- **manual-analysis.yml** - Manual analysis runs
- **gp-demo.yml** - GP ringing demo
- **sims.yml** - Simulation smoke tests
- **sim-validate.yml** - Manual simulation validation
- **poison-detection-test.yml** - Poison detection tests

### October 2025 - Initial Cleanup (Batch 1)

Superseded or deprecated workflows:

- **pandoc-build.yml** - Legacy Pandoc build workflow
- **build-papers.yml** - Legacy paper build workflow
- **dissertation-build.yml** (old version) - Paused during analysis integration
- **ci-core.yml** - Deprecated core CI workflow

## Why Archive?

Workflows were archived to:
1. Simplify GitHub Actions UI for dissertation-focused development
2. Reduce CI complexity and resource usage
3. Prevent confusion from multiple overlapping workflows
4. Preserve historical workflow definitions for future reference

## Restoration

To restore archived workflows for future use:
1. Move the desired YAML file back to `.github/workflows/`
2. Review and update any deprecated GitHub Actions syntax or dependencies
3. Update trigger conditions (branches, paths) as needed
4. Test thoroughly before enabling on main branch

## Notes

- The new `dissertation-build.yml` (active) is a complete rewrite, not the same as the archived version
- All archived workflows remain functional and can be restored with minimal changes
- Consider restoring `ci.yml` or `pages.yml` if expanding beyond dissertation work
