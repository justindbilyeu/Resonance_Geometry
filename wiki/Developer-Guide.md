# Developer Guide — Contributing to Resonance Geometry

This guide explains how to collaborate effectively on Resonance Geometry across both research streams. It covers environment setup, coding standards, documentation expectations, and review workflows.

## 🧭 Repository Layout
- `src/` — core resonance geometry libraries shared by both streams
- `rg_empirical/` — runnable pipelines, datasets, and experiment orchestration
- `experiments/` — curated notebooks and scripts grouped by research theme
- `docs/` — formal documentation, including the dissertation artifacts
- `wiki/` — GitHub wiki source files mirrored in this repository

## 🛠 Environment Setup
1. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```
2. Configure pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. (Optional) Enable CUDA acceleration by installing the appropriate PyTorch build.

## 📐 Coding Standards
- Follow the formatting enforced by `ruff` and `black` (run `make lint`).
- Prefer functional decomposition with clear docstrings describing resonance-specific terminology.
- Avoid introducing try/except blocks around imports (per project guidelines).
- Write deterministic tests using fixtures in `tests/conftest.py` when possible.

## ✅ Testing Matrix
| Command | Purpose |
|---------|---------|
| `pytest` | Run the full unit/integration suite |
| `make lint` | Execute static analysis (`ruff`, `black`, `mypy` when configured) |
| `make docs` | Build documentation and ensure Sphinx configuration passes |
| `python -m rg_empirical.run --config <file>` | Launch experiment workflows for manual validation |

## 🧾 Documentation Workflow
- Update the relevant wiki page when introducing new experiments or metrics.
- Keep changelog entries synchronized with `docs/CHANGELOG.md`.
- Use the stream tags (`[Holonomic Memory]`, `[Geometric Plasticity]`) in pull request titles for clarity.

## 🔄 Review Process
1. Open a descriptive issue or discussion thread for significant changes.
2. Submit a pull request referencing the issue and include:
   - Summary of modifications
   - Test evidence
   - Impact on both research streams (if applicable)
3. Request review from maintainers listed in `CODEOWNERS`.
4. Address feedback promptly; keep discussions focused and respectful.

## 🤝 Community Standards
- Adhere to the project [Code of Conduct](../CODE_OF_CONDUCT.md).
- Share reproducible artifacts (configs, seeds, datasets) when reporting results.
- Celebrate cross-stream collaboration—insights from one stream often inform the other.

Need more detail? Reach out on [Discussions](https://github.com/justindbilyeu/Resonance_Geometry/discussions) or mention maintainers directly in issues.
