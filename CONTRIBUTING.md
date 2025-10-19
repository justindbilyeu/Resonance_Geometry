# Contributing to Resonance Geometry

Thanks for contributing! This repository prioritizes **reproducibility** and **CI stability**.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
pytest -q
```

Style
- Code formatted by black (line length 100)
- Linted via ruff (auto-fix enabled)
- Run locally before committing:

```
pre-commit run -a
pytest -q
```



Tests
- Tests live in tests/
- Skip heavy tests in CI using RG_CI=1
- Generate temporary files on-the-fly; keep <1MB committed data

PRs
- Small, focused changes
- Describe purpose and acceptance criteria
- CI must be green before merge

⸻

— FILE: CODEOWNERS —
- @justindbilyeu
docs/** @justindbilyeu
experiments/** @justindbilyeu
scripts/** @justindbilyeu
.github/** @justindbilyeu

⸻

— README Badges (append to top of README.md) —


⸻

— Verification Commands —

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
pre-commit run -a
pytest -q
```

⸻

— Commit message template —

```
chore: repo hygiene (pre-commit, ruff/black, pins, gitignore, contributor docs)
```
