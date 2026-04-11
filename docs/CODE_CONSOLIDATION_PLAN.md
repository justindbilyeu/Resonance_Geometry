# Phase 3.1: Code Package Consolidation - Analysis & Plan

## Current State (4 Package Roots)

### 1. `src/resonance_geometry/` ✅ (Canonical per setup.py)
```
src/resonance_geometry/
├── __init__.py
├── core/
├── hallucination/ (adaptive_gain.py, phase_dynamics.py)
├── utils/
└── visualization/
```

### 2. `resonance_geometry/` (Root-level duplicate)
```
resonance_geometry/
├── __init__.py
├── state_vector.py
├── forbidden.py
└── fractals.py
```

### 3. `rg/` (Root-level)
```
rg/
├── __init__.py
├── llm/ (eval_truthfulqa_lambda, geom_monitor, null_controls)
├── sims/ (meta_flow_min_pair_v2)
└── validation/ (6 files: phase_boundary_fit, hysteresis_sweep, etc.)
```

### 4. `rg_empirical/` (Root-level)
```
rg_empirical/
├── __init__.py
├── run_truthfulqa_lambda.py
└── laplacian_lambda.py
```

## Import Analysis

### Who imports what:
- **Tests** → `from resonance_geometry.hallucination...` (src/)
- **Tests** → `from rg.validation...` (root-level rg/)
- **rg/sims/** → `from resonance_geometry.hallucination...` (src/)
- **Scripts** → `from src.resonance_geometry...` (explicit src prefix)
- **Scripts** → `from src.f_ai...` (f_ai module)

### setup.py Configuration:
```python
packages=find_packages(where="src")
package_dir={"": "src"}
```
**This means**: When installed, `src/resonance_geometry` becomes `import resonance_geometry`

## Consolidation Strategy

### Option A: Full Consolidation (RECOMMENDED)
Move all code into `src/` to match setup.py:

1. **Merge resonance_geometry/ → src/resonance_geometry/core/**
   ```
   resonance_geometry/state_vector.py → src/resonance_geometry/core/state_vector.py
   resonance_geometry/forbidden.py → src/resonance_geometry/core/forbidden.py
   resonance_geometry/fractals.py → src/resonance_geometry/core/fractals.py
   ```

2. **Move rg/ → src/rg/**
   ```
   rg/* → src/rg/* (entire directory)
   ```

3. **Move rg_empirical/ → src/rg_empirical/**
   ```
   rg_empirical/* → src/rg_empirical/* (entire directory)
   ```

### Required Import Updates:

**After consolidation, imports should work as:**
- `from resonance_geometry.core import state_vector` (for moved files)
- `from resonance_geometry.hallucination import phase_dynamics` (already works)
- `from rg.validation import phase_boundary_fit` (works via setup.py)
- `from rg_empirical import run_truthfulqa_lambda` (works via setup.py)

**Scripts using explicit `from src.` prefix:**
- Can keep working (src/ on PYTHONPATH in dev mode)
- OR update to standard imports (preferred for installed package)

## Risks & Mitigations

### HIGH RISK:
1. **Breaking imports** - Tests may fail if imports incorrect
2. **CI/CD breakage** - Workflows may use specific import paths
3. **Active development** - Other branches may conflict

### MITIGATIONS:
1. **Do in separate commit** - Easy to revert
2. **Update imports systematically** - Use grep to find all
3. **Test after consolidation** - Run tests to verify
4. **Document changes** - Clear commit message

## Alternative: Symlinks (Lower Risk)

Instead of moving code, create symlinks:
```bash
ln -s src/resonance_geometry resonance_geometry
ln -s src/rg rg  
ln -s src/rg_empirical rg_empirical
```

**Pros**: Backwards compatible, no import changes needed
**Cons**: Messy, doesn't actually clean up structure

## Recommendation

**PROCEED WITH OPTION A** but in careful steps:

1. **First**: Move rg_empirical (smallest, least used)
2. **Second**: Move rg/ (moderate risk)
3. **Third**: Merge resonance_geometry/ into src/resonance_geometry/core/
4. **After each**: Check for broken imports, update as needed
5. **Final**: Run full test suite

**Alternative recommendation:** STOP HERE and document this for the user to review before proceeding, since this is the highest-risk change.
