# Quick Start — 15 Minute Onboarding

This quick start guide helps you install the core Resonance Geometry tooling and execute a minimal pipeline that exercises both holonomic memory and geometric plasticity features.

## ⏱ Prerequisites (5 minutes)
1. Clone the main repository and install dependencies:
   ```bash
   git clone https://github.com/justindbilyeu/Resonance_Geometry.git
   cd Resonance_Geometry
   pip install -r requirements.txt
   pip install -e .
   ```
2. (Optional) Enable extras for tutorials:
   ```bash
   pip install -r requirements-optional.txt
   ```
3. Verify the CLI is available:
   ```bash
   rg --help
   ```

## 🚀 Minimal Experiment (7 minutes)
Follow these steps to produce the reference resonance signature used in later tutorials.

```bash
# 1. Prepare dataset cache
python scripts/data/fetch_reference_dataset.py

# 2. Run the baseline resonance embedding experiment
python -m rg_empirical.run --config config/baselines/reference_resonance.yml

# 3. Summarize outputs
python scripts/analysis/summarize_resonance.py results/reference_resonance/
```

The summary script prints validation metrics and stores diagnostic plots in `results/reference_resonance/plots/`.

## 🔍 Validate Installation (3 minutes)
Run smoke tests to ensure the environment matches the validated baseline.

```bash
pytest tests/smoke -q
```

All tests should pass. Failures usually indicate missing dependencies or GPU drivers.

## ➡️ Next Steps
- Choose a research track on the [Home](Home) router page.
- For theory and historical context, read [Holonomic Memory Home](Holonomic-Memory-Home).
- For empirical work, continue with the [Interactive Tutorial](Interactive-Tutorial).

Need help? Reach out on the project [Discussions board](https://github.com/justindbilyeu/Resonance_Geometry/discussions).
