.RECIPEPREFIX := >

PY=python
FIG_DIR=docs/papers/neurips/figures
PAPER_MD=docs/papers/neurips/manuscript.md
PAPER_PDF=docs/papers/neurips/pdfs/A_Geometric_Theory_of_AI_Hallucination.pdf

# Defaults: your “good” run
LAM?=1.0
GAMMA?=0.5
ETA_MIN?=0.2
ETA_MAX?=5.0
ETA_STEPS?=41
ALPHA?=0.6
BETA?=0.02
SKEW?=0.12
MI_WINDOW?=30
MI_EMA?=0.1

.PHONY: all figures pdf clean test smoke mapper-smoke sim-smoke

all: figures pdf

figures:
>$(PY) -m rg.validation.hysteresis_sweep --lam $(LAM) --gamma $(GAMMA) \
>  --eta_min $(ETA_MIN) --eta_max $(ETA_MAX) --eta_steps $(ETA_STEPS) \
>  --alpha $(ALPHA) --beta $(BETA) --skew $(SKEW) --mi_window $(MI_WINDOW) --mi_ema $(MI_EMA)
>-$(PY) -m rg.validation.phase_boundary_fit --gamma $(GAMMA) \
>  --lam_min 0.1 --lam_max 5.0 --lam_steps 11 \
>  --eta_min 0.2 --eta_max 5.0 --eta_steps 101 \
>  --alpha $(ALPHA) --beta $(BETA) --skew $(SKEW) --mi_window $(MI_WINDOW) --mi_ema $(MI_EMA)

pdf:
>pandoc $(PAPER_MD) \
>  --from markdown+tex_math_single_backslash \
>  --pdf-engine=xelatex \
>  -V geometry:margin=1in \
>  --resource-path=.:"docs/papers/neurips/figures":"docs/papers/neurips" \
>  --output $(PAPER_PDF)

clean:
>rm -f $(PAPER_PDF)

# Existing project targets

test:
>pytest -q

smoke:
>python -c "import json, os; os.makedirs('tmp', exist_ok=True); json.dump({'metadata':{},'bands':[{'band_id':'low','frequency_range':[0.0,1.0],'lambda_star':1.0,'hysteresis_area':0.5,'transition_sharpness':0.3,'mi_peaks':[],'trajectory':[0.0,0.5,1.0],'betti_trace':[0,1,0]}],'cross_band':{}}, open('tmp/fake.json','w'))"
>python -m tools.resonance_mapper.cli tmp/fake.json results/mapper --tda

mapper-smoke: smoke
>@ls -la results/mapper || true
>@ls -la figures || true

sim-smoke:
>$(PY) -m rg.validation.hysteresis_sweep --lam 1.0 --gamma 0.5 \
>  --eta_min 0.2 --eta_max 3.0 --eta_steps 21 \
>  --alpha 0.6 --beta 0.02 --skew 0.12 \
>  --mi_window 30 --mi_ema 0.1 \
>  --algebra so3 --antisym_coupling --noise_std 0.01 \
>  --mi_est corr --mi_scale 1.0
>$(PY) -m rg.validation.phase_boundary_fit --gamma 0.5 \
>  --lam_min 0.1 --lam_max 2.0 --lam_steps 5 \
>  --eta_min 0.2 --eta_max 3.0 --eta_steps 51 \
>  --alpha 0.6 --beta 0.02 --skew 0.12 --mi_window 30 --mi_ema 0.1 \
>  --algebra su2 --noise_std 0.0 --mi_est svd --mi_scale 1.0

.PHONY: theory-status
theory-status:
>python scripts/update_theory_status.py

.PHONY: jacobian-sweep fluency-probe
jacobian-sweep:
>python scripts/run_ringing_sweep.py --include-jacobian

fluency-probe:
>python scripts/run_fluency_probe.py

.PHONY: fluency-sweep fluency-status test-fluency fluency-all

fluency-sweep:
>python scripts/run_fluency_sweep.py

fluency-status:
>python scripts/update_theory_status.py

test-fluency:
>pytest -q tests/test_fluency_velocity.py

fluency-all: fluency-sweep fluency-status

.PHONY: ring-sweep ring-smoke

ring-sweep:
>python scripts/run_ringing_sweep.py

ring-smoke:
>pytest -q tests/experiments/test_forbidden_minimal.py
>pytest -q tests/topo_test/test_skeleton_runs.py
>pytest -q tests/test_ringing_detector.py

.PHONY: theory-ensure theory-exports theory-all
# Wolfram theory exports
theory-ensure:
>mkdir -p docs/data/theory docs/assets/figures

theory-exports: theory-ensure
>@if command -v wolframscript >/dev/null 2>&1; then \
>  echo "Running Wolfram theory exports..."; \
>  wolframscript -file docs/theory/theory_exports.wl || exit 1; \
>else \
>  echo "wolframscript not found; skipping theory exports."; \
>fi

theory-all: theory-exports
>@echo "Theory export pipeline completed (or skipped)."

.PHONY: dashboard pages-preview

dashboard:
>python scripts/update_theory_status.py
>python scripts/build_dashboard.py

pages-preview:
>python -m http.server -d docs 8080

# === Dissertation builds ===
DIS_YAML := docs/dissertation/dissertation.yml
DIS_BUILD_DIR := docs/dissertation/build
DIS_PDF := $(DIS_BUILD_DIR)/resonance_geometry_dissertation.pdf
DIS_HTML := $(DIS_BUILD_DIR)/resonance_geometry_dissertation.html

.PHONY: dissertation
dissertation: $(DIS_PDF)

$(DIS_PDF): $(DIS_YAML) \
docs/dissertation/00_prologue.md \
docs/dissertation/01_introduction.md \
docs/dissertation/02_foundations.md \
docs/dissertation/03_general_theory.md \
docs/dissertation/04_retrospective.md
>@mkdir -p $(DIS_BUILD_DIR)
>pandoc -d $(DIS_YAML)
>@echo "[make] Built $(DIS_PDF)"

.PHONY: dissertation-html
dissertation-html:
>@mkdir -p $(DIS_BUILD_DIR)
>pandoc \
  --metadata-file=$(DIS_YAML) \
  -t html5 \
  -o $(DIS_HTML) \
  docs/dissertation/00_prologue.md \
  docs/dissertation/01_introduction.md \
  docs/dissertation/02_foundations.md \
  docs/dissertation/03_general_theory.md \
  docs/dissertation/04_retrospective.md
>@echo "[make] Built $(DIS_HTML)"

.PHONY: dissertation-clean
dissertation-clean:
>rm -rf $(DIS_BUILD_DIR)
>@echo "[make] Cleaned dissertation build artifacts"


.PHONY: eq-scan fractal-alpha fractal-alpha-noplastic

eq-scan:
>python scripts/equilibrium_analysis.py

fractal-alpha:
>python scripts/fractal_dimension.py

# quick no-plastic run (alpha sweep bounds are in the script)
fractal-alpha-noplastic:
>PYTHONUNBUFFERED=1 python -c "import scripts.fractal_dimension as f; p=f.RGParams(plastic=False); f.fractal_sweep('alpha', __import__('numpy').linspace(0.25,0.65,17), p)"

.PHONY: paper-figs
paper-figs:
>mkdir -p docs/analysis docs/analysis/figures results/phase/traces
>$(PY) scripts/equilibrium_analysis.py \
>  --alpha-start 0.25 --alpha-stop 0.55 --alpha-steps 61 \
>  --out-csv docs/analysis/eigs_scan_alpha.csv \
>  --out-json docs/analysis/eigs_scan_summary.json \
>  --seed 1337
>$(PY) scripts/plot_eigs_scan.py \
>  --summary docs/analysis/eigs_scan_summary.json \
>  --out figures/eigenvalue_real_vs_alpha.png
>$(PY) scripts/plot_eigs_scan.py \
>  --summary docs/analysis/eigs_scan_summary.json \
>  --out docs/analysis/figures/eigenvalue_real_vs_alpha.png
>$(PY) scripts/generate_phase_traces.py \
>  --alphas 0.25 0.30 0.35 0.40 0.45 0.50 0.55 \
>  --outdir results/phase/traces \
>  --seed 1337

.PHONY: ci-smoke
ci-smoke:
>$(PY) scripts/ci_smoke.py

# === Equilibrium sweep targets for validation ===
.PHONY: sweep-narrow sweep-wide sweep-zoom test-asserts

# Narrow: 0.25–0.55 (core falsification window)
sweep-narrow:
>$(PY) scripts/equilibrium_analysis.py \
>  --alpha-start 0.25 --alpha-stop 0.55 --alpha-step 0.005 \
>  --seed 42 \
>  --out-csv docs/analysis/eigs_scan_alpha_narrow.csv \
>  --out-json docs/analysis/eigs_scan_summary_narrow.json \
>  --plot-svg

# Wide: 0.10–1.00 (PR #105 claim)
sweep-wide:
>$(PY) scripts/equilibrium_analysis.py \
>  --config configs/equilibrium_sweep.yaml \
>  --out-csv docs/analysis/eigs_scan_alpha.csv \
>  --out-json docs/analysis/eigs_scan_summary.json \
>  --plot-svg

# Zoom: 0.80–0.86 (dense grid around crossing)
sweep-zoom:
>$(PY) scripts/equilibrium_analysis.py \
>  --alpha-start 0.80 --alpha-stop 0.86 --alpha-step 0.001 \
>  --seed 42 --out-dir docs/analysis/zoom \
>  --export-csv --export-json --plot-svg

test-asserts:
>pytest -q
