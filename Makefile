.RECIPEPREFIX := >

PY=python
FIG_DIR=papers/neurips/figures
PAPER_MD=docs/papers/neurips/A_Geometric_Theory_of_AI_Hallucination.md
PAPER_PDF=docs/papers/neurips/A_Geometric_Theory_of_AI_Hallucination.pdf

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
>  --resource-path=.:"papers/neurips/figures":"docs/papers/neurips" \
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
