.RECIPEPREFIX := >
.PHONY: test smoke mapper-smoke

test:
>pytest -q

smoke:
>python -c "import json, os; os.makedirs('tmp', exist_ok=True); json.dump({'metadata':{},'bands':[{'band_id':'low','frequency_range':[0.0,1.0],'lambda_star':1.0,'hysteresis_area':0.5,'transition_sharpness':0.3,'mi_peaks':[],'trajectory':[0.0,0.5,1.0],'betti_trace':[0,1,0]}],'cross_band':{}}, open('tmp/fake.json','w'))"
>python -m tools.resonance_mapper.cli tmp/fake.json results/mapper --tda

mapper-smoke: smoke
>@ls -la results/mapper || true
>@ls -la figures || true
