# GP Ringing Demo (Synthetic)

Run a minimal, prereg-aligned synthetic demo of P1/P2:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r experiments/requirements.txt
python experiments/gp_ringing_demo.py
Outputs:
	•	figures/gp_demo/mi_timeseries.png
	•	figures/gp_demo/lambda_schedule.png
	•	figures/gp_demo/hysteresis_curve.png
	•	figures/gp_demo/summary.json (contains alpha_power_MI)


