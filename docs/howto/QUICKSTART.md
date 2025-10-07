# Resonance Geometry — Quickstart

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
pip install torch transformers datasets scikit-learn scipy matplotlib
```

## 2) Run the toy SU(2) sim (already in repo)

```bash
python -m rg.validation.hysteresis_sweep --lam 1.0 --gamma 0.5 --eta_min 0.2 --eta_max 5.0 --eta_steps 41 --alpha 0.6 --beta 0.02 --skew 0.12 --mi_window 30 --mi_ema 0.1
```

## 3) Evaluate λ_max on TruthfulQA (CPU ok)

```bash
python -m rg.llm.eval_truthfulqa_lambda --model gpt2 --n_samples 50 --out results/quick_truthfulqa
```

**Outputs**

- `metrics.json` — ROC-AUC of λ_max vs baselines
- `predictions.csv`
- `roc_pr_curves.png`

## 4) Null controls

```bash
python -m rg.llm.null_controls --model gpt2 --n_samples 50 --out results/null_controls
```

## 5) Build PDFs and a paper zip (via Actions)

Push to GitHub; CI will produce PDFs and upload a ZIP artifact with paper + figures.

**Note:** This diagnostic is theory-motivated and needs validation. Please compare λ_max against entropy/margin and share results via the replication issue template.
