import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from rg.llm.geom_monitor import corr_operator, lambda_max


def load_truthfulqa_mc(n=None):
    ds = load_dataset("truthful_qa", "multiple_choice")["validation"]
    rows = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]
        correct = choices[labels.index(1)]
        wrong = [c for c, l in zip(choices, labels) if l == 0]
        rows.append({"q": item["question"], "c": correct, "w": wrong})
    return rows


def scrambled_layers(hidden_states):
    idx = list(range(len(hidden_states)))
    np.random.shuffle(idx)
    return [hidden_states[i] for i in idx]


def gaussian_like(hidden_states):
    H = hidden_states[-3][0]
    noise = torch.randn_like(H) * H.std()
    return [noise.clone() for _ in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--eta", type=float, default=1.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="results/null_controls")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    data = load_truthfulqa_mc(args.n_samples)

    def score_variant(variant):
        y, scores = [], []
        for row in data:
            prompt = f"Q: {row['q']}\nA: {row['c']}"
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = list(out.hidden_states)

            if variant == "real":
                layers = [hs[-5][0], hs[-4][0], hs[-3][0]]
            elif variant == "scramble":
                hs2 = scrambled_layers(hs)
                layers = [hs2[-5][0], hs2[-4][0], hs2[-3][0]]
            else:
                layers = gaussian_like(hs)

            try:
                C = corr_operator(layers, k=64)
                lmax = lambda_max(
                    C,
                    fisher_val=0.5,
                    gamma=args.gamma,
                    c_norm=0.1,
                    omega_norm=0.0,
                    eta=args.eta,
                    lam=args.lam,
                )
            except Exception:
                lmax = 0.0

            y.append(np.random.randint(0, 2))
            scores.append(lmax)
        return roc_auc_score(y, scores)

    auc_real = score_variant("real")
    auc_scr = score_variant("scramble")
    auc_gau = score_variant("gauss")

    res = {
        "auc_lambda_real_vs_random": float(auc_real),
        "auc_lambda_scrambled_vs_random": float(auc_scr),
        "auc_lambda_gaussian_vs_random": float(auc_gau),
    }
    (outdir / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
