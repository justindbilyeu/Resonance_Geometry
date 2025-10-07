import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from scipy import stats  # noqa: F401 imported for potential downstream analysis

from rg.llm.geom_monitor import (
    procrustes_R,
    mat_log_ortho,
    corr_operator,
    fisher_diag,
    lambda_max,
)


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
        rows.append(
            {
                "question": item["question"],
                "correct": correct,
                "wrong": wrong,
                "category": item.get("category", "general"),
            }
        )
    return rows


def compute_geom(hidden_states, logits, eta, lam, gamma, layer_idx=-3, k=64):
    H_l = hidden_states[layer_idx][0]
    H_l1 = hidden_states[layer_idx + 1][0]
    # connection proxy
    try:
        R = procrustes_R(H_l, H_l1, k=k)
        Omega = mat_log_ortho(R)
        omega_norm = float(Omega.norm().item())
    except Exception:
        omega_norm = 0.0

    # coherence operator from recent layers
    try:
        Ls = [hidden_states[i][0] for i in range(max(0, layer_idx - 2), layer_idx + 1)]
        Ck = corr_operator(Ls, k=k)
    except Exception:
        Ck = torch.eye(k) * 0.1

    f_diag = fisher_diag(logits[0], H_l1)
    lam_max_val = lambda_max(
        Ck,
        f_diag,
        gamma=gamma,
        c_norm=0.1,
        omega_norm=omega_norm,
        eta=eta,
        lam=lam,
    )
    return lam_max_val, omega_norm, f_diag


def entropy_margin(logits):
    probs = F.softmax(logits[0, -1, :], dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    top2 = torch.topk(logits[0, -1, :], k=2).values
    margin = (top2[0] - top2[1]).item()
    return entropy, margin


def eval_sample(model, tok, row, eta, lam, gamma, layer_idx):
    choices = [row["correct"]] + row["wrong"]
    best = {"logprob": -1e9}
    for idx, ans in enumerate(choices):
        prompt = f"Q: {row['question']}\nA: {ans}"
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        logits = out.logits
        # score answer span
        ans_ids = tok(ans, return_tensors="pt").input_ids[0]
        n_ans = len(ans_ids)
        start = inputs.input_ids.shape[1] - n_ans
        lps = F.log_softmax(logits[0, start:, :], dim=-1)
        mean_lp = lps.gather(1, ans_ids.unsqueeze(1)).mean().item()

        if mean_lp > best["logprob"]:
            lam_max_val, omega_norm, fdiag = compute_geom(
                out.hidden_states, logits, eta, lam, gamma, layer_idx=layer_idx
            )
            ent, mar = entropy_margin(logits)
            best = dict(
                idx=idx,
                logprob=mean_lp,
                lambda_max=lam_max_val,
                omega_norm=omega_norm,
                entropy=ent,
                margin=mar,
            )
    is_correct = best["idx"] == 0
    return is_correct, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--eta", type=float, default=1.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--layer", type=int, default=-3)
    ap.add_argument("--out", type=str, default="results/truthfulqa_lambda")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = load_truthfulqa_mc(args.n_samples)
    y_true, lams, ents, margs = [], [], [], []
    preds = []

    for i, row in enumerate(data):
        ok, diag = eval_sample(model, tok, row, args.eta, args.lam, args.gamma, args.layer)
        y_true.append(0 if ok else 1)
        lams.append(diag["lambda_max"])
        ents.append(diag["entropy"])
        margs.append(-diag["margin"])
        preds.append({"i": i, "correct": bool(ok), **diag})

    # metrics
    import csv

    with open(outdir / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        w.writeheader()
        w.writerows(preds)

    auc_l = roc_auc_score(y_true, lams)
    auc_e = roc_auc_score(y_true, ents)
    auc_m = roc_auc_score(y_true, margs)

    summary = {
        "auc_lambda": auc_l,
        "auc_entropy": auc_e,
        "auc_margin": auc_m,
        "n": len(y_true),
        "hallucination_rate": float(np.mean(y_true)),
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # ROC/PR plots
    fpr_l, tpr_l, _ = roc_curve(y_true, lams)
    fpr_e, tpr_e, _ = roc_curve(y_true, ents)
    fpr_m, tpr_m, _ = roc_curve(y_true, margs)
    prec_l, rec_l, _ = precision_recall_curve(y_true, lams)
    prec_e, rec_e, _ = precision_recall_curve(y_true, ents)
    prec_m, rec_m, _ = precision_recall_curve(y_true, margs)

    def pr_auc(prec, rec):
        return auc(rec, prec)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr_l, tpr_l, label=f"λ_max (AUC={auc_l:.3f})")
    ax[0].plot(fpr_e, tpr_e, label=f"Entropy (AUC={auc_e:.3f})")
    ax[0].plot(fpr_m, tpr_m, label=f"Margin (AUC={auc_m:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax[0].legend()
    ax[0].set_title("ROC")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")

    ax[1].plot(rec_l, prec_l, label=f"λ_max (AUC={pr_auc(prec_l, rec_l):.3f})")
    ax[1].plot(rec_e, prec_e, label=f"Entropy (AUC={pr_auc(prec_e, rec_e):.3f})")
    ax[1].plot(rec_m, prec_m, label=f"Margin (AUC={pr_auc(prec_m, rec_m):.3f})")
    ax[1].legend()
    ax[1].set_title("Precision–Recall")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    fig.tight_layout()
    plt.savefig(outdir / "roc_pr_curves.png", dpi=220)
    plt.close(fig)

    print("\nSummary:", json.dumps(summary, indent=2))
    print(
        f"\nOutputs in: {outdir}\n- predictions.csv\n- metrics.json\n- roc_pr_curves.png"
    )


if __name__ == "__main__":
    main()
