import os, json, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from rg_empirical.laplacian_lambda import lambda_max_Lsym
from rg.validation.truthfulqa_labels import label_hallucination


def generate_text(model, tok, prompt, temperature=1.0, top_k=50, max_new_tokens=64, device="cuda"):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(gen_ids[0], skip_special_tokens=True)


def hidden_states_for_text(model, tok, text, device="cuda"):
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple length L+1; each [1, T, d]
    T = inputs.input_ids.shape[1]
    return [h[0, :T, :].float().cpu().numpy() for h in hs]


def first_crossing_layer(values, lam_ref):
    for idx, val in enumerate(values):
        if np.isfinite(val) and val > lam_ref:
            return idx
    return None


def summarize(records, out_dir, lam_ref):
    # Only samples with concrete labels (0 clean, 1 hallucinated)
    labeled = [r for r in records if r["label"] in (0, 1)]
    if not labeled:
        return {}

    y_true = np.array([r["label"] for r in labeled], dtype=int)
    y_score = np.array([r["lambda_max_base"] for r in labeled], dtype=float)

    # ROC-AUC (fallback if sklearn not available)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")

    # First crossing layers
    first_layers = []
    for r in labeled:
        l = first_crossing_layer(r["lambda_layers_base"], lam_ref)
        first_layers.append(None if l is None else int(l))

    # Intervention deltas
    deltas = []
    for r in labeled:
        if np.isfinite(r["lambda_max_base"]) and np.isfinite(r["lambda_max_intv"]):
            deltas.append(r["lambda_max_intv"] - r["lambda_max_base"])
    deltas = np.array(deltas) if len(deltas) else np.array([np.nan])

    summary = {
        "n_total": len(records),
        "n_labeled": len(labeled),
        "n_clean": int(np.sum(y_true == 0)),
        "n_hallu": int(np.sum(y_true == 1)),
        "lambda_ref_median_clean": float(lam_ref) if np.isfinite(lam_ref) else None,
        "auc": auc,
        "first_crossing_layer_counts": {
            "none": int(np.sum([l is None for l in first_layers])),
            "by_layer": {str(i): int(np.sum([l == i for l in first_layers])) for i in set([x for x in first_layers if x is not None])}
        },
        "intervention": {
            "mean_delta": float(np.nanmean(deltas)),
            "frac_negative": float(np.mean(deltas < 0)) if np.isfinite(deltas).any() else None
        }
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Compute λ_max(L_sym) per layer on TruthfulQA with baseline vs. temp/top-k intervention.")
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--n", type=int, default=100, help="number of TruthfulQA samples")
    ap.add_argument("--k", type=int, default=15, help="k-NN for graph")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--baseline_temp", type=float, default=1.0)
    ap.add_argument("--baseline_topk", type=int, default=50)
    ap.add_argument("--intervention_temp", type=float, default=0.3)
    ap.add_argument("--intervention_topk", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/truthfulqa_lambda")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading TruthfulQA (generation) validation split…")
    ds = load_dataset("truthful_qa", "generation")["validation"]

    print(f"Loading model {args.model} on {args.device}…")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_dir) / "records.jsonl"
    fout = open(out_jsonl, "w")

    records = []
    N = min(args.n, len(ds))
    print(f"Processing {N} samples…")

    for i in tqdm(range(N), desc="Samples"):
        ex = ds[i]
        q = ex["question"]
        correct = ex.get("correct_answers") or []
        incorrect = ex.get("incorrect_answers") or []

        # Generate outputs
        out_base = generate_text(model, tok, q, args.baseline_temp, args.baseline_topk, args.max_new_tokens, args.device)
        out_intv = generate_text(model, tok, q, args.intervention_temp, args.intervention_topk, args.max_new_tokens, args.device)

        # Label baseline output
        label = label_hallucination(out_base, correct, incorrect)  # 0 clean, 1 hallucinated, -1 borderline

        # Hidden states (prompt + output)
        text_base = q + "\n\n" + out_base
        text_intv = q + "\n\n" + out_intv
        hs_layers_base = hidden_states_for_text(model, tok, text_base, device=args.device)
        hs_layers_intv = hidden_states_for_text(model, tok, text_intv, device=args.device)

        lam_layers_base = [lambda_max_Lsym(H, k=args.k) for H in hs_layers_base]
        lam_layers_intv = [lambda_max_Lsym(H, k=args.k) for H in hs_layers_intv]

        lam_max_base = float(np.nanmax(lam_layers_base)) if len(lam_layers_base) else float("nan")
        lam_max_intv = float(np.nanmax(lam_layers_intv)) if len(lam_layers_intv) else float("nan")

        rec = {
            "id": i,
            "label": label,
            "lambda_layers_base": lam_layers_base,
            "lambda_layers_intv": lam_layers_intv,
            "lambda_max_base": lam_max_base,
            "lambda_max_intv": lam_max_intv,
            "question": q,
            "out_base": out_base,
            "out_intv": out_intv,
        }
        records.append(rec)
        fout.write(json.dumps(rec) + "\n")

    fout.close()

    # λ_ref from clean samples (median of max over layers)
    clean = [r["lambda_max_base"] for r in records if r["label"] == 0 and np.isfinite(r["lambda_max_base"])]
    lam_ref = float(np.median(clean)) if len(clean) else float("nan")

    summary = summarize(records, args.out_dir, lam_ref)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-sample to: {out_jsonl}")
    print(f"Wrote summary to:    {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
