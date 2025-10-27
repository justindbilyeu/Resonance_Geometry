#!/usr/bin/env python3
"""
Extract Activations from LLMs

Extracts intermediate activations from transformer models on benchmark datasets
(TruthfulQA, HaluEval) for downstream curvature analysis.

Usage:
    python rg_empirical/extract_activations.py --model gpt2 --dataset truthfulqa --output results/activations/

Output:
    - activations_{model}_{dataset}.npz: Compressed activations per layer
    - metadata_{model}_{dataset}.json: Sample info, labels, prompts
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using mock mode")


def load_truthfulqa_subset(n_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Load TruthfulQA subset for activation extraction.

    Returns:
        List of dicts with 'question', 'best_answer', 'incorrect_answers'
    """
    # Placeholder - in real implementation, would load from HuggingFace datasets
    # For now, return mock data
    return [
        {
            'question': f"Sample question {i}",
            'best_answer': f"Correct answer {i}",
            'incorrect_answers': [f"Wrong {i}a", f"Wrong {i}b"],
            'category': 'mock'
        }
        for i in range(n_samples)
    ]


def extract_layer_activations(
    model,
    tokenizer,
    prompt: str,
    layer_indices: List[int]
) -> Dict[int, np.ndarray]:
    """
    Extract activations from specified layers during forward pass.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Input text
        layer_indices: Which layers to extract from

    Returns:
        Dict mapping layer_idx -> activations (seq_len, hidden_dim)
    """
    if not TRANSFORMERS_AVAILABLE:
        # Mock mode
        return {
            idx: np.random.randn(10, 768)
            for idx in layer_indices
        }

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Hook-based extraction
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output[0] is the hidden states tensor
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations[layer_idx] = hidden.detach().cpu().numpy()
        return hook

    # Register hooks
    hooks = []
    for idx in layer_indices:
        # Adjust for model architecture (GPT-2, GPT-J, LLaMA, etc.)
        try:
            layer_module = model.transformer.h[idx]  # GPT-2 style
        except AttributeError:
            try:
                layer_module = model.model.layers[idx]  # LLaMA style
            except AttributeError:
                print(f"Warning: Could not find layer {idx}, skipping")
                continue

        hook = layer_module.register_forward_hook(make_hook(idx))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def main():
    parser = argparse.ArgumentParser(description="Extract LLM activations")
    parser.add_argument('--model', default='gpt2', help='Model name (HuggingFace)')
    parser.add_argument('--dataset', default='truthfulqa', choices=['truthfulqa', 'halueval'])
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--layers', default='10,15,20', help='Comma-separated layer indices')
    parser.add_argument('--output', default='results/activations/', help='Output directory')
    parser.add_argument('--mock', action='store_true', help='Use mock data (no model loading)')
    args = parser.parse_args()

    print("=" * 60)
    print("Activation Extraction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.n_samples}")
    print(f"Layers: {args.layers}")
    print()

    # Parse layer indices
    layer_indices = [int(x.strip()) for x in args.layers.split(',')]

    # Load dataset
    print("Loading dataset...")
    if args.dataset == 'truthfulqa':
        samples = load_truthfulqa_subset(args.n_samples)
    else:
        print(f"Dataset {args.dataset} not yet implemented, using mock")
        samples = load_truthfulqa_subset(args.n_samples)

    print(f"Loaded {len(samples)} samples")

    # Load model and tokenizer
    if not args.mock and TRANSFORMERS_AVAILABLE:
        print(f"Loading model {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.eval()
        print("Model loaded")
    else:
        print("Using mock mode (no model loading)")
        model = None
        tokenizer = None

    # Extract activations
    print()
    print("Extracting activations...")

    all_activations = {layer_idx: [] for layer_idx in layer_indices}
    metadata = []

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{len(samples)}")

        prompt = sample['question']

        # Extract activations
        layer_acts = extract_layer_activations(model, tokenizer, prompt, layer_indices)

        # Store
        for layer_idx, acts in layer_acts.items():
            # Use last token activation as representative
            all_activations[layer_idx].append(acts[0, -1, :])  # (hidden_dim,)

        metadata.append({
            'sample_idx': i,
            'question': prompt,
            'best_answer': sample.get('best_answer', ''),
            'category': sample.get('category', 'unknown')
        })

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save activations as NPZ
    activations_dict = {
        f'layer_{idx}': np.array(acts)
        for idx, acts in all_activations.items()
    }

    output_npz = output_dir / f"activations_{args.model.replace('/', '_')}_{args.dataset}.npz"
    np.savez_compressed(output_npz, **activations_dict)
    print(f"Activations saved: {output_npz}")

    # Save metadata as JSON
    output_json = output_dir / f"metadata_{args.model.replace('/', '_')}_{args.dataset}.json"
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {output_json}")

    print()
    print("=" * 60)
    print("Extraction complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
