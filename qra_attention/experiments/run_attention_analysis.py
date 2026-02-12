"""
Run Attention Analysis

Loads a DistilBERT (baseline) or kernel-patched variant (RFF), loads checkpoint
weights, and computes attention-distribution metrics over a subset of IMDb test set.

Important:
- We force *eager* attention so `output_attentions=True` returns attention weights
  (SDPA often does not support that).
- We read attentions from `outputs.attentions` (no hooks).
"""

from __future__ import annotations

import os
import argparse
import json
from typing import List, Dict, Any

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
from tqdm import tqdm

from qra_attention.patching import patch_distilbert_attention
from qra_attention.experiments.metrics_attention import (
    compute_entropy,
    compute_distance_bias,
    compute_gini,
    plot_metric_distribution,
)


def _load_checkpoint_weights(model: torch.nn.Module, model_path: str, device: torch.device) -> None:
    """Load checkpoint weights (safetensors preferred, .bin fallback)."""
    state_dict_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(state_dict_path):
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")

    if not os.path.exists(state_dict_path):
        print(f"⚠ Warning: No checkpoint found at {model_path} (expected model.safetensors or pytorch_model.bin)")
        return

    if state_dict_path.endswith(".bin"):
        state_dict = torch.load(state_dict_path, map_location=device)
    else:
        from safetensors.torch import load_file

        state_dict = load_file(state_dict_path, device=str(device))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("✓ Loaded checkpoint weights")
    if missing:
        print(f"  - Missing keys: {len(missing)}")
    if unexpected:
        print(f"  - Unexpected keys: {len(unexpected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--is_rff", action="store_true", help="Whether this is an RFF (kernel-patched) model")
    parser.add_argument("--output_dir", type=str, default="analysis/attention", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of IMDb test samples")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    # RFF config (only used when --is_rff)
    parser.add_argument("--alpha", type=float, default=0.9, help="RFF alpha (blend factor)")
    parser.add_argument("--num_features", type=int, default=128, help="RFF num_features")
    parser.add_argument("--sigma", type=float, default=2.0, help="RFF sigma")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load base model forcing eager attention (so attentions are available)
    print("Loading base model (forcing eager attention)...")
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            attn_implementation="eager",
        )
    except TypeError:
        # Older transformers: fall back to config flag
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model.config.attn_implementation = "eager"

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # 2) Patch if RFF
    if args.is_rff:
        print("Patching with kernel attention...")
        kernel_config = {"num_features": args.num_features, "sigma": args.sigma, "normalize": True}
        model = patch_distilbert_attention(
            model,
            layers_to_patch=[4, 5],
            kernel_config=kernel_config,
            alpha=args.alpha,
        )

    # 3) Load checkpoint weights after patching (structure must match)
    print(f"Loading checkpoint weights from {args.model_path}...")
    _load_checkpoint_weights(model, args.model_path, device)

    model.to(device)
    model.eval()

    model_type = "rff" if args.is_rff else "baseline"
    layers_to_analyze: List[int] = [-2, -1] if args.is_rff else [4, 5]


    # 4) Load data
    print(f"Loading {args.num_samples} samples from IMDb test set...")
    dataset = load_dataset("imdb", split=f"test[:{args.num_samples}]")

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 5) Collect metrics from outputs.attentions (no hooks)
    print("Extracting attention weights (via outputs.attentions)...")

    all_entropies = []
    all_biases = []
    all_ginis = []

    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_dataset), args.batch_size)):
            batch = tokenized_dataset[i : i + args.batch_size]
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            outputs = model(**inputs, output_attentions=True, return_dict=True)

            attentions = outputs.attentions
            if i == 0:
                print(f"DEBUG: outputs.attentions length = {len(attentions)}")
            if attentions is None:
                raise RuntimeError(
                    "Model did not return attentions. Ensure eager attention is enabled and output_attentions=True."
                )

            n_attn = len(attentions)

            for layer_idx in layers_to_analyze:
                # Support negative indices (-1, -2, ...)
                idx = layer_idx if layer_idx >= 0 else (n_attn + layer_idx)

                if idx < 0 or idx >= n_attn:
                    raise RuntimeError(
                        f"Requested attention layer index {layer_idx} (resolved {idx}), "
                        f"but model returned only {n_attn} attention tensors."
                    )

                attn = attentions[idx]  # (b, heads, seq, seq)
                if attn is None:
                    continue
                all_entropies.append(compute_entropy(attn).cpu())
                all_biases.append(compute_distance_bias(attn).cpu())
                all_ginis.append(compute_gini(attn).cpu())

    if not all_entropies:
        raise RuntimeError("No attention tensors collected. Check model/patching and attention implementation.")

    # 6) Aggregate
    entropies = torch.cat(all_entropies, dim=0)
    biases = torch.cat(all_biases, dim=0)
    ginis = torch.cat(all_ginis, dim=0)

    # 7) Stats
    stats = {
        "entropy": {
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
            "min": float(entropies.min()),
            "max": float(entropies.max()),
        },
        "distance_bias": {
            "mean": float(biases.mean()),
            "std": float(biases.std()),
            "min": float(biases.min()),
            "max": float(biases.max()),
        },
        "gini": {
            "mean": float(ginis.mean()),
            "std": float(ginis.std()),
            "min": float(ginis.min()),
            "max": float(ginis.max()),
        },
        "num_samples": int(args.num_samples),
        "layers_analyzed": layers_to_analyze,
        "model_type": model_type,
    }

    print("\n" + "=" * 60)
    print("ATTENTION METRICS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_type.upper()}")
    print(f"Layers: {layers_to_analyze}")
    print(f"Samples: {args.num_samples}")
    print("-" * 60)
    print(f"Entropy:       {stats['entropy']['mean']:.4f} ± {stats['entropy']['std']:.4f}")
    print(f"               [{stats['entropy']['min']:.4f}, {stats['entropy']['max']:.4f}]")
    print(f"Distance Bias: {stats['distance_bias']['mean']:.4f} ± {stats['distance_bias']['std']:.4f}")
    print(f"               [{stats['distance_bias']['min']:.4f}, {stats['distance_bias']['max']:.4f}]")
    print(f"Gini Index:    {stats['gini']['mean']:.4f} ± {stats['gini']['std']:.4f}")
    print(f"               [{stats['gini']['min']:.4f}, {stats['gini']['max']:.4f}]")
    print("=" * 60)

    # 8) Plots
    print("\nGenerating plots...")
    plot_metric_distribution(entropies, "Entropy", save_path=os.path.join(args.output_dir, f"{model_type}_entropy.png"))
    plot_metric_distribution(biases, "Distance Bias", save_path=os.path.join(args.output_dir, f"{model_type}_bias.png"))
    plot_metric_distribution(ginis, "Gini Index", save_path=os.path.join(args.output_dir, f"{model_type}_gini.png"))

    # 9) Save stats
    stats_file = os.path.join(args.output_dir, f"{model_type}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("\n✓ Analysis complete!")
    print(f"✓ Results saved to {args.output_dir}")
    print(f"  - {model_type}_stats.json")
    print(f"  - {model_type}_*.png")


if __name__ == "__main__":
    main()
