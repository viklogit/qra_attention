"""
Robustness Evaluation for Transformer Models.

Evaluates performance under common text perturbations:
- Word Dropout: Randomly removing tokens.
- Synonym Swap: Replacing words with WordNet synonyms.
- Local Shuffling: Perturbing word order.

Key fixes vs earlier versions:
- Always include labels in tokenized dataset (Trainer expects 'labels').
- Disable HF dataset caching for perturbations (prevents weird sample count issues).
- Deterministic RFF loading: base -> patch -> load checkpoint (never direct-load RFF).
- Explicit binary F1.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import numpy as np
import torch
import nltk
from nltk.corpus import wordnet
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
import evaluate

from qra_attention.patching import patch_distilbert_attention


# ----------------------------
# NLTK setup + perturbations
# ----------------------------

def setup_nltk() -> None:
    """Ensure required NLTK assets are available."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


def apply_word_dropout(text: str, p: float = 0.05) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    kept = [w for w in words if random.random() > p]
    if not kept:
        return random.choice(words)
    return " ".join(kept)


def apply_synonym_swap(text: str, p: float = 0.1) -> str:
    words = text.split()
    new_words = words.copy()

    for i, w in enumerate(words):
        if random.random() >= p:
            continue

        syns = []
        for syn in wordnet.synsets(w):
            for l in syn.lemmas():
                cand = l.name().replace("_", " ")
                if cand.lower() != w.lower():
                    syns.append(cand)

        if syns:
            new_words[i] = random.choice(syns)

    return " ".join(new_words)


def apply_local_shuffling(text: str, window: int = 3) -> str:
    words = text.split()
    if len(words) < 2:
        return text

    new_words = words.copy()
    w = max(2, int(window))

    for i in range(len(words) - w + 1):
        sub = new_words[i : i + w]
        random.shuffle(sub)
        new_words[i : i + w] = sub

    return " ".join(new_words)


# ----------------------------
# Model loading helpers
# ----------------------------

def _load_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    safep = os.path.join(model_dir, "model.safetensors")
    binp = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safep):
        from safetensors.torch import load_file
        return load_file(safep)
    if os.path.exists(binp):
        return torch.load(binp, map_location="cpu")

    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in: {model_dir}")


def _auto_detect_kernel_cfg(model_dir: str, default_num_features: int, default_sigma: float, default_alpha: float):
    kernel_config = {"num_features": default_num_features, "sigma": default_sigma, "normalize": True}
    alpha = default_alpha

    parent_dir = os.path.dirname(model_dir)
    for f in os.listdir(parent_dir):
        if f.startswith("seed_") and f.endswith(".json"):
            try:
                with open(os.path.join(parent_dir, f), "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                if "num_features" in data:
                    kernel_config["num_features"] = int(data["num_features"])
                    kernel_config["sigma"] = float(data.get("sigma", kernel_config["sigma"]))
                    alpha = float(data.get("alpha", alpha))
                    print(f"Auto-detected kernel config: {kernel_config}, alpha: {alpha}")
                    break
            except Exception:
                pass

    return kernel_config, alpha

def remap_layernorm_keys(state_dict):
    # Map beta/gamma -> bias/weight for embedding LayerNorm
    g = "distilbert.embeddings.LayerNorm.gamma"
    b = "distilbert.embeddings.LayerNorm.beta"
    w = "distilbert.embeddings.LayerNorm.weight"
    bi = "distilbert.embeddings.LayerNorm.bias"

    if g in state_dict and w not in state_dict:
        state_dict[w] = state_dict.pop(g)
    if b in state_dict and bi not in state_dict:
        state_dict[bi] = state_dict.pop(b)

    return state_dict

def load_model_for_eval(
    model_path: str,
    num_features: int,
    sigma: float,
    alpha: float,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer, bool]:
    """
    Loads either baseline or RFF model for evaluation.
    Returns: (model, tokenizer, is_rff)
    """
    model_dir = os.path.abspath(model_path)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print(f"Loading model and tokenizer from {model_dir}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    is_rff = "rff" in model_dir.lower()

    if is_rff:
        print("Detected RFF model - using manual reconstruction (base -> patch -> load weights)...")
        kernel_config, alpha_detected = _auto_detect_kernel_cfg(model_dir, num_features, sigma, alpha)
        alpha = alpha_detected

        # 1) Load base model (architecture)
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        # 2) Patch architecture
        model = patch_distilbert_attention(
            model,
            layers_to_patch=[4, 5],
            alpha=alpha,
            kernel_config=kernel_config,
        )

        # 3) Load checkpoint weights
        state_dict = _load_state_dict(model_dir)
        state_dict = remap_layernorm_keys(state_dict)
        load_info = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", load_info.missing_keys)
        print("Unexpected keys:", load_info.unexpected_keys)

        # Sanity checks
        missing = list(load_info.missing_keys)
        unexpected = list(load_info.unexpected_keys)

        # For a properly saved RFF checkpoint, kernel_attention keys should NOT be unexpected.
        kernel_unexpected = [k for k in unexpected if "kernel_attention" in k]
        if kernel_unexpected:
            print("⚠️  ERROR: kernel_attention keys were UNEXPECTED. This means architecture didn't match checkpoint.")
            print("First few unexpected kernel keys:", kernel_unexpected[:5])
            raise RuntimeError("RFF checkpoint did not load into patched architecture correctly.")

        print(f"✓ Loaded RFF checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
        return model, tokenizer, True

    # Baseline model
    print("Detected BASELINE model - loading from distilbert-base-uncased + checkpoint weights...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    state_dict = _load_state_dict(model_dir)
    load_info = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)
    print(f"✓ Loaded baseline checkpoint (missing={len(load_info.missing_keys)}, unexpected={len(load_info.unexpected_keys)})")
    return model, tokenizer, False


# ----------------------------
# Robustness evaluator
# ----------------------------

@dataclass
class EvalConfig:
    max_length: int = 256
    eval_batch_size: int = 32


class RobustnessEvaluator:
    def __init__(self, model, tokenizer, compute_metrics_fn, seed: int = 42, cfg: Optional[EvalConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics_fn
        self.seed = seed
        self.cfg = cfg or EvalConfig()

    def evaluate_perturbation(
        self,
        dataset,
        perturbation_fn: Callable[[str], str],
        name: str,
        limit: int = 500,
    ) -> Dict[str, Any]:
        print(f"Evaluating robustness against: {name}...")

        limit = min(len(dataset), limit)

        # Build balanced indices: half label 0, half label 1
        half = limit // 2

        rng = random.Random(self.seed)

        all_idx0 = [i for i, y in enumerate(dataset["label"]) if y == 0]
        all_idx1 = [i for i, y in enumerate(dataset["label"]) if y == 1]
        rng.shuffle(all_idx0)
        rng.shuffle(all_idx1)

        idx0 = all_idx0[:half]
        idx1 = all_idx1[:half]

        # If limit is odd, add one extra from whichever class has more available
        indices = idx0 + idx1
        rng.shuffle(indices)

        subset = dataset.select(indices)

        # Apply perturbation (disable cache to avoid fingerprinting weirdness)
        def perturb(example):
            return {"text": perturbation_fn(example["text"]), "label": example["label"]}

        perturbed = subset.map(perturb, load_from_cache_file=False)

        # Tokenize + INCLUDE LABELS
        def tokenize(examples):
            out = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_length,
            )
            out["labels"] = examples["label"]
            return out

        tokenized = perturbed.map(tokenize, batched=True, load_from_cache_file=False)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        batch = {k: tokenized[k][:32] for k in ["input_ids", "attention_mask", "labels"]}
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        with torch.no_grad():
            out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = out.logits.argmax(dim=1).cpu()

        print("Batch labels:", torch.bincount(batch["labels"].cpu()).tolist())
        print("Batch preds :", torch.bincount(preds, minlength=2).tolist())

        # Print label distribution (sanity check)
        labels = tokenized["labels"]
        try:
            binc = torch.bincount(labels).tolist()
            print(f"Label counts (0/1): {binc}")
        except Exception:
            pass

        args = TrainingArguments(
            output_dir="results/temp_robustness",
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            remove_unused_columns=False,  # IMPORTANT: keep labels and anything else we pass
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=tokenized,
            compute_metrics=self.compute_metrics,
        )

        return trainer.evaluate()


# ----------------------------
# Main
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--limit", type=int, default=500, help="Samples per perturbation")
    parser.add_argument("--num_features", type=int, default=128, help="Number of random features (RFF)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Kernel bandwidth sigma")
    parser.add_argument("--alpha", type=float, default=0.9, help="Hybrid blending alpha")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size")
    parser.add_argument("--results_json", type=str, default=None, help="Path to save results as JSON")
    parser.add_argument("--eval_seed", type=int, default=42, help="Random seed for evaluation (subset selection and perturbations)")
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)
    print(f"✓ Set evaluation seed to {args.eval_seed}")

    os.makedirs("results", exist_ok=True)
    setup_nltk()

    model, tokenizer, is_rff = load_model_for_eval(
        args.model_path,
        num_features=args.num_features,
        sigma=args.sigma,
        alpha=args.alpha,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ Model moved to {device} and set to eval mode")


    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
        # Explicit binary F1 (IMDb is binary)
        f1v = f1.compute(predictions=preds, references=labels, average="binary")["f1"]
        return {"accuracy": acc, "f1": f1v}

    evaluator = RobustnessEvaluator(
        model,
        tokenizer,
        compute_metrics,
        seed=args.eval_seed,
        cfg=EvalConfig(max_length=args.max_length, eval_batch_size=args.eval_batch_size),
    )

    dataset = load_dataset("imdb", split="test")

    print("\n--- Clean Evaluation ---")
    clean_results = evaluator.evaluate_perturbation(dataset, lambda x: x, "Clean", limit=args.limit)
    print(f"Clean Results: {clean_results}")

    print("\n--- Word Dropout (5%) ---")
    dropout_results = evaluator.evaluate_perturbation(
        dataset, lambda x: apply_word_dropout(x, 0.05), "Word Dropout (5%)", limit=args.limit
    )
    print(f"Dropout Results: {dropout_results}")

    print("\n--- Synonym Swap (10%) ---")
    synonym_results = evaluator.evaluate_perturbation(
        dataset, lambda x: apply_synonym_swap(x, 0.1), "Synonym Swap (10%)", limit=args.limit
    )
    print(f"Synonym Results: {synonym_results}")

    print("\n--- Local Shuffling (window=3) ---")
    shuffle_results = evaluator.evaluate_perturbation(
        dataset, lambda x: apply_local_shuffling(x, 3), "Local Shuffling (window=3)", limit=args.limit
    )
    print(f"Shuffle Results: {shuffle_results}")

    print("\n" + "=" * 30)
    print("ROBUSTNESS SUMMARY")
    print("=" * 30)
    print(f"Clean Accuracy:    {clean_results['eval_accuracy']:.4f} | F1: {clean_results['eval_f1']:.4f}")
    print(
        f"Dropout Accuracy:  {dropout_results['eval_accuracy']:.4f} (Drop: {clean_results['eval_accuracy'] - dropout_results['eval_accuracy']:.4f})"
        f" | F1: {dropout_results['eval_f1']:.4f}"
    )
    print(
        f"Synonym Accuracy:  {synonym_results['eval_accuracy']:.4f} (Drop: {clean_results['eval_accuracy'] - synonym_results['eval_accuracy']:.4f})"
        f" | F1: {synonym_results['eval_f1']:.4f}"
    )
    print(
        f"Shuffle Accuracy:  {shuffle_results['eval_accuracy']:.4f} (Drop: {clean_results['eval_accuracy'] - shuffle_results['eval_accuracy']:.4f})"
        f" | F1: {shuffle_results['eval_f1']:.4f}"
    )

    if args.results_json:
        final_results = {
            "model_path": args.model_path,
            "is_rff": is_rff,
            "config": {
                "num_features": args.num_features,
                "sigma": args.sigma,
                "alpha": args.alpha
            },
            "metrics": {
                "clean": clean_results,
                "dropout": dropout_results,
                "synonym": synonym_results,
                "shuffle": shuffle_results
            }
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        print(f"\n✓ Saved results to {args.results_json}")


if __name__ == "__main__":
    main()
