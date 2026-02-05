"""
Robustness Evaluation for Transformer Models.

This module evaluates model performance under common text perturbations:
- Word Dropout: Randomly removing tokens.
- Synonym Swap: Replacing words with synonyms.
- Local Shuffling: Perturbing word order.
"""

import os
import random
import torch
import numpy as np
from typing import List, Callable, Dict, Any
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
from transformers import Trainer, TrainingArguments, set_seed, DistilBertForSequenceClassification, DistilBertTokenizer
from qra_attention.patching import patch_distilbert_attention

# Ensure NLTK data is available
def setup_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

def apply_word_dropout(text: str, p: float = 0.05) -> str:
    """Randomly remove words with probability p."""
    words = text.split()
    if len(words) <= 1:
        return text
    new_words = [w for w in words if random.random() > p]
    if not new_words:
        return random.choice(words)
    return " ".join(new_words)

def apply_synonym_swap(text: str, p: float = 0.1) -> str:
    """Replace words with synonyms with probability p."""
    words = text.split()
    new_words = words.copy()
    for i in range(len(words)):
        if random.random() < p:
            syns = []
            for syn in wordnet.synsets(words[i]):
                for l in syn.lemmas():
                    if l.name().lower() != words[i].lower():
                        syns.append(l.name().replace('_', ' '))
            if syns:
                new_words[i] = random.choice(syns)
    return " ".join(new_words)

def apply_local_shuffling(text: str, window: int = 3) -> str:
    """Perturb word order within a local window."""
    words = text.split()
    if len(words) < 2:
        return text
    
    new_words = words.copy()
    for i in range(len(words) - window + 1):
        # Shuffle a small window
        sub = new_words[i:i+window]
        random.shuffle(sub)
        new_words[i:i+window] = sub
    return " ".join(new_words)

class RobustnessEvaluator:
    """Helper to evaluate model performance on perturbed datasets."""
    
    def __init__(self, model, tokenizer, compute_metrics_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics_fn
        
    def evaluate_perturbation(
        self, 
        dataset, 
        perturbation_fn: Callable[[str], str],
        name: str,
        limit: int = 500
    ) -> Dict[str, Any]:
        """Evaluate model on a perturbed version of the dataset."""
        print(f"Evaluating robustness against: {name}...")
        
        # Select subset for speed
        subset = dataset.select(range(min(len(dataset), limit)))
        
        # Apply perturbation
        def perturb(example):
            example["text"] = perturbation_fn(example["text"])
            return example
            
        perturbed_subset = subset.map(perturb)
        
        # Tokenize perturbed subset
        def tokenize(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length if self.tokenizer.model_max_length < 1000 else 512
            )
            
        tokenized_perturbed = perturbed_subset.map(tokenize, batched=True)
        
        # Eval settings
        args = TrainingArguments(
            output_dir="results/temp_robustness",
            per_device_eval_batch_size=32,
            remove_unused_columns=True,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=tokenized_perturbed,
            compute_metrics=self.compute_metrics
        )
        
        results = trainer.evaluate()
        return results

if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    import numpy as np
    import evaluate
    
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate per perturbation")
    parser.add_argument("--num_features", type=int, default=128, help="Number of random features")
    parser.add_argument("--sigma", type=float, default=1.0, help="Kernel bandwidth sigma")
    parser.add_argument("--alpha", type=float, default=0.9, help="Hybrid blending alpha")
    args = parser.parse_args()
    
    setup_nltk()
    
    # 1. Setup paths
    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        # Check if it was meant to be in results
        alt_path = os.path.join("results", "rff", args.model_path)
        if os.path.exists(alt_path):
            model_path = os.path.abspath(alt_path)
        else:
            raise FileNotFoundError(f"Model path not found: {args.model_path}")

    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    # Auto-detect kernel config if possible
    kernel_config = {"num_features": args.num_features, "sigma": args.sigma, "normalize": True}
    alpha = args.alpha
    
    parent_dir = os.path.dirname(model_path)
    # Search for any seed_*.json in the parent directory
    import json
    for f in os.listdir(parent_dir):
        if f.startswith("seed_") and f.endswith(".json"):
            try:
                with open(os.path.join(parent_dir, f), "r") as jf:
                    data = json.load(jf)
                    if "num_features" in data:
                        kernel_config["num_features"] = data["num_features"]
                        kernel_config["sigma"] = data.get("sigma", 1.0)
                        alpha = data.get("alpha", 0.9)
                        print(f"Auto-detected kernel config: {kernel_config}, alpha: {alpha}")
                        break
            except:
                pass

    # Load base model structure
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    # 2. Patch if it's an RFF model
    if "rff" in model_path.lower():
        print(f"Detected RFF model. Patching with {kernel_config['num_features']} features...")
        model = patch_distilbert_attention(
            model, 
            layers_to_patch=[4, 5], 
            alpha=alpha,
            kernel_config=kernel_config
        )
    
    # 3. Load weights from the checkpoint
    print("Loading weights from checkpoint...")
    
    # Check for safetensors or bin
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    else:
        raise FileNotFoundError(f"No weight file (model.safetensors or pytorch_model.bin) found in {model_path}")
        
    # Map old names to new names if necessary (beta/gamma -> bias/weight)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace(".LayerNorm.gamma", ".LayerNorm.weight").replace(".LayerNorm.beta", ".LayerNorm.bias")
        new_state_dict[new_key] = v
    state_dict = new_state_dict
        
    print("Loading state dict (non-strict)...")
    load_info = model.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys:
        print(f"  Missing keys: {len(load_info.missing_keys)}")
    if load_info.unexpected_keys:
        print(f"  Unexpected keys: {len(load_info.unexpected_keys)}")
    
    def compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels)
        return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}
        
    evaluator = RobustnessEvaluator(model, tokenizer, compute_metrics)
    dataset = load_dataset("imdb", split="test")
    
    # 1. Clean Baseline
    print("\n--- Clean Evaluation ---")
    clean_results = evaluator.evaluate_perturbation(dataset, lambda x: x, "Clean", limit=args.limit)
    print(f"Clean Results: {clean_results}")
    
    # 2. Word Dropout
    print("\n--- Word Dropout (5%) ---")
    dropout_results = evaluator.evaluate_perturbation(dataset, lambda x: apply_word_dropout(x, 0.05), "Word Dropout (5%)", limit=args.limit)
    print(f"Dropout Results: {dropout_results}")
    
    # 3. Synonym Swap
    print("\n--- Synonym Swap ---")
    synonym_results = evaluator.evaluate_perturbation(dataset, lambda x: apply_synonym_swap(x, 0.1), "Synonym Swap", limit=args.limit)
    print(f"Synonym Results: {synonym_results}")
    
    # 4. Local Shuffling
    print("\n--- Local Shuffling ---")
    shuffle_results = evaluator.evaluate_perturbation(dataset, lambda x: apply_local_shuffling(x, 3), "Local Shuffling", limit=args.limit)
    print(f"Shuffle Results: {shuffle_results}")
    
    # Summary
    print("\n" + "="*30)
    print("ROBUSTNESS SUMMARY")
    print("="*30)
    print(f"Clean Accuracy:    {clean_results['eval_accuracy']:.4f}")
    print(f"Dropout Accuracy:  {dropout_results['eval_accuracy']:.4f} (Drop: {clean_results['eval_accuracy'] - dropout_results['eval_accuracy']:.4f})")
    print(f"Synonym Accuracy:  {synonym_results['eval_accuracy']:.4f} (Drop: {clean_results['eval_accuracy'] - synonym_results['eval_accuracy']:.4f})")
    print(f"Shuffle Accuracy:  {shuffle_results['eval_accuracy']:.4f} (Drop: {clean_results['eval_accuracy'] - shuffle_results['eval_accuracy']:.4f})")
