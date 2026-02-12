"""
Script to train DistilBERT with RFF Kernel Attention on IMDb.
Patches layers L4 and L5 with KernelSelfAttention.
"""

import os
import argparse
import numpy as np
import torch
import json
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)
import evaluate
from datasets import load_dataset

from qra_attention.experiments.config import ExperimentConfig
from qra_attention.patching import patch_distilbert_attention, freeze_layers

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels)
    
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

def main():
    parser = argparse.ArgumentParser(description="Train RFF Kernel DistilBERT on IMDb")
    parser.add_argument("--output_dir", type=str, default="results/rff", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=ExperimentConfig.batch_size, help="Batch size")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed, help="Random seed")
    parser.add_argument("--no_save_model", action="store_true", help="Do not save the model checkpoint")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    
    # RFF Specific Arguments
    parser.add_argument("--num_features", type=int, default=64, help="Number of random features")
    parser.add_argument("--sigma", type=float, default=1.0, help="RBF kernel bandwidth sigma")
    parser.add_argument("--alpha", type=float, default=0.9, help="Hybrid blending alpha (default: 0.9)")
    parser.add_argument("--learning_rate", type=float, default=ExperimentConfig.learning_rate, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training data to use")
    
    args = parser.parse_args()
    
    # 1. Setup
    config = ExperimentConfig()
    config.seed = args.seed
    set_seed(config.seed)
    
    if args.smoke_test:
        print("!!! RUNNING SMOKE TEST !!!")
        config.num_epochs = 1
        config.logging_dir = "logs/smoke_test_rff"
        # Don't override output_dir - respect argument from run_experiments.py
    
    print(f"Loading model: {config.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Configuration: RFF features={args.num_features}, sigma={args.sigma}, alpha={args.alpha}")
    
    # 2. Data Loading
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    # Handle train fraction
    if args.train_fraction < 1.0:
        num_train = len(dataset["train"])
        subset_size = int(num_train * args.train_fraction)
        print(f"Subsetting training data: {args.train_fraction} ({subset_size}/{num_train})")
        # Use seed for reproducible subsetting
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(subset_size))

    tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=config.max_length
        )
    
    print("Tokenizing dataset...")
    if args.smoke_test:
        dataset["train"] = dataset["train"].select(range(20))
        dataset["test"] = dataset["test"].select(range(10))
        print("  Truncated dataset for smoke test")
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Model Initialization
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=config.num_labels
    )
    
    # 4. Patching with Kernel Attention
    print("Patching attention layers...")
    kernel_config = {
        "num_features": args.num_features,
        "sigma": args.sigma,
        "normalize": True # FIX 3: Enable feature normalization
    }
    
    patched_model = patch_distilbert_attention(
        model, 
        kernel_config=kernel_config,
        layers_to_patch=[4, 5], # Patch last two layers
        alpha=args.alpha # FIX 1: Pass blending factor
    )
    
    # 5. Freezing
    print("Applying freezing constraints...")
    freeze_summary = freeze_layers(
        patched_model, 
        freeze_embeddings=config.freeze_embeddings, 
        freeze_layer_indices=list(config.freeze_layers)
    )
    
    print(f"Model Summary:")
    print(f"  Total params: {freeze_summary['frozen_params'] + freeze_summary['trainable_params']:,}")
    print(f"  Trainable: {freeze_summary['trainable_params']:,}")
    print(f"  Frozen: {freeze_summary['frozen_params']:,}")
    
    # 6. Training Setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=config.logging_dir,
        seed=config.seed,
        fp16=args.fp16,
        max_grad_norm=1.0,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    trainer = Trainer(
        model=patched_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    
    # Add metadata to results
    eval_results["kernel_type"] = "rff"
    eval_results["num_features"] = args.num_features
    eval_results["sigma"] = args.sigma
    eval_results["alpha"] = args.alpha
    
    print(f"Evaluation Results: {eval_results}")
    
    # 9. Save
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, f"seed_{config.seed}.json")
    
    with open(metrics_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    
    if not args.no_save_model:
        print("Saving model...")
        seed_output_dir = os.path.join(args.output_dir, f"checkpoint-seed-{config.seed}")
        trainer.save_model(seed_output_dir)
        tokenizer.save_pretrained(seed_output_dir)
    else:
        print("Skipping model save")
        
    print("Done!")

if __name__ == "__main__":
    main()
