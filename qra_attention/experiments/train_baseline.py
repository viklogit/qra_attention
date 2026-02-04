"""
Script to train the baseline DistilBERT model on IMDb.
This establishes the performance benchmark for standard attention
under the same freezing constraints as the kernel experiments.
"""

import os
import argparse
import numpy as np
import torch  # Import torch first
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
from qra_attention.patching import freeze_layers, get_trainable_params_summary

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
    parser = argparse.ArgumentParser(description="Train baseline DistilBERT on IMDb")
    parser.add_argument("--output_dir", type=str, default=ExperimentConfig.output_dir, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=ExperimentConfig.batch_size, help="Batch size")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed, help="Random seed")
    parser.add_argument("--no_save_model", action="store_true", help="Do not save the model checkpoint (to save space)")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test with minimal data")
    args = parser.parse_args()
    
    # 1. Setup
    config = ExperimentConfig()
    config.seed = args.seed  # Override seed from args
    set_seed(config.seed)
    
    if args.smoke_test:
        print("!!! RUNNING SMOKE TEST !!!")
        config.num_epochs = 1
        config.logging_dir = "logs/smoke_test"
        args.output_dir = "results/smoke_test"
    else:
        # Standard run structure: results/baseline/seed_{seed}
        # But user asked for flat files: results/baseline/seed_{seed}.json
        # So we keep output_dir as base folder, but we'll use specific paths for saving files.
        # Actually, Trainer needs a directory. Let's make a subdir per seed to be safe given Trainer conventions,
        # or we just save the JSON specially at the end.
        pass

    print(f"Loading model: {config.model_name}")
    print(f"Output directory: {args.output_dir}")
    
    # 2. Data Loading & Tokenization
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
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
        print("  Truncated dataset for smoke test: 20 train, 10 test")
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Model Initialization
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=config.num_labels
    )
    
    # 4. Apply Freezing (Baseline Constraint)
    # We freeze L0-L3 to match the experimental condition for kernel attention
    print("Applying freezing constraints...")
    freeze_summary = freeze_layers(
        model, 
        freeze_embeddings=config.freeze_embeddings, 
        freeze_layer_indices=list(config.freeze_layers)
    )
    
    print(f"Model Summary:")
    print(f"  Total params: {freeze_summary['frozen_params'] + freeze_summary['trainable_params']:,}")
    print(f"  Trainable: {freeze_summary['trainable_params']:,}")
    print(f"  Frozen: {freeze_summary['frozen_params']:,}")
    
    # 5. Training Setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=config.logging_dir,
        seed=config.seed,
        fp16=True, # Use mixed precision if available
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")
    
    # 8. Save
    import json
    
    # Save metrics to specific JSON file requested: results/baseline/seed_{seed}.json
    # Ensure output directory exists provided by args.output_dir
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
        print("Skipping model save (--no_save_model used)")
        
    print("Done!")

if __name__ == "__main__":
    main()
