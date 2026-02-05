"""
Master script to run multi-seed experiments and aggregate results.
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from datetime import datetime

SEEDS = [13, 42, 1234]
MODELS = ["baseline", "rff"]

def run_training_script(script_name, seed, output_base, no_save_model=True):
    """Run a single training script with a specific seed."""
    cmd = [
        sys.executable,
        f"qra_attention/experiments/{script_name}",
        "--seed", str(seed),
        "--output_dir", output_base
    ]
    
    if no_save_model:
        cmd.append("--no_save_model")
        
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False) # stream output to console
    
    if result.returncode != 0:
        print(f"Error running {script_name} with seed {seed}")
        return False
    return True

def aggregate_results(results_dir, seeds, experiment_name):
    """Read JSON files and aggregate metrics."""
    records = []
    
    for seed in seeds:
        json_path = os.path.join(results_dir, f"seed_{seed}.json")
        if not os.path.exists(json_path):
            print(f"Warning: Missing results file {json_path}")
            continue
            
        with open(json_path, "r") as f:
            data = json.load(f)
            
        record = {
            "experiment": experiment_name,
            "seed": seed,
            "accuracy": data.get("eval_accuracy"),
            "f1": data.get("eval_f1"),
            "loss": data.get("eval_loss"),
            "runtime": data.get("eval_runtime"),
            "samples_per_second": data.get("eval_samples_per_second"),
            "alpha": data.get("alpha"),
            "num_features": data.get("num_features")
        }
        records.append(record)
        
    if not records:
        return None
        
    df = pd.DataFrame(records)
    
    # Calculate summary stats

    summary = {
        "experiment": experiment_name,
        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy": df["accuracy"].std(),
        "mean_f1": df["f1"].mean(),
        "std_f1": df["f1"].std(),
        "n_seeds": len(df)
    }
    
    return df, summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--alpha", type=float, default=0.9, help="Interpolation factor for RFF")
    parser.add_argument("--num_features", type=int, default=128, help="Number of random features")
    parser.add_argument("--sigma", type=float, default=2.0, help="Kernel bandwidth sigma")
    parser.add_argument("--save_models", action="store_true", help="Save model checkpoints (required for robustness eval)")
    parser.add_argument("--use_accelerate", action="store_true", help="Use 'accelerate launch' for training scripts")
    args = parser.parse_args()
    
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Initialize lists to store results
    all_summary_records = []
    all_detail_records = []
    
    current_seeds = SEEDS
    if args.smoke_test:
        print("!!! SMOKE TEST MODE: Running only seed 13 for 1 epoch !!!")
        current_seeds = [13]
    
    for model in MODELS:
        print(f"\n{'='*20}\nRunning Experiment: {model}\n{'='*20}")
        script_name = f"train_{model}.py"
        output_dir = os.path.join(base_results_dir, model)
        
        # 1. Run seeds
        for seed in current_seeds:
            print(f"\n--- Seed {seed} ---")
            
            # Prepare command
            if args.use_accelerate:
                cmd = ["accelerate", "launch"]
            else:
                cmd = [sys.executable]
            
            cmd.extend([
                f"qra_attention/experiments/{script_name}",
                "--seed", str(seed),
                "--output_dir", output_dir
            ])
            
            if not args.save_models:
                cmd.append("--no_save_model")
            
            if args.smoke_test:
                cmd.append("--smoke_test")
            
            if model == "rff":
                cmd.extend(["--alpha", str(args.alpha)])
                cmd.extend(["--num_features", str(args.num_features)])
                cmd.extend(["--sigma", str(args.sigma)])
                
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
        # 2. Aggregate
        print(f"Aggregating results for {model}...")
        df, summary = aggregate_results(output_dir, current_seeds, model)
        
        if df is not None:
            all_detail_records.append(df)
            all_summary_records.append(summary)
            print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} \u00b1 {summary['std_accuracy']:.4f}")
            
    # 3. Final Outputs
    if all_detail_records:
        full_df = pd.concat(all_detail_records, ignore_index=True)
        summary_df = pd.DataFrame(all_summary_records)
        
        # Save detailed breakdown
        full_df.to_csv(os.path.join(base_results_dir, "details.csv"), index=False)
        
        # Save summary
        summary_path = os.path.join(base_results_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nSaved summary to {summary_path}")
        print(summary_df)

if __name__ == "__main__":
    main()
