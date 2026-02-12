"""
Utility to re-aggregate all results from JSON files into CSVs.
"""
import os
import json
import pandas as pd
import numpy as np

def aggregate():
    base_dir = "results"
    seeds = [13, 42, 1234]
    models = ["baseline", "rff"]
    
    all_records = []
    summary_records = []
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"Warning: {model_dir} not found. Skipping.")
            continue
            
        model_accuracies = []
        model_f1s = []
        
        for seed in seeds:
            json_path = os.path.join(model_dir, f"seed_{seed}.json")
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found.")
                continue
                
            with open(json_path, "r") as f:
                data = json.load(f)
            
            record = {
                "experiment": model,
                "seed": seed,
                "accuracy": data.get("eval_accuracy"),
                "f1": data.get("eval_f1"),
                "loss": data.get("eval_loss"),
                "runtime": data.get("eval_runtime"),
                "samples_per_second": data.get("eval_samples_per_second"),
                "alpha": data.get("alpha"),
                "num_features": data.get("num_features")
            }
            all_records.append(record)
            model_accuracies.append(record["accuracy"])
            model_f1s.append(record["f1"])
            
        if model_accuracies:
            summary_records.append({
                "experiment": model,
                "mean_accuracy": np.mean(model_accuracies),
                "std_accuracy": np.std(model_accuracies),
                "mean_f1": np.mean(model_f1s),
                "std_f1": np.std(model_f1s),
                "n_seeds": len(model_accuracies)
            })

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(os.path.join(base_dir, "details.csv"), index=False)
        
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(os.path.join(base_dir, "summary.csv"), index=False)
        
        print(f"Successfully re-aggregated {len(all_records)} records into details.csv and summary.csv")
        print("\nSummary:")
        print(summary_df)
    else:
        print("No records found to aggregate.")

if __name__ == "__main__":
    aggregate()
