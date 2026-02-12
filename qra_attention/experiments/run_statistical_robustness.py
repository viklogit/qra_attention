"""
Statistical Robustness Evaluation Script.

This script runs the robustness evaluation 5 times with different evaluation seeds
for both Baseline and RFF models, then computes mean ± std and paired t-tests.
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from scipy import stats


EVAL_SEEDS = [13, 42, 1234, 5678, 9999]
MODELS = {
    "baseline": "results/baseline/checkpoint-seed-13",
    "rff": "results/rff/checkpoint-seed-13"
}


def run_robustness_eval(model_name, model_path, eval_seed, output_json):
    """Run a single robustness evaluation."""
    cmd = [
        sys.executable,
        "qra_attention/experiments/eval_robustness.py",
        "--model_path", model_path,
        "--eval_seed", str(eval_seed),
        "--limit", "500",
        "--num_features", "128",
        "--sigma", "2.0",
        "--alpha", "0.9",
        "--results_json", output_json
    ]
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} with eval_seed={eval_seed}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Error running {model_name} with eval_seed {eval_seed}")
        return False
    
    print(f"✓ Completed {model_name} with eval_seed={eval_seed}")
    return True


def aggregate_and_analyze():
    """Aggregate results and perform statistical analysis."""
    results_dir = "results/statistical_robustness"
    
    # Collect all results
    all_results = {"baseline": [], "rff": []}
    
    for model_name in ["baseline", "rff"]:
        for eval_seed in EVAL_SEEDS:
            json_path = os.path.join(results_dir, f"{model_name}_seed_{eval_seed}.json")
            if not os.path.exists(json_path):
                print(f"⚠️  Warning: Missing {json_path}")
                continue
            
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            all_results[model_name].append({
                "eval_seed": eval_seed,
                "clean_acc": data["metrics"]["clean"]["eval_accuracy"],
                "clean_f1": data["metrics"]["clean"]["eval_f1"],
                "dropout_acc": data["metrics"]["dropout"]["eval_accuracy"],
                "dropout_f1": data["metrics"]["dropout"]["eval_f1"],
                "synonym_acc": data["metrics"]["synonym"]["eval_accuracy"],
                "synonym_f1": data["metrics"]["synonym"]["eval_f1"],
                "shuffle_acc": data["metrics"]["shuffle"]["eval_accuracy"],
                "shuffle_f1": data["metrics"]["shuffle"]["eval_f1"],
            })
    
    # Convert to DataFrames
    baseline_df = pd.DataFrame(all_results["baseline"])
    rff_df = pd.DataFrame(all_results["rff"])
    
    if len(baseline_df) == 0 or len(rff_df) == 0:
        print("❌ Insufficient data for analysis")
        return
    
    # Compute perturbation gaps (Clean - Perturbed)
    baseline_df["dropout_gap"] = baseline_df["clean_acc"] - baseline_df["dropout_acc"]
    baseline_df["synonym_gap"] = baseline_df["clean_acc"] - baseline_df["synonym_acc"]
    baseline_df["shuffle_gap"] = baseline_df["clean_acc"] - baseline_df["shuffle_acc"]
    
    rff_df["dropout_gap"] = rff_df["clean_acc"] - rff_df["dropout_acc"]
    rff_df["synonym_gap"] = rff_df["clean_acc"] - rff_df["synonym_acc"]
    rff_df["shuffle_gap"] = rff_df["clean_acc"] - rff_df["shuffle_acc"]
    
    # Statistical Analysis
    print("\n" + "="*80)
    print("STATISTICAL ROBUSTNESS ANALYSIS")
    print("="*80)
    
    # Summary statistics
    print("\n--- BASELINE ---")
    print(f"Clean Accuracy:   {baseline_df['clean_acc'].mean():.4f} ± {baseline_df['clean_acc'].std():.4f}")
    print(f"Dropout Gap:      {baseline_df['dropout_gap'].mean():.4f} ± {baseline_df['dropout_gap'].std():.4f}")
    print(f"Synonym Gap:      {baseline_df['synonym_gap'].mean():.4f} ± {baseline_df['synonym_gap'].std():.4f}")
    print(f"Shuffle Gap:      {baseline_df['shuffle_gap'].mean():.4f} ± {baseline_df['shuffle_gap'].std():.4f}")
    
    print("\n--- RFF ---")
    print(f"Clean Accuracy:   {rff_df['clean_acc'].mean():.4f} ± {rff_df['clean_acc'].std():.4f}")
    print(f"Dropout Gap:      {rff_df['dropout_gap'].mean():.4f} ± {rff_df['dropout_gap'].std():.4f}")
    print(f"Synonym Gap:      {rff_df['synonym_gap'].mean():.4f} ± {rff_df['synonym_gap'].std():.4f}")
    print(f"Shuffle Gap:      {rff_df['shuffle_gap'].mean():.4f} ± {rff_df['shuffle_gap'].std():.4f}")
    
    # Paired t-tests on perturbation gaps
    print("\n--- PAIRED T-TESTS (Baseline vs RFF on Perturbation Gaps) ---")
    
    statistical_results = []
    
    for gap_type in ["dropout_gap", "synonym_gap", "shuffle_gap"]:
        baseline_gaps = baseline_df[gap_type].values
        rff_gaps = rff_df[gap_type].values
        
        t_stat, p_value = stats.ttest_rel(baseline_gaps, rff_gaps)
        
        mean_diff = baseline_gaps.mean() - rff_gaps.mean()
        
        statistical_results.append({
            "perturbation": gap_type.replace("_gap", ""),
            "baseline_mean_gap": baseline_gaps.mean(),
            "baseline_std_gap": baseline_gaps.std(),
            "rff_mean_gap": rff_gaps.mean(),
            "rff_std_gap": rff_gaps.std(),
            "mean_difference": mean_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_05": p_value < 0.05,
            "significant_01": p_value < 0.01
        })
        
        print(f"\n{gap_type.replace('_', ' ').title()}:")
        print(f"  Baseline: {baseline_gaps.mean():.4f} ± {baseline_gaps.std():.4f}")
        print(f"  RFF:      {rff_gaps.mean():.4f} ± {rff_gaps.std():.4f}")
        print(f"  Difference: {mean_diff:.4f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        print(f"  Significant (p<0.05): {'✓ YES' if p_value < 0.05 else '✗ NO'}")
    
    # Save results
    stats_df = pd.DataFrame(statistical_results)
    output_csv = os.path.join(results_dir, "statistical_analysis.csv")
    stats_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved statistical analysis to {output_csv}")
    
    # Save detailed results
    baseline_df.to_csv(os.path.join(results_dir, "baseline_detailed.csv"), index=False)
    rff_df.to_csv(os.path.join(results_dir, "rff_detailed.csv"), index=False)
    print(f"✓ Saved detailed results to {results_dir}/")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("A LOWER perturbation gap means the model is MORE ROBUST.")
    print("If RFF has significantly lower gaps (p<0.05), it is more robust than Baseline.")
    print("="*80)


def main():
    results_dir = "results/statistical_robustness"
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print("STATISTICAL ROBUSTNESS EXPERIMENT")
    print("="*80)
    print(f"Evaluation seeds: {EVAL_SEEDS}")
    print(f"Models: {list(MODELS.keys())}")
    print("="*80)
    
    # Run all evaluations
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"❌ Model checkpoint not found: {model_path}")
            print(f"   Please ensure the model is trained and saved.")
            sys.exit(1)
        
        for eval_seed in EVAL_SEEDS:
            output_json = os.path.join(results_dir, f"{model_name}_seed_{eval_seed}.json")
            success = run_robustness_eval(model_name, model_path, eval_seed, output_json)
            if not success:
                print(f"❌ Failed to complete evaluation for {model_name} with seed {eval_seed}")
                sys.exit(1)
    
    # Aggregate and analyze
    print("\n" + "="*80)
    print("All evaluations complete. Running statistical analysis...")
    print("="*80)
    aggregate_and_analyze()
    
    print("\n✓ Statistical robustness experiment complete!")


if __name__ == "__main__":
    main()
