"""
Statistical Rigor Analysis for QRA Attention.

This module performs paired-samples statistical tests (Paired T-test, Cohen's d)
to compare the performance of Kernel Attention against the Baseline.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def compute_cohens_d_paired(x, y):
    """Compute Cohen's d for paired samples."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def run_analysis(details_path="results/details.csv"):
    if not os.path.exists(details_path):
        print(f"Error: {details_path} not found.")
        return

    df = pd.read_csv(details_path)
    
    # Pivot to get paired results per seed
    pivot_df = df.pivot(index="seed", columns="experiment", values=["accuracy", "f1"])
    
    results = []
    
    for metric in ["accuracy", "f1"]:
        baseline = pivot_df[metric]["baseline"]
        rff = pivot_df[metric]["rff"]
        
        # Paired T-test
        t_stat, p_val = stats.ttest_rel(baseline, rff)
        
        # Cohen's d (Paired)
        d = compute_cohens_d_paired(baseline, rff)
        
        results.append({
            "metric": metric,
            "baseline_mean": baseline.mean(),
            "rff_mean": rff.mean(),
            "mean_diff": baseline.mean() - rff.mean(),
            "p_value": p_val,
            "cohens_d": d,
            "significant_05": p_val < 0.05
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("STATISTICAL RIGOR REPORT (Paired Samples)")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # Save results
    output_path = "results/statistical_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved statistical report to {output_path}")

if __name__ == "__main__":
    run_analysis()
