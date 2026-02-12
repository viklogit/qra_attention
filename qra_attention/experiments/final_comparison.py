import pandas as pd
import numpy as np
import os

def final_comparison(details_path="results/details.csv"):
    if not os.path.exists(details_path):
        print(f"Error: {details_path} not found.")
        return

    df = pd.read_csv(details_path)
    
    # Filter by experiment
    baseline_accs = df[df["experiment"] == "baseline"]["accuracy"].values
    rff_accs = df[df["experiment"] == "rff"]["accuracy"].values
    
    if len(baseline_accs) == 0 or len(rff_accs) == 0:
        print("Error: Missing data for either baseline or rff.")
        return

    print("="*60)
    print("FINAL RESULTS")
    print("="*60)

    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs)
    rff_mean = np.mean(rff_accs)
    rff_std = np.std(rff_accs)

    gap = rff_mean - baseline_mean

    print(f"\nBaseline:  {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"RFF:       {rff_mean:.4f} ± {rff_std:.4f}")
    print(f"\nGap:       {gap:+.4f} ({gap*100:+.2f}%)")

    if abs(gap) < 0.005:
        print("\n✅ SUCCESS: RFF matches baseline!")
    elif gap > 0:
        print("\n🎉 SUCCESS: RFF beats baseline!")
    elif abs(gap) < 0.015:
        print("\n✓ ACCEPTABLE: RFF within 1.5% of baseline")
    else:
        print("\n⚠️  NEEDS WORK: Gap larger than expected")

if __name__ == "__main__":
    final_comparison()
