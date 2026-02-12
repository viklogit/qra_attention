import json
import argparse
import os
import pandas as pd

def compare_robustness(baseline_path, rff_path):
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline results not found at {baseline_path}")
        return
    if not os.path.exists(rff_path):
        print(f"Error: RFF results not found at {rff_path}")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(rff_path, "r", encoding="utf-8") as f:
        rff = json.load(f)

    print("\n" + "="*80)
    print(f"{'ROBUSTNESS COMPARISON':^80}")
    print("="*80)
    
    # Extract metrics
    rows = []
    for perturb in ["clean", "dropout", "synonym", "shuffle"]:
        b_acc = base["metrics"][perturb]["eval_accuracy"]
        r_acc = rff["metrics"][perturb]["eval_accuracy"]
        diff = r_acc - b_acc
        
        rows.append({
            "Perturbation": perturb.capitalize(),
            "Baseline Acc": f"{b_acc:.4f}",
            "RFF Acc": f"{r_acc:.4f}",
            "Gap": f"{diff:+.4f}"
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("="*80)

    # Average Drop
    b_clean = base["metrics"]["clean"]["eval_accuracy"]
    r_clean = rff["metrics"]["clean"]["eval_accuracy"]
    
    b_drops = [b_clean - base["metrics"][p]["eval_accuracy"] for p in ["dropout", "synonym", "shuffle"]]
    r_drops = [r_clean - rff["metrics"][p]["eval_accuracy"] for p in ["dropout", "synonym", "shuffle"]]
    
    avg_b_drop = sum(b_drops) / len(b_drops)
    avg_r_drop = sum(r_drops) / len(r_drops)
    
    print(f"Average Robustness Drop:")
    print(f"  Baseline: {avg_b_drop*100:.2f}%")
    print(f"  RFF:      {avg_r_drop*100:.2f}%")
    
    improvement = avg_b_drop - avg_r_drop
    if improvement > 0:
        print(f"\n🎉 RFF is {improvement*100:.2f}% MORE robust than baseline!")
    else:
        print(f"\n⚠️  Baseline is {abs(improvement)*100:.2f}% MORE robust than RFF.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs RFF robustness results")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline_results.json")
    parser.add_argument("--rff", type=str, required=True, help="Path to rff_results.json")
    args = parser.parse_args()
    
    compare_robustness(args.baseline, args.rff)
