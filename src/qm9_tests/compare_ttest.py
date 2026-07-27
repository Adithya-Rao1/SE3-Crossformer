import argparse
import csv
import json
import math

import numpy as np
from scipy import stats

def read_trial_maes(path):
    maes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            maes.append(float(row["mae"]))
    return maes

def confidence_interval_95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan")
    std = values.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    half_width = float(t_crit * std / math.sqrt(n))
    return mean, half_width

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True,
                         help="trial_maes.csv for model A (e.g. custom model).")
    parser.add_argument("--model_b", type=str, required=True,
                         help="trial_maes.csv for model B (e.g. authors' model).")
    parser.add_argument("--label_a", type=str, default="Model A")
    parser.add_argument("--label_b", type=str, default="Model B")
    parser.add_argument("--equal_var", action="store_true",
                         help="Use pooled-variance Student's t-test instead of Welch's.")
    parser.add_argument("--out_json", type=str, default="ttest_summary.json")
    args = parser.parse_args()

    maes_a = read_trial_maes(args.model_a)
    maes_b = read_trial_maes(args.model_b)

    mean_a, hw_a = confidence_interval_95(maes_a)
    mean_b, hw_b = confidence_interval_95(maes_b)

    t_stat, p_value = stats.ttest_ind(maes_a, maes_b, equal_var=args.equal_var)

    test_name = "Student's two-sample t-test (equal variance)" if args.equal_var \
        else "Welch's two-sample t-test (unequal variance)"

    print(f"{args.label_a}: n={len(maes_a)} trials, "
          f"mean MAE = {mean_a:.4f} +/- {hw_a:.4f} (95% CI)")
    print(f"{args.label_b}: n={len(maes_b)} trials, "
          f"mean MAE = {mean_b:.4f} +/- {hw_b:.4f} (95% CI)")
    print(f"\n{test_name}")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value     = {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        better = args.label_a if mean_a < mean_b else args.label_b
        print(f"\n  Result: significant difference at alpha=0.05 "
              f"(p={p_value:.4f} < {alpha}). Lower-MAE model: {better}.")
    else:
        print(f"\n  Result: not significant at alpha=0.05 "
              f"(p={p_value:.4f} >= {alpha}). No evidence the mean MAEs differ.")

    summary = {
        "test": test_name,
        "model_a": {
            "label": args.label_a, "trial_maes": maes_a,
            "mean_mae": mean_a, "ci95_half_width": hw_a,
        },
        "model_b": {
            "label": args.label_b, "trial_maes": maes_b,
            "mean_mae": mean_b, "ci95_half_width": hw_b,
        },
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out_json}")


if __name__ == "__main__":
    main()