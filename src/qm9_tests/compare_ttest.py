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


def compare(maes_a, maes_b, label_a="Model A", label_b="Model B", equal_var=False, alpha=0.05):
    """Return a summary dict for a two-sample t-test between two MAE samples."""
    mean_a, hw_a = confidence_interval_95(maes_a)
    mean_b, hw_b = confidence_interval_95(maes_b)
    t_stat, p_value = stats.ttest_ind(maes_a, maes_b, equal_var=equal_var)
    test_name = ("Student's two-sample t-test (equal variance)" if equal_var
                 else "Welch's two-sample t-test (unequal variance)")

    return {
        "test": test_name,
        "model_a": {"label": label_a, "trial_maes": list(maes_a), "mean_mae": mean_a, "ci95_half_width": hw_a},
        "model_b": {"label": label_b, "trial_maes": list(maes_b), "mean_mae": mean_b, "ci95_half_width": hw_b},
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "lower_mae_model": label_a if mean_a < mean_b else label_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True,
                         help="trial_maes.csv for model A (e.g. se3-cross model).")
    parser.add_argument("--model_b", type=str, required=True,
                         help="trial_maes.csv for model B (e.g. se3-trans' model).")
    parser.add_argument("--label_a", type=str, default="Model A")
    parser.add_argument("--label_b", type=str, default="Model B")
    parser.add_argument("--equal_var", action="store_true",
                         help="Use pooled-variance Student's t-test instead of Welch's.")
    parser.add_argument("--out_json", type=str, default="ttest_summary.json")
    args = parser.parse_args()

    maes_a = read_trial_maes(args.model_a)
    maes_b = read_trial_maes(args.model_b)

    summary = compare(maes_a, maes_b, args.label_a, args.label_b, equal_var=args.equal_var)

    print(f"{args.label_a}: n={len(maes_a)} trials, "
          f"mean MAE = {summary['model_a']['mean_mae']:.4f} +/- {summary['model_a']['ci95_half_width']:.4f} (95% CI)")
    print(f"{args.label_b}: n={len(maes_b)} trials, "
          f"mean MAE = {summary['model_b']['mean_mae']:.4f} +/- {summary['model_b']['ci95_half_width']:.4f} (95% CI)")
    print(f"\n{summary['test']}")
    print(f"  t-statistic = {summary['t_statistic']:.4f}")
    print(f"  p-value     = {summary['p_value']:.6f}")

    if summary["significant"]:
        print(f"\n  Result: significant difference at alpha=0.05 "
              f"(p={summary['p_value']:.4f} < 0.05). Lower-MAE model: {summary['lower_mae_model']}.")
    else:
        print(f"\n  Result: not significant at alpha=0.05 "
              f"(p={summary['p_value']:.4f} >= 0.05). No evidence the mean MAEs differ.")

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out_json}")


if __name__ == "__main__":
    main()