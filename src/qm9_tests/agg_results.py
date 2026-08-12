import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_models import compare, read_trial_maes

TARGET_NAMES = {
    0: "mu", 1: "alpha", 2: "homo", 3: "lumo", 4: "gap",
    5: "R2", 6: "zpve", 7: "U0", 8: "U", 9: "H", 10: "G", 11: "Cv",
}
MODEL_LABELS = {"se3-cross": "SE3IntraOnlyTransformer", "se3-trans": "se3-transformer-pytorch"}


def load_summary(pred_root, model, target):
    path = os.path.join(pred_root, model, f"target_{target}", "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+",
                         default=[1, 4, 3, 2, 0, 11])
    parser.add_argument("--models", type=str, nargs="+", default=["se3-cross", "se3-trans"])
    parser.add_argument("--pred_root", type=str, default="./predictions",
                         help="--out_root passed to predict_all_targets.py.")
    parser.add_argument("--out_dir", type=str, default="./comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    ttests = {}
    for target in args.targets:
        summaries = {}
        for model in args.models:
            summary = load_summary(args.pred_root, model, target)
            if summary is None:
                print(f"[warn] no summary.json for model={model} target={target}, skipping.")
                continue
            summaries[model] = summary
            rows.append({
                "target": target,
                "target_name": TARGET_NAMES.get(target, str(target)),
                "model": model,
                "n_trials": len(summary["trial_maes"]),
                "mean_mae": summary["mean_mae"],
                "ci95_half_width": summary["ci95_half_width"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "trial_maes": ";".join(f"{m:.6f}" for m in summary["trial_maes"]),
            })

        if "se3-cross" in summaries and "se3-trans" in summaries:
            result = compare(
                summaries["se3-cross"]["trial_maes"], summaries["se3-trans"]["trial_maes"],
                label_a="se3-cross", label_b="se3-trans",
            )
            ttests[target] = result

    # comparison.csv
    csv_path = os.path.join(args.out_dir, "comparison.csv")
    fieldnames = ["target", "target_name", "model", "n_trials", "mean_mae",
                  "ci95_half_width", "ci95_low", "ci95_high", "trial_maes",
                  "ttest_p_value", "ttest_significant", "ttest_lower_mae_model"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            t = ttests.get(row["target"])
            row = dict(row)
            row["ttest_p_value"] = t["p_value"] if t else ""
            row["ttest_significant"] = t["significant"] if t else ""
            row["ttest_lower_mae_model"] = t["lower_mae_model"] if t else ""
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    ttest_path = os.path.join(args.out_dir, "ttests_by_target.json")
    with open(ttest_path, "w") as f:
        json.dump({str(k): v for k, v in ttests.items()}, f, indent=2)
    print(f"Wrote {ttest_path}")

    # comparison.png
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [TARGET_NAMES.get(t, str(t)) for t in args.targets]
    x = range(len(args.targets))
    width = 0.35

    for i, model in enumerate(args.models):
        means, errs, xs = [], [], []
        for j, target in enumerate(args.targets):
            summary = load_summary(args.pred_root, model, target)
            if summary is None:
                continue
            means.append(summary["mean_mae"])
            errs.append(summary["ci95_half_width"])
            xs.append(j + (i - 0.5) * width)
        ax.errorbar(xs, means, yerr=errs, fmt="o", capsize=4, label=MODEL_LABELS.get(model, model))

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("QM9 target")
    ax.set_ylabel("Test MAE (95% CI over trials)")
    ax.set_title("Custom model vs. se3-transformer-pytorch, per QM9 target")
    ax.legend()
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "comparison.png")
    fig.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()