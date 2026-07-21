import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.se3_crossformer.model import SE3InterNeighborhoodTransformer, SE3IntraOnlyTransformer
from src.se3_crossformer.se3_utils import RadialNetworkGRBF, RadialNetworkGSFB, RadialNetworkSFB
from src.train import load_qm9, _filter_small_graphs, ATOM_TYPES

GROUP_ORDER = ["graph_construction", "model_architecture", "rbf"]
RBF_LOOKUP = {"grbf": RadialNetworkGRBF, "gsfb": RadialNetworkGSFB, "sfb": RadialNetworkSFB}
MODEL_LOOKUP = {"inter": SE3InterNeighborhoodTransformer, "intra_only": SE3IntraOnlyTransformer}

def load_summary(base_metrics_dir):
    path = os.path.join(base_metrics_dir, "summary.csv")
    df = pd.read_csv(path)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df.sort_values(["group", "swept_param"]).reset_index(drop=True)

def load_trial_history(metrics_dir, trial=0):
    path = os.path.join(metrics_dir, f"metrics_trial{trial}.csv")
    if not os.path.exists(path):
        warnings.warn(f"No trial history at {path}, skipping.")
        return None
    return pd.read_csv(path)

def plot_training_curves_by_group(summary_df, out_dir, trial=0):
    panels = [
        ("loss", "Loss (MAE)"),
        ("gpu_util_pct", "GPU Utilization (%)"),
        ("cpu_util_pct", "CPU Utilization (%)"),
        ("mem_util_pct", "System Memory Utilization (%)"),
    ]

    for group in summary_df["group"].cat.categories:
        rows = summary_df[summary_df["group"] == group]
        if rows.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for _, row in rows.iterrows():
            hist = load_trial_history(row["metrics_dir"], trial)
            if hist is None:
                continue
            for (col, ylabel), ax in zip(panels, axes.flat):
                ax.plot(hist["step"], hist[col], linewidth=1.2, label=row["swept_param"])

        for (col, ylabel), ax in zip(panels, axes.flat):
            ax.set_xlabel("Accumulated batch step")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8, loc="best")
        fig.suptitle(f"Training curves -- {group}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(out_dir, f"01_training_curves__{group}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

def plot_test_mae_by_group(summary_df, out_dir):
    groups = [g for g in summary_df["group"].cat.categories
              if not summary_df[summary_df["group"] == g].empty]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    axes = axes[0]

    for ax, group in zip(axes, groups):
        rows = summary_df[summary_df["group"] == group]
        ok = rows[rows["status"] == "success"]
        x = np.arange(len(rows))
        ax.bar(
            x, rows["test_mae"].astype(float),
            yerr=rows["test_mae_ci95"].astype(float),
            capsize=4, color="#4C72B0", alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rows["swept_param"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Test MAE")
        ax.set_title(group)
        ax.grid(alpha=0.3, axis="y")
        if len(ok) < len(rows):
            ax.text(0.5, 0.95, f"{len(rows) - len(ok)} run(s) failed/missing",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7, color="crimson")

    fig.suptitle("Test MAE by ablation group (error bars = 95% CI over trials)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(out_dir, "02_test_mae_by_group.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_wall_time(summary_df, out_dir):
    df = summary_df.dropna(subset=["wall_time_sec"])
    if df.empty:
        warnings.warn("No wall_time_sec values found; skipping wall-time plot.")
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(df)), 5))
    colors = {"graph_construction": "#4C72B0", "model_architecture": "#DD8452", "rbf": "#55A868"}
    bar_colors = df["group"].map(colors).fillna("#888888")
    ax.bar(np.arange(len(df)), df["wall_time_sec"] / 60.0, color=bar_colors)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["swept_param"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title("Wall-clock time per ablation run")
    ax.grid(alpha=0.3, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "03_wall_time_by_run.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_mae_vs_wall_time(summary_df, out_dir):
    df = summary_df.dropna(subset=["wall_time_sec", "test_mae"])
    if df.empty:
        warnings.warn("No paired MAE/wall-time values found; skipping tradeoff plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"graph_construction": "#4C72B0", "model_architecture": "#DD8452", "rbf": "#55A868"}
    for group, sub in df.groupby("group"):
        ax.scatter(sub["wall_time_sec"] / 60.0, sub["test_mae"],
                   label=group, color=colors.get(group, "#888888"), s=60, edgecolor="k")
        for _, row in sub.iterrows():
            ax.annotate(row["swept_param"], (row["wall_time_sec"] / 60.0, row["test_mae"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Wall time (minutes)")
    ax.set_ylabel("Test MAE")
    ax.set_title("Accuracy vs. compute tradeoff")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "04_mae_vs_wall_time.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

def rebuild_model(row, args, device):
    radial_net = RBF_LOOKUP[row["rbf_type"]]
    model_cls = MODEL_LOOKUP[row["model_type"]]
    model = model_cls(
        radial_net=radial_net,
        in_features=len(ATOM_TYPES),
        max_degree=args.max_degree,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_parts=args.num_parts,
        out_dim=19,
        task="regression",
        partition_type=row["partition_type"],
    ).to(device)
    return model

@torch.no_grad()
def get_predictions(model, loader, num_parts, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts, device)
        if tensors is None:
            continue
        node_feat, pos, edge_index, atomic_mass, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue
        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)
        preds.append(pred.squeeze(-1).cpu().numpy())
        trues.append(target.cpu().numpy())
    if not preds:
        return None, None
    return np.concatenate(preds), np.concatenate(trues)

def lsrl_stats(true, pred):
    slope, intercept = np.polyfit(true, pred, 1)
    fit = slope * true + intercept
    ss_res = np.sum((pred - fit) ** 2)
    ss_tot = np.sum((pred - pred.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(pred - true))
    return slope, intercept, r2, mae

def plot_lsrl(ax, true, pred, title):
    slope, intercept, r2, mae = lsrl_stats(true, pred)
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.scatter(true, pred, s=8, alpha=0.35, color="#4C72B0", linewidths=0)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, xs, "k--", linewidth=1, label="y = x")
    ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=1.5,
            label=f"LSRL (R²={r2:.3f})")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nMAE={mae:.4f}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    return {"slope": slope, "intercept": intercept, "r2": r2, "mae": mae}

def plot_lsrl_grid(summary_df, args, device, out_dir, trial=0):
    ok_rows = summary_df[summary_df["status"] == "success"].reset_index(drop=True)
    if ok_rows.empty:
        warnings.warn("No successful runs to load checkpoints from; skipping LSRL grid.")
        return {}

    print("Loading QM9 test split once for all runs...")
    _, _, test_loader = load_qm9(args.target, args.batch_size, args.data_root, device)

    n = len(ok_rows)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    all_stats = {}
    for i, row in ok_rows.iterrows():
        ax = axes.flat[i]
        ckpt_path = os.path.join(row["metrics_dir"], f"best_model_trial{trial}.pt")
        if not os.path.exists(ckpt_path):
            ax.set_title(f"{row['run_name'] if 'run_name' in row else row['swept_param']}\n(checkpoint missing)")
            ax.axis("off")
            continue

        model = rebuild_model(row, args, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        pred, true = get_predictions(model, test_loader, args.num_parts, device)
        if pred is None:
            ax.set_title(f"{row['swept_param']}\n(no valid test graphs)")
            ax.axis("off")
            continue

        label = f"{row['group']}: {row['swept_param']}"
        stats = plot_lsrl(ax, true, pred, label)
        all_stats[label] = stats

    for j in range(i + 1, nrows * ncols):
        axes.flat[j].axis("off")

    fig.suptitle(f"Predicted vs. true (target index {args.target}) -- test set")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, "05_lsrl_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return all_stats

def plot_error_distribution_by_group(summary_df, args, device, out_dir, trial=0):
    ok_rows = summary_df[summary_df["status"] == "success"].reset_index(drop=True)
    if ok_rows.empty:
        return

    _, _, test_loader = load_qm9(args.target, args.batch_size, args.data_root, device)

    groups = [g for g in summary_df["group"].cat.categories]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    axes = axes[0]

    for ax, group in zip(axes, groups):
        rows = ok_rows[ok_rows["group"] == group]
        data, labels = [], []
        for _, row in rows.iterrows():
            ckpt_path = os.path.join(row["metrics_dir"], f"best_model_trial{trial}.pt")
            if not os.path.exists(ckpt_path):
                continue
            model = rebuild_model(row, args, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            pred, true = get_predictions(model, test_loader, args.num_parts, device)
            if pred is None:
                continue
            data.append(np.abs(pred - true))
            labels.append(row["swept_param"])

        if not data:
            ax.axis("off")
            continue
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel("|prediction error|")
        ax.set_title(group)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Absolute error distribution by ablation group (test set)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(out_dir, "06_error_distribution_by_group.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_metrics_dir", type=str, default="./ablation_metrics")
    p.add_argument("--out_dir", type=str, default="./ablation_figures")
    p.add_argument("--data_root", type=str, default="/home/ubuntu/se3-crossformer-data/data")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trial", type=int, default=0, help="Which trial's checkpoint/history to use.")

    p.add_argument("--target", type=int, default=1)
    p.add_argument("--num_parts", type=int, default=4)
    p.add_argument("--max_degree", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--feature_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)

    p.add_argument("--skip_model_eval", action="store_true",
                    help="Only plot CSV-derived figures (01-04); skip loading "
                         "checkpoints / running inference (05-06).")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    summary_df = load_summary(args.base_metrics_dir)
    print(f"Loaded {len(summary_df)} run(s) from summary.csv "
          f"({(summary_df['status'] == 'success').sum()} successful).")

    plot_training_curves_by_group(summary_df, args.out_dir, trial=args.trial)
    plot_test_mae_by_group(summary_df, args.out_dir)
    plot_wall_time(summary_df, args.out_dir)
    plot_mae_vs_wall_time(summary_df, args.out_dir)

    if args.skip_model_eval:
        print("Skipping model-loading plots (--skip_model_eval).")
        return

    stats = plot_lsrl_grid(summary_df, args, device, args.out_dir, trial=args.trial)
    for label, s in stats.items():
        print(f"{label}: R²={s['r2']:.4f}  MAE={s['mae']:.4f}  slope={s['slope']:.3f}")

    plot_error_distribution_by_group(summary_df, args, device, args.out_dir, trial=args.trial)

    print(f"\nAll figures written to {args.out_dir}")

if __name__ == "__main__":
    main()