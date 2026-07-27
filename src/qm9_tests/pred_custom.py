import argparse
import csv
import json
import math
import os

import numpy as np
import torch
from scipy import stats

from src.se3_crossformer.model import SE3InterNeighborhoodTransformer
from src.se3_crossformer.se3_utils import RadialNetworkGRBF, RadialNetworkGSFB
from src.train import load_qm9, _filter_small_graphs, ATOM_TYPES


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


def build_model(args, device):
    if args.rbf_type == "grbf":
        radial_net = RadialNetworkGRBF
    elif args.rbf_type == "gsfb":
        radial_net = RadialNetworkGSFB
    else:
        radial_net = RadialNetworkGSFB

    return SE3InterNeighborhoodTransformer(
        radial_net=radial_net,
        in_features=len(ATOM_TYPES),
        max_degree=args.max_degree,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_parts=args.num_parts,
        scalar_out_dim=1,
        task=0,
        partition_type=args.partition_type,
    ).to(device)


@torch.no_grad()
def predict_and_record(model, loader, num_parts, device, out_csv):
    model.eval()
    rows = []
    total_abs_err = 0.0
    n_graphs = 0
    mol_idx = 0

    for batch in loader:
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts, device)
        if tensors is None:
            continue
        node_feat, pos, edge_index, atomic_mass, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch).squeeze(-1)
        abs_err = (pred - target).abs()

        for p, t, e in zip(pred.tolist(), target.tolist(), abs_err.tolist()):
            rows.append({"mol_index": mol_idx, "prediction": p, "target": t, "abs_error": e})
            mol_idx += 1

        total_abs_err += abs_err.sum().item()
        n_graphs += pred.shape[0]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mol_index", "prediction", "target", "abs_error"])
        writer.writeheader()
        writer.writerows(rows)

    mae = total_abs_err / max(n_graphs, 1)
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--rbf_type", type=str, default="gsfb",
                         help="Must match the value used at training time.")
    parser.add_argument("--partition_type", type=str, default="spectral",
                         help="Must match the value used at training time.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/ubuntu/se3-crossformer-data/data",
                         help="Directory containing best_model_trial{i}_target{t}.pt files.")
    parser.add_argument("--checkpoint_template", type=str,
                         default="best_model_trial{trial}_target{target}.pt")
    parser.add_argument("--data_root", type=str,
                         default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--out_dir", type=str, default="./predictions_custom")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    _, _, test_loader = load_qm9(args.target, args.batch_size, args.data_root, args.device)

    trial_maes = []
    for trial in range(args.trials):
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            args.checkpoint_template.format(trial=trial, target=args.target),
        )
        if not os.path.exists(ckpt_path):
            print(f"[trial {trial}] checkpoint not found at {ckpt_path}, skipping.")
            continue

        model = build_model(args, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        out_csv = os.path.join(args.out_dir, f"predictions_trial{trial}.csv")
        mae = predict_and_record(model, test_loader, args.num_parts, device, out_csv)
        print(f"[trial {trial}] test MAE: {mae:.4f}  (predictions -> {out_csv})")
        trial_maes.append(mae)

    if len(trial_maes) == 0:
        raise RuntimeError("No checkpoints were found/evaluated.")

    mean_mae, half_width = confidence_interval_95(trial_maes)

    summary = {
        "model": "custom_se3_interneighborhood_transformer",
        "target_index": args.target,
        "trial_maes": trial_maes,
        "mean_mae": mean_mae,
        "ci95_half_width": half_width,
        "ci95_low": mean_mae - half_width if not math.isnan(half_width) else None,
        "ci95_high": mean_mae + half_width if not math.isnan(half_width) else None,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "trial_maes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "mae"])
        for i, m in enumerate(trial_maes):
            writer.writerow([i, m])

    print(f"\nTest MAE over {len(trial_maes)} trials: "
          f"{mean_mae:.4f} +/- {half_width:.4f}  (95% CI)")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()