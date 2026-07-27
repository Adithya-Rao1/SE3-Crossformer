import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch_geometric.utils import to_dense_batch, to_dense_adj

from se3_transformer_pytorch import SE3Transformer

from src.train import load_qm9, _filter_small_graphs

ATOM_VOCAB = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
NUM_TOKENS = len(ATOM_VOCAB)
NUM_EDGE_TOKENS = 5  


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


class SE3RegressionWrapper(nn.Module):
    def __init__(self, dim=32, depth=2, num_degrees=2, dim_head=16, heads=4):
        super().__init__()
        self.backbone = SE3Transformer(
            num_tokens=NUM_TOKENS,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            depth=depth,
            input_degrees=1,
            num_degrees=num_degrees,
            output_degrees=1,
            num_edge_tokens=NUM_EDGE_TOKENS,
            edge_dim=16,
            reduce_dim_out=True,
        )
        self.readout = nn.Linear(1, 19)

    def forward(self, atoms, coors, mask, edges):
        per_atom = self.backbone(atoms, coors, mask, edges=edges, return_type=0)
        per_atom = per_atom.squeeze(-1)  

        mask_f = mask.float()
        pooled = (per_atom * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = pooled.unsqueeze(-1) 
        return self.readout(pooled).squeeze(-1)


def batch_to_dense(batch, device, max_nodes=None):
    z = batch.x.long().to(device) 
    unknown = torch.tensor(
        [ATOM_VOCAB.get(int(zi), -1) for zi in z], device=device
    )
    valid = unknown >= 0
    if not valid.all():
        pass

    tokens = torch.zeros_like(z)
    for raw_z, tok in ATOM_VOCAB.items():
        tokens[z == raw_z] = tok

    atoms_dense, mask = to_dense_batch(tokens, batch.batch, max_num_nodes=max_nodes)
    coors_dense, _ = to_dense_batch(batch.pos.to(device), batch.batch, max_num_nodes=max_nodes)

    bond_order = batch.edge_attr[:, 0].round().long().clamp(min=0, max=NUM_EDGE_TOKENS - 1)
    edges_dense = to_dense_adj(
        batch.edge_index.to(device), batch.batch, edge_attr=bond_order.to(device),
        max_num_nodes=atoms_dense.shape[1],
    ).long()

    return atoms_dense, coors_dense, mask, edges_dense

@torch.no_grad()
def predict_and_record(model, loader, device, out_csv):
    model.eval()
    rows = []
    total_abs_err = 0.0
    n_graphs = 0
    mol_idx = 0

    for batch in loader:
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts=1, device=device)
        if tensors is None:
            continue
        _, _, _, _, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue

        atoms, coors, mask, edges = batch_to_dense(batch, device)
        pred = model(atoms, coors, mask, edges)

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
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Dense (b, n, n) edge tensors are memory-heavy; "
                              "keep this smaller than the custom model's batch size.")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_degrees", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/ubuntu/se3-crossformer-data/se3-transformer-pytorch/se3_transformer_pytorch/data",
                         help="Directory containing the authors' checkpoints.")
    parser.add_argument("--checkpoint_template", type=str,
                         default="/home/ubuntu/se3-crossformer-data/data/authors_trial0.pt")
    parser.add_argument("--data_root", type=str,
                         default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--out_dir", type=str, default="./predictions_se3_transformer_pytorch")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    _, _, test_loader = load_qm9(args.target, args.batch_size, args.data_root, args.device)

    trial_maes = []
    for trial in range(args.trials):
        ckpt_path = os.path.join(
            args.checkpoint_dir, args.checkpoint_template.format(trial=trial)
        )
        if not os.path.exists(ckpt_path):
            print(f"[trial {trial}] checkpoint not found at {ckpt_path}, skipping.")
            continue

        model = SE3RegressionWrapper(
            dim=args.dim, depth=args.depth, num_degrees=args.num_degrees
        ).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  

        out_csv = os.path.join(args.out_dir, f"predictions_trial{trial}.csv")
        mae = predict_and_record(model, test_loader, device, out_csv)
        print(f"[trial {trial}] test MAE: {mae:.4f}  (predictions -> {out_csv})")
        trial_maes.append(mae)

    if len(trial_maes) == 0:
        raise RuntimeError("No checkpoints were found/evaluated.")

    mean_mae, half_width = confidence_interval_95(trial_maes)

    summary = {
        "model": "se3_transformer_pytorch_authors",
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