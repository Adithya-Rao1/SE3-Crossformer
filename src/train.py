"""
train.py
--------
Training loop and QM9 regression experiment (Section 4.2 of the manuscript).

Usage:
    python train.py --target 0 --num_parts 8 --max_degree 2 --epochs 300

    # Effective batch size of 256 with physical batch size of 32:
    python train.py --batch_size 32 --accum_steps 8 --target 0

QM9 target indices (0-based, following torch_geometric convention):
    0:  mu      (dipole moment, D)
    1:  alpha   (isotropic polarizability, a0^3)
    2:  homo    (HOMO energy, eV)
    3:  lumo    (LUMO energy, eV)
    4:  gap     (HOMO-LUMO gap, eV)
    5:  R2      (electronic spatial extent, a0^2)
    6:  zpve    (zero-point vibrational energy, eV)
    7:  U0      (internal energy at 0K, eV)
    8:  U       (internal energy at 298K, eV)
    9:  H       (enthalpy at 298K, eV)
    10: G       (free energy at 298K, eV)
    11: Cv      (heat capacity at 298K, cal/mol/K)
"""

import argparse
import math
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.se3_crossformer.model import SE3InterNeighborhoodTransformer, SE3IntraOnlyTransformer
from src.load_data import CustomQM9Dataset
from src.training_monitor import SystemMonitor
from src.se3_crossformer.se3_utils import RadialNetworkGRBF, RadialNetworkGSFB, RadialNetworkSFB

ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06
}

def get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [ATOMIC_MASSES.get(zi.item(), 12.0) for zi in z],
        dtype=torch.float32,
    )

def load_qm9(target_idx: int, batch_size=16, r: str = "./data", device=torch.device("cpu")):
    try:
        from torch_geometric.datasets import QM9
        from torch_geometric.loader import DataLoader
        from torch_geometric.transforms import Compose, Distance, NormalizeFeatures
    except ImportError:
        raise ImportError(
            "torch_geometric is required. Install with:\n"
            "  pip install torch_geometric"
        )

    dataset = CustomQM9Dataset(
        root=r+"/qm1",
        sdf_file=r+"/qm9/raw/gdb9.sdf",
        csv_file=r+"/qm9/raw/gdb9.sdf.csv",
        device=device,
    )

    dataset.y = dataset.y[:, target_idx]

    idx1 = 110000
    idx2 = 120000
    perm = torch.randperm(len(dataset))
    train_dataset = dataset[perm[:idx1]]
    val_dataset   = dataset[perm[idx1:idx2]]
    test_dataset  = dataset[perm[idx2:]]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

ATOM_TYPES = [1, 6, 7, 8, 9]

def one_hot_z(z: torch.Tensor) -> torch.Tensor:
    one_hot = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for idx, a in enumerate(ATOM_TYPES):
        one_hot[:, idx] = (z == a).float()
    return one_hot


def _filter_small_graphs(batch, num_parts, device):
    """
    Drop graphs with fewer nodes than num_parts and remap indices.
    Returns filtered tensors, or None if the whole batch is dropped.
    Extracted as a helper so train_epoch and evaluate share the same logic.
    """
    graph_batch = batch.batch.to(device)
    counts      = torch.bincount(graph_batch)
    valid_graph = counts >= num_parts

    if valid_graph.all():
        return (
            one_hot_z(batch.x).to(device),
            batch.pos.to(device),
            batch.edge_index.to(device),
            get_atomic_masses(batch.x).to(device),
            batch.y.to(device),
            graph_batch,
        )

    keep_nodes  = valid_graph[graph_batch]
    keep_graphs = valid_graph.nonzero(as_tuple=True)[0]

    if keep_graphs.numel() == 0:
        return None

    remap = torch.full((valid_graph.shape[0],), -1, dtype=torch.long, device=device)
    remap[keep_graphs] = torch.arange(keep_graphs.shape[0], device=device)

    node_feat   = one_hot_z(batch.x).to(device)[keep_nodes]
    pos         = batch.pos.to(device)[keep_nodes]
    atomic_mass = get_atomic_masses(batch.x).to(device)[keep_nodes]
    graph_batch = remap[graph_batch[keep_nodes]]
    target      = batch.y.to(device)[keep_graphs]

    src, dst   = batch.edge_index.to(device)
    edge_mask  = keep_nodes[src] & keep_nodes[dst]
    old_to_new = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[keep_nodes.nonzero(as_tuple=True)[0]] = \
        torch.arange(keep_nodes.sum(), device=device)
    edge_index = old_to_new[batch.edge_index.to(device)[:, edge_mask]]

    return node_feat, pos, edge_index, atomic_mass, target, graph_batch


def train_epoch(model, loader, optimizer, num_parts, device, accum_steps, epoch, 
                 monitor: SystemMonitor = None):
    """
    One training epoch with gradient accumulation.

    Args:
        accum_steps: number of micro-batches to accumulate before stepping.
                     Effective batch size = loader.batch_size * accum_steps.
                     Loss is averaged over the accumulated micro-batches so
                     the gradient magnitude is independent of accum_steps.
        monitor: optional SystemMonitor. If given, sample()d once per
                 micro-batch and commit()ted once per accumulated step, so
                 the recorded gpu/cpu/mem values are the mean over that
                 step's micro-batches.
    """
    model.train()
    total_loss  = 0.0
    n_graphs    = 0
    accum_loss  = torch.tensor(0.0, device=device)  # running sum within window

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch   = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts, device)
        if tensors is None:
            continue

        node_feat, pos, edge_index, atomic_mass, target, graph_batch = tensors

        if graph_batch.max() < 0:
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)

        # Divide loss by accum_steps so the accumulated gradient equals the
        # gradient of the mean loss over the full effective batch.
        loss = nn.functional.l1_loss(pred.squeeze(-1), target) / accum_steps
        loss.backward()

        if monitor is not None:
            monitor.sample()

        accum_loss = accum_loss + loss.detach()

        B = pred.shape[0]
        total_loss += loss.item() * accum_steps * B   # undo the /accum_steps for logging
        n_graphs   += B

        step_idx = i + 1
        if step_idx % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # print(f"Batch {step_idx} | accum MAE: {accum_loss.item():.4f}")
            if monitor is not None:
                monitor.commit(step_idx, accum_loss.item())
            accum_loss = torch.tensor(0.0, device=device)
        # else:
        #     print(f"Batch {step_idx} | micro-batch MAE: {loss.item() * accum_steps:.4f} "
        #           f"({step_idx % accum_steps}/{accum_steps})")

    # Flush any remaining accumulated gradients at epoch end (when
    # len(loader) is not divisible by accum_steps).
    remainder = len(loader) % accum_steps
    if remainder != 0:
        optimizer.step()
        optimizer.zero_grad()
        print(f"Flushed {remainder} remaining micro-batch(es).")
        if monitor is not None:
            monitor.commit(len(loader), accum_loss.item())

    print(f"Epoch: {epoch + 1} | Epoch MAE: {total_loss/max(n_graphs, 1):.4f}")
    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, num_parts, device):
    model.eval()
    total_mae = 0.0
    n_graphs  = 0

    for i, batch in enumerate(loader):
        batch   = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts, device)
        if tensors is None:
            continue

        node_feat, pos, edge_index, atomic_mass, target, graph_batch = tensors

        if graph_batch.max() < 0:
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)
        mae  = nn.functional.l1_loss(pred.squeeze(-1), target).item()
        B    = pred.shape[0]
        total_mae += mae * B
        n_graphs  += B
        print(f"[Eval Batch {i+1}] MAE: {mae:.4f}")

    return total_mae / max(n_graphs, 1)


def confidence_interval_95(values):
    n    = len(values)
    mean = np.mean(values)
    std  = np.std(values, ddof=1)
    from scipy import stats
    t_crit     = stats.t.ppf(0.975, df=n - 1)
    half_width = t_crit * std / math.sqrt(n)
    return mean, half_width


def main(partition_type="spectral", model_type="inter", rbf_type="grbf"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int,   default=1)
    parser.add_argument("--num_parts",   type=int,   default=4)
    parser.add_argument("--max_degree",  type=int,   default=2)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--accum_steps", type=int,   default=8,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size * accum_steps.")
    parser.add_argument("--partition_type", type=str, default="spectral")
    parser.add_argument("--model_type", type=str, default="inter")
    parser.add_argument("--rbf_type", type=str, default="grbf")
    parser.add_argument("--num_layers",  type=int,   default=4)
    parser.add_argument("--feature_dim", type=int,   default=32)
    parser.add_argument("--hidden_dim",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--trials",      type=int,   default=1)
    parser.add_argument("--data_root",   type=str,   default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--metrics_dir", type=str,   default="./training_metrics",
                        help="Directory for per-trial loss/GPU/CPU/memory "
                             "utilization CSVs and plots.")
    parser.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)

    device = torch.device(args.device)
    effective_batch = args.batch_size * args.accum_steps
    print(f"Device:           {device}")
    print(f"QM9 target:       {args.target}")
    print(f"Physical batch:   {args.batch_size}")
    print(f"Accum steps:      {args.accum_steps}")
    print(f"Effective batch:  {effective_batch}")

    train_loader, val_loader, test_loader = load_qm9(
        args.target, args.batch_size, args.data_root, args.device
    )

    trial_maes = []
    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1} / {args.trials} ---")
        torch.manual_seed(trial)

        checkpoint_path = os.path.join(
            args.metrics_dir,
            f"best_model_trial{trial}.pt"
        )

        if args.rbf_type == "grbf":
            radial_net = RadialNetworkGRBF
        elif args.rbf_type == "gsfb":
            radial_net = RadialNetworkGSFB
        else:
            radial_net = RadialNetworkSFB

        if args.model_type == "inter":
            model = SE3InterNeighborhoodTransformer(
                radial_net=radial_net,
                in_features=len(ATOM_TYPES),
                max_degree=args.max_degree,
                num_layers=args.num_layers,
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_parts=args.num_parts,
                out_dim=19,
                task="regression",
                partition_type=args.partition_type
            ).to(device)
        else:
            model = SE3IntraOnlyTransformer(
                radial_net=radial_net,
                in_features=len(ATOM_TYPES),
                max_degree=args.max_degree,
                num_layers=args.num_layers,
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_parts=args.num_parts,
                out_dim=19,
                task="regression",
                partition_type=args.partition_type
            ).to(device)


        
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        monitor = SystemMonitor(device)

        best_val_mae = float("inf")
        for epoch in range(args.epochs):
            print(f"[Epoch {epoch + 1}]")
            train_loss = train_epoch(
                model, train_loader, optimizer, args.num_parts, device,
                accum_steps=args.accum_steps, epoch=epoch, monitor=monitor,
            )
            val_mae = evaluate(model, val_loader, args.num_parts, device)
            scheduler.step()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"  [New best] val MAE: {val_mae:.4f} — saving checkpoint.")
                torch.save(model.state_dict(), checkpoint_path)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train MAE: {train_loss:.4f} | val MAE: {val_mae:.4f}")

        monitor.save_csv(os.path.join(args.metrics_dir, f"metrics_trial{trial}.csv"))
        monitor.save_plots(
            os.path.join(args.metrics_dir, f"metrics_trial{trial}.png"),
            title_prefix=(
                f"PT={args.partition_type}, "
                f"MT={args.model_type}, "
                f"RBF={args.rbf_type}"
            ),
        )

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_mae = evaluate(model, test_loader, args.num_parts, device)
        print(f"  Trial {trial + 1} test MAE: {test_mae:.4f}")
        trial_maes.append(test_mae)

    mean, hw = confidence_interval_95(trial_maes)
    print(f"\nTest MAE over {args.trials} trials: {mean:.4f} ± {hw:.4f}  (95% CI)")


if __name__ == "__main__":
    main()