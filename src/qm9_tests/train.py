"""
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

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Distance, NormalizeFeatures

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.se3_crossformer.model import SE3IntraOnlyTransformer
from src.load_data import CustomQM9Dataset, BOND_TYPES
from src.training_monitor import SystemMonitor
from src.se3_crossformer.se3_utils import RadialNetworkGRBF, RadialNetworkGSFB, RadialNetworkSFB
from src.qm9_tests.dummy_data import load_qm9_dummy

ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06
}

# alpha, gap, homo, lumo, mu, and Cv
target_indices = [1, 4, 3, 2, 0, 11]

def get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [ATOMIC_MASSES.get(zi.item(), 12.0) for zi in z],
        dtype=torch.float32,
    )

def load_qm9(batch_size=16, r: str = "./data", device=torch.device("cpu")):
    dataset = CustomQM9Dataset(
        root=r+"/qm1",
        sdf_file=r+"/qm9/raw/gdb9.sdf",
        csv_file=r+"/qm9/raw/gdb9.sdf.csv",
        device=torch.device('cpu'),
    )

    idx1 = 110000
    idx2 = 120000
    idx3 = 130000
    perm = torch.randperm(len(dataset))
    train_dataset = dataset[perm[:idx1]]
    val_dataset   = dataset[perm[idx1:idx2]]
    test_dataset  = dataset[perm[idx2:idx3]]

    loader_kwargs = dict(num_workers=8, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

ATOM_TYPES = [1, 6, 7, 8, 9]

def one_hot_z(z: torch.Tensor) -> torch.Tensor:
    one_hot = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for idx, a in enumerate(ATOM_TYPES):
        one_hot[:, idx] = (z == a).float()
    return one_hot


def _filter_small_graphs(target_idx, batch, min_nodes, device):
    graph_batch = batch.batch.to(device)
    counts      = torch.bincount(graph_batch)
    valid_graph = counts >= min_nodes

    if valid_graph.all():
        return (
            one_hot_z(batch.x).to(device),
            batch.pos.to(device),
            batch.edge_index.to(device),
            batch.edge_attr.to(device),
            get_atomic_masses(batch.x).to(device),
            batch.y[:, target_idx].to(device),
            graph_batch,
            batch.idx.to(device),
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
    target      = batch.y[:, target_idx].to(device)[keep_graphs]
    graph_idx   = batch.idx.to(device)[keep_graphs]

    src, dst   = batch.edge_index.to(device)
    edge_mask  = keep_nodes[src] & keep_nodes[dst]
    # Compact remap via prefix-sum instead of arange(keep_nodes.sum()) -- avoids
    # resolving the kept-node count to a Python int (a sync) just to size the
    # arange; entries for dropped nodes are never read since edge_mask already
    # guarantees both endpoints of every surviving edge are kept.
    old_to_new = torch.cumsum(keep_nodes.long(), dim=0) - 1
    edge_index = old_to_new[batch.edge_index.to(device)[:, edge_mask]]
    edge_attr  = batch.edge_attr.to(device)[edge_mask]

    return node_feat, pos, edge_index, edge_attr, atomic_mass, target, graph_batch, graph_idx


@torch.no_grad()
def compute_target_stats(loader, target_idx, device):
    total    = torch.tensor(0.0, device=device)
    total_sq = torch.tensor(0.0, device=device)
    count    = 0
    for batch in loader:
        y = batch.y[:, target_idx].to(device)
        total    = total + y.sum()
        total_sq = total_sq + (y ** 2).sum()
        count    += y.numel()   # .numel() is shape metadata, not a sync
    mean = total / max(count, 1)
    var  = (total_sq / max(count, 1) - mean ** 2).clamp(min=1e-12)
    std  = var.sqrt()
    return mean, std


def train_epoch(target_idx, model, loader, optimizer, min_nodes, device, accum_steps, epoch,
                 target_mean, target_std, monitor: SystemMonitor = None):
    model.train()
    total_mae   = torch.tensor(0.0, device=device)   # stays on-device all epoch; synced once at the end
    n_graphs    = 0
    accum_loss  = torch.tensor(0.0, device=device)

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch   = batch.to(device)
        tensors = _filter_small_graphs(target_idx, batch, min_nodes, device)
        if tensors is None:
            continue

        node_feat, pos, edge_index, edge_attr, atomic_mass, target, graph_batch, graph_idx = tensors

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch,
                     edge_attr=edge_attr, graph_idx=graph_idx)
        target_norm = (target - target_mean) / target_std
        loss = nn.functional.l1_loss(pred.squeeze(-1), target_norm) / accum_steps
        loss.backward()

        if monitor is not None:
            monitor.sample()

        with torch.no_grad():
            pred_denorm = pred.squeeze(-1) * target_std + target_mean
            mae = nn.functional.l1_loss(pred_denorm, target)   # tensor, no sync

        accum_loss = accum_loss + mae / accum_steps

        B = pred.shape[0]   # shape metadata, not a sync
        total_mae = total_mae + mae * B
        n_graphs  += B

        step_idx = i + 1
        if step_idx % accum_steps == 0:
            if monitor is not None:
                monitor.record_grad_stats(model)
            optimizer.step()
            optimizer.zero_grad()

            if monitor is not None:
                monitor.commit(accum_loss.item())   # one sync per accum window, not per micro-batch
            accum_loss = torch.tensor(0.0, device=device)
        else:
            print(f"Batch {step_idx} | micro-batch {step_idx % accum_steps}/{accum_steps}")

    remainder = len(loader) % accum_steps
    if remainder != 0:
        if monitor is not None:
            monitor.record_grad_stats(model)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Flushed {remainder} remaining micro-batch(es).")
        if monitor is not None:
            monitor.commit(accum_loss.item())

    total_mae = total_mae.item()   # single sync for the whole epoch

    print(f"Epoch: {epoch + 1} | Epoch MAE: {total_mae/max(n_graphs, 1):.4f}")
    return total_mae / max(n_graphs, 1)


@torch.no_grad()
def evaluate(target_idx, model, loader, min_nodes, device, target_mean, target_std, epoch=None):
    model.eval()
    total_mae = torch.tensor(0.0, device=device)   # synced once at the end, not per batch
    n_graphs  = 0

    PRINT_EVERY = 10   # only sync for a printed value this often, not every batch

    for i, batch in enumerate(loader):
        batch   = batch.to(device)
        tensors = _filter_small_graphs(target_idx, batch, min_nodes, device)
        if tensors is None:
            continue

        node_feat, pos, edge_index, edge_attr, atomic_mass, target, graph_batch, graph_idx = tensors

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch,
                     edge_attr=edge_attr, graph_idx=graph_idx)
        pred_denorm = pred.squeeze(-1) * target_std + target_mean
        mae  = nn.functional.l1_loss(pred_denorm, target)   # tensor, no sync
        B    = pred.shape[0]
        total_mae = total_mae + mae * B
        n_graphs  += B

        if (i + 1) % PRINT_EVERY == 0:
            print(f"[Eval Batch {i+1}] MAE: {mae.item():.4f}")

    total_mae = total_mae.item()   # single sync for the whole call

    if epoch:
        print(f"Epoch: {epoch + 1} | Epoch Eval MAE: {total_mae/max(n_graphs, 1):.4f}")
    else:
        print(f"Eval MAE: {total_mae / max(n_graphs, 1):.4f}")

    return total_mae / max(n_graphs, 1)


def confidence_interval_95(values):
    n    = len(values)
    mean = np.mean(values)
    std  = np.std(values, ddof=1)
    from scipy import stats
    t_crit     = stats.t.ppf(0.975, df=n - 1)
    half_width = t_crit * std / math.sqrt(n)
    return mean, half_width


def main(rbf_type="grbf"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int,   default=1)
    parser.add_argument("--target_indices", type=list, default=target_indices, help="Target indices for multiple runs")
    parser.add_argument("--radius_cutoff", type=float, default=5.0,
                        help="Atoms within this distance (Angstrom) attend to each other.")
    parser.add_argument("--max_degree",  type=int,   default=2)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--accum_steps", type=int,   default=4,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size * accum_steps.")
    parser.add_argument("--rbf_type", type=str, default="grbf")
    parser.add_argument("--activation_checkpointing", action="store_true",
                        help="Recompute each layer's activations during backward instead "
                             "of keeping all of them resident -- trades ~30% extra compute "
                             "for much lower peak memory at this depth.")
    parser.add_argument("--feature_dim", type=int,   default=32)
    parser.add_argument("--hidden_dim",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--min_lr",      type=float, default=1e-4,
                        help="Floor of the single-cycle cosine LR decay "
                             "(CosineAnnealingLR eta_min).")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--trials",      type=int,   default=1)
    parser.add_argument("--data_root",   type=str,   default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--metrics_dir", type=str,   default="./training_metrics",
                        help="Directory for per-trial loss/GPU/CPU/memory "
                             "utilization CSVs and plots.")
    parser.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dummy_data",  action="store_true",
                        help="Use synthetic in-memory molecules instead of load_qm9, "
                             "for smoke-testing the pipeline without real QM9 data on disk.")
    parser.add_argument("--dummy_batches", type=int, default=6,
                        help="Number of training batches to synthesize when --dummy_data "
                             "is set (val/test get roughly a third of this each).")
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)

    device = torch.device(args.device)
    effective_batch = args.batch_size * args.accum_steps
    print(f"Device:           {device}")
    print(f"QM9 target:       {args.target}")
    print(f"Physical batch:   {args.batch_size}")
    print(f"Accum steps:      {args.accum_steps}")
    print(f"Effective batch:  {effective_batch}")

    if args.dummy_data:
        train_loader, val_loader, test_loader = load_qm9_dummy(
            args.batch_size, args.data_root, args.device, num_batches=args.dummy_batches
        )
    else:
        train_loader, val_loader, test_loader = load_qm9(
            args.batch_size, args.data_root, args.device
        )

    target_mean, target_std = compute_target_stats(train_loader, args.target, device)
    print(f"Target stats:     mean={target_mean.item():.4f}, std={target_std.item():.4f}")

    trial_maes = []
    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1} / {args.trials} ---")
        torch.manual_seed(trial)

        checkpoint_path = os.path.join(
            args.metrics_dir,
            f"best_model_trial{trial}_target{args.target}.pt"
        )

        if args.rbf_type == "grbf":
            radial_net = RadialNetworkGRBF
        elif args.rbf_type == "gsfb":
            radial_net = RadialNetworkGSFB
        else:
            radial_net = RadialNetworkGRBF

        model = SE3IntraOnlyTransformer(
            radial_net=radial_net,
            in_features=len(ATOM_TYPES),
            max_degree=args.max_degree,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            radius_cutoff=args.radius_cutoff,
            scalar_out_dim=1,
            task=0,
            bond_feature_dim=len(BOND_TYPES),
            use_checkpointing=args.activation_checkpointing,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        monitor = SystemMonitor(device)

        best_val_mae = float("inf")
        for epoch in range(args.epochs):
            print(f"[Epoch {epoch + 1}]")
            train_loss = train_epoch(
                args.target, model, train_loader, optimizer, 1, device,
                accum_steps=args.accum_steps, epoch=epoch,
                target_mean=target_mean, target_std=target_std, monitor=monitor,
            )
            val_mae = evaluate(
                args.target, model, val_loader, 1, device,
                target_mean=target_mean, target_std=target_std, epoch=epoch,
            )
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
            title_prefix=f"RC={args.radius_cutoff}, RBF={args.rbf_type}",
        )

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_mae = evaluate(
            args.target, model, test_loader, 1, device,
            target_mean=target_mean, target_std=target_std,
        )
        print(f"  Trial {trial + 1} test MAE: {test_mae:.4f}")
        trial_maes.append(test_mae)

    mean, hw = confidence_interval_95(trial_maes)
    print(f"\nTest MAE over {args.trials} trials: {mean:.4f} ± {hw:.4f}  (95% CI)")

if __name__ == "__main__":
    main()