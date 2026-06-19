"""
train.py
--------
Training loop and QM9 regression experiment (Section 4.2 of the manuscript).

Usage:
    python train.py --target 0 --num_parts 8 --max_degree 2 --epochs 300

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
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.se3_crossformer.model import SE3InterNeighborhoodTransformer

from src.load_data import CustomQM9Dataset

# TODO: Extend or replace with a more complete lookup if heavier atoms appear.
ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06
}

def get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    """Map atomic number tensor [N] to mass tensor [N]."""
    return torch.tensor(
        [ATOMIC_MASSES.get(zi.item(), 12.0) for zi in z],
        dtype=torch.float32,
    )

def load_qm9(target_idx: int, batch_size=16, r: str = "./data", device=torch.device("cpu")):
    """
    Load QM9 dataset via torch_geometric.

    Returns train/val/test DataLoader objects.

    # TODO: Adjust split sizes to match the SE(3)-Transformer paper's protocol
    #       (110k / 10k / ~10k) if you want a direct comparison.
    """
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

ATOM_TYPES = [1, 6, 7, 8, 9]   # H, C, N, O, F  (QM9 atoms)

def one_hot_z(z: torch.Tensor) -> torch.Tensor:
    """Atomic number [N] -> one-hot [N, len(ATOM_TYPES)]."""
    one_hot = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for idx, a in enumerate(ATOM_TYPES):
        one_hot[:, idx] = (z == a).float()
    return one_hot

def train_epoch(model, loader, optimizer, num_parts, device):
    model.train()
    total_loss = 0.0
    n_graphs   = 0

    for i, batch in enumerate(loader):
        print(f"[Batch]: {i+1}")
        batch = batch.to(device)

        node_feat   = one_hot_z(batch.x).to(device)       
        pos         = batch.pos.to(device)                  
        edge_index  = batch.edge_index.to(device)
        atomic_mass = get_atomic_masses(batch.x).to(device)
        target      = batch.y.to(device)                   
        graph_batch = batch.batch.to(device)               

        counts      = torch.bincount(graph_batch)          
        valid_graph = counts >= num_parts                  
        if not valid_graph.all():
            keep_nodes  = valid_graph[graph_batch]
            keep_graphs = valid_graph.nonzero(as_tuple=True)[0]
            remap       = torch.full((valid_graph.shape[0],), -1,
                                     dtype=torch.long, device=device)
            remap[keep_graphs] = torch.arange(keep_graphs.shape[0], device=device)

            node_mask   = keep_nodes
            node_feat   = node_feat[node_mask]
            pos         = pos[node_mask]
            atomic_mass = atomic_mass[node_mask]
            graph_batch = remap[graph_batch[node_mask]]
            target      = target[keep_graphs]

            src, dst    = edge_index
            edge_mask   = node_mask[src] & node_mask[dst]
            old_to_new  = torch.full((batch.num_nodes,), -1,
                                     dtype=torch.long, device=device)
            old_to_new[node_mask.nonzero(as_tuple=True)[0]] = \
                torch.arange(node_mask.sum(), device=device)
            edge_index  = old_to_new[edge_index[:, edge_mask]]

        if graph_batch.max() < 0:  
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)  
        loss = nn.functional.l1_loss(pred.squeeze(-1), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = pred.shape[0]
        total_loss += loss.item() * B
        n_graphs   += B
        print(f"Batch {i+1} Loss: {loss.item():.4f}")

    return total_loss / max(n_graphs, 1)

@torch.no_grad()
def evaluate(model, loader, num_parts, device):
    model.eval()
    total_mae = 0.0
    n_graphs  = 0

    for i, batch in enumerate(loader):
        print(f"[Eval Batch]: {i+1}")
        batch = batch.to(device)

        node_feat   = one_hot_z(batch.x).to(device)
        pos         = batch.pos.to(device)
        edge_index  = batch.edge_index.to(device)
        atomic_mass = get_atomic_masses(batch.x).to(device)
        target      = batch.y.to(device)
        graph_batch = batch.batch.to(device)

        counts      = torch.bincount(graph_batch)
        valid_graph = counts >= num_parts
        if not valid_graph.all():
            keep_nodes  = valid_graph[graph_batch]
            keep_graphs = valid_graph.nonzero(as_tuple=True)[0]
            remap       = torch.full((valid_graph.shape[0],), -1,
                                     dtype=torch.long, device=device)
            remap[keep_graphs] = torch.arange(keep_graphs.shape[0], device=device)

            node_mask   = keep_nodes
            node_feat   = node_feat[node_mask]
            pos         = pos[node_mask]
            atomic_mass = atomic_mass[node_mask]
            graph_batch = remap[graph_batch[node_mask]]
            target      = target[keep_graphs]

            src, dst   = edge_index
            edge_mask  = node_mask[src] & node_mask[dst]
            old_to_new = torch.full((batch.num_nodes,), -1,
                                    dtype=torch.long, device=device)
            old_to_new[node_mask.nonzero(as_tuple=True)[0]] = \
                torch.arange(node_mask.sum(), device=device)
            edge_index = old_to_new[edge_index[:, edge_mask]]

        if graph_batch.max() < 0:
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)
        mae  = nn.functional.l1_loss(pred.squeeze(-1), target).item()
        B    = pred.shape[0]
        total_mae += mae * B
        n_graphs  += B
        print(f"Batch {i+1} MAE: {mae:.4f}")

    return total_mae / max(n_graphs, 1)

def confidence_interval_95(values):
    """Compute mean and 95% CI half-width from a list of values."""
    n    = len(values)
    mean = np.mean(values)
    std  = np.std(values, ddof=1)
    # t critical value for 95% CI, df=n-1
    from scipy import stats  # type: ignore
    t_crit = stats.t.ppf(0.975, df=n - 1)
    half_width = t_crit * std / math.sqrt(n)
    return mean, half_width

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=int,   default=1,   help="QM9 target index (0-11)")
    parser.add_argument("--num_parts",  type=int,   default=4,   help="Number of spectral subgraphs")
    parser.add_argument("--max_degree", type=int,   default=2,   help="Max SE(3) irrep degree")
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--feature_dim",type=int,   default=32)
    parser.add_argument("--hidden_dim", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--trials",     type=int,   default=5,   help="Trials for CI")
    parser.add_argument("--data_root",  type=str,   default="./data")
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"QM9 target: {args.target}")

    train_loader, val_loader, test_loader = load_qm9(args.target, args.batch_size, args.data_root, args.device)

    trial_maes = []
    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1} / {args.trials} ---")
        torch.manual_seed(trial)

        model = SE3InterNeighborhoodTransformer(
            in_features=len(ATOM_TYPES),
            max_degree=args.max_degree,
            num_layers=args.num_layers,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_parts=args.num_parts,
            out_dim=19,
            task="regression",
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_mae = float("inf")
        for epoch in range(args.epochs):
            print(f"[Epoch]: {epoch+1}")
            train_loss = train_epoch(model, train_loader, optimizer, args.num_parts, device)
            val_mae    = evaluate(model, val_loader, args.num_parts, device)
            scheduler.step()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"[Saving model with MAE loss]: {val_mae}")
                torch.save(model.state_dict(), f"best_model_trial{trial}.pt")

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train MAE: {train_loss:.4f} | val MAE: {val_mae:.4f}")

        model.load_state_dict(torch.load(f"best_model_trial{trial}.pt", map_location=device))
        test_mae = evaluate(model, test_loader, args.num_parts, device)
        print(f"  Trial {trial + 1} test MAE: {test_mae:.4f}")
        trial_maes.append(test_mae)

    mean, hw = confidence_interval_95(trial_maes)
    print(f"\nTest MAE over {args.trials} trials: {mean:.4f} ± {hw:.4f}  (95% CI)")


if __name__ == "__main__":
    main()