import argparse
import os

import h5py
import torch
import torch.nn as nn
from rdkit import Chem
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.se3_crossformer.model import SH1_TO_CARTESIAN, SH2_TO_CARTESIAN, SE3IntraOnlyTransformer
from src.se3_crossformer.se3_utils import *
from src.training_monitor import SystemMonitor

ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06
}
ATOM_TYPES = [1, 6, 7, 8, 9]

# Mirrors src/load_data.py's BOND_TYPES ordering for QM9 -- kept identical so
# a one-hot bond feature means the same thing across both datasets.
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

_warned_no_smiles = False


def get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [ATOMIC_MASSES.get(zi.item(), 12.0) for zi in z],
        dtype=torch.float32,
    )

def one_hot_z(z: torch.Tensor) -> torch.Tensor:
    one_hot = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for idx, a in enumerate(ATOM_TYPES):
        one_hot[:, idx] = (z == a).float()
    return one_hot


def _build_edge_attr_from_smiles(smiles: str, edge_index: torch.Tensor, n_atoms: int):
    """Best-effort bond-order/type one-hot for each edge in `edge_index`, built
    by parsing the molecule's SMILES with RDKit -- mirrors src/load_data.py's
    CustomQM9Dataset pattern. Returns None if the SMILES is missing, fails to
    parse, or its atom count doesn't match `n_atoms` (i.e. atom-index
    alignment between the RDKit mol and the HDF5 pos/z arrays can't be
    trusted), so callers should treat a None return as "no bond features
    available for this molecule" rather than an error.
    """
    global _warned_no_smiles
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != n_atoms:
        # Atom count mismatch means RDKit's atom ordering can't be trusted to
        # line up with the HDF5 z/pos ordering -- skip rather than guess.
        if not _warned_no_smiles:
            print(
                "Warning: RDKit atom count from SMILES doesn't match HDF5 "
                "atom count for at least one molecule -- bond features will "
                "be skipped for those molecules. This needs verifying against "
                "the real QMe14S HDF5 schema (atom-index alignment between "
                "the stored 'smile' attribute and pos/z is assumed, not confirmed)."
            )
            _warned_no_smiles = True
        return None

    bond_lookup = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        onehot = [1.0 if bond_type == t else 0.0 for t in BOND_TYPES]
        bond_lookup[(i, j)] = onehot
        bond_lookup[(j, i)] = onehot

    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    edge_features = [
        bond_lookup.get((i, j), [0.0] * len(BOND_TYPES))
        for i, j in zip(src, dst)
    ]
    return torch.tensor(edge_features, dtype=torch.float32)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, field: str):
        self.samples = []
        with h5py.File(h5_path, "r") as h5file:
            for group_name in h5file.keys():
                group = h5file[group_name]
                try:
                    edge_index = torch.tensor(group["edge_index"][:], dtype=torch.long)
                    pos = torch.tensor(group["pos"][:], dtype=torch.float32)
                    z = torch.tensor(group["z"][:], dtype=torch.long)
                    if field == "polar":
                        polar = torch.tensor(group["polar"][:], dtype=torch.float32).reshape(3, 3)
                    else:
                        n_atoms = z.shape[0]
                        dedipole = torch.tensor(group["dedipole"][:], dtype=torch.float32).reshape(n_atoms, 3, 3)
                except KeyError as e:
                    print(f"Missing key {e} in group {group_name}. Skipping this group.")
                    continue
                except Exception as e:
                    print(f"Unexpected error in group {group_name}: {e}. Skipping this group.")
                    continue

                target = polar if field == "polar" else dedipole

                smiles = group.attrs.get("smile", group.attrs.get("smiles"))
                if smiles is not None and not isinstance(smiles, str):
                    smiles = smiles.decode("utf-8") if isinstance(smiles, bytes) else str(smiles)
                edge_attr = _build_edge_attr_from_smiles(smiles, edge_index, z.shape[0])

                self.samples.append(
                    Data(edge_index=edge_index, edge_attr=edge_attr, pos=pos, z=z, y=target)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def load_data(h5_path: str, batch_size: int, field: str, val_frac=0.1, test_frac=0.1):
    dataset = HDF5Dataset(h5_path, field)
    n = len(dataset)
    perm = torch.randperm(n)

    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

def _flat_to_traceless_symmetric(flat: torch.Tensor) -> torch.Tensor:
    Qxx, Qxy, Qxz, Qyy, Qyz = flat.unbind(-1)
    Qzz = -Qxx - Qyy
    B = flat.shape[0]
    T = torch.zeros(B, 3, 3, device=flat.device, dtype=flat.dtype)
    T[:, 0, 0] = Qxx
    T[:, 1, 1] = Qyy
    T[:, 2, 2] = Qzz
    T[:, 0, 1] = T[:, 1, 0] = Qxy
    T[:, 0, 2] = T[:, 2, 0] = Qxz
    T[:, 1, 2] = T[:, 2, 1] = Qyz
    return T

class PolarizabilityHead(nn.Module):
    per_atom = False

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        self.irrep_lin0 = IrrepLinear(feature_dim)
        self.irrep_lin1 = IrrepLinear(feature_dim)
        self.register_buffer("sh1_to_cartesian", SH1_TO_CARTESIAN.clone())
        self.iso_linear = nn.Linear(feature_dim, 1)

        self.irrep_lin2 = IrrepLinear(feature_dim)
        self.aniso_readout = EquivariantReadout(feature_dim, hidden_dim)
        self.register_buffer("sh2_to_cartesian", SH2_TO_CARTESIAN.clone())

        self.register_buffer("identity3", torch.eye(3))

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B = f0.shape[0]

        f0m = self.irrep_lin0(f0).squeeze(-1)
        alpha_iso = self.iso_linear(f0m)
        iso_tensor = alpha_iso.view(B, 1, 1) * self.identity3.unsqueeze(0)

        f2m = self.irrep_lin2(f2)
        out2_sh = self.aniso_readout(f2m)
        aniso_flat = torch.einsum("ij,bj->bi", self.sh2_to_cartesian, out2_sh)
        aniso_tensor = _flat_to_traceless_symmetric(aniso_flat)

        return iso_tensor + aniso_tensor

class DeDipoleHead(nn.Module):
    per_atom = True

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        self.irrep_lin0 = IrrepLinear(feature_dim)
        self.irrep_lin1 = IrrepLinear(feature_dim)
        self.register_buffer("sh1_to_cartesian", SH1_TO_CARTESIAN.clone())
        self.iso_linear = nn.Linear(feature_dim, 1)
        self.register_buffer("identity3", torch.eye(3))

        self.irrep_lin11 = IrrepLinear(feature_dim)
        self.skew_red = EquivariantReadout(feature_dim)

        self.irrep_lin2 = IrrepLinear(feature_dim)
        self.aniso_readout = EquivariantReadout(feature_dim, hidden_dim)
        self.register_buffer("sh2_to_cartesian", SH2_TO_CARTESIAN.clone())

    def f1_to_skew(self, f1_cart):
        vx, vy, vz = f1_cart[..., 0], f1_cart[..., 1], f1_cart[..., 2]
        zeros = torch.zeros_like(vx)
        out = torch.stack(
            [
                torch.stack([zeros, -vz, vy], dim=-1),
                torch.stack([vz, zeros, -vx], dim=-1),
                torch.stack([-vy, vx, zeros], dim=-1),
            ],
            dim=-2,
        )
        return out

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        N = f0.shape[0]

        f0m = self.irrep_lin0(f0).squeeze(-1)
        alpha_iso = self.iso_linear(f0m)
        iso_tensor = alpha_iso.view(N, 1, 1) * self.identity3.unsqueeze(0)

        f1_skew_emb = self.irrep_lin11(f1)
        f1_skew_red = self.skew_red(f1_skew_emb).squeeze(1)
        v1_skew = torch.einsum("ij,bj->bi", self.sh1_to_cartesian, f1_skew_red)
        skew_tensor = self.f1_to_skew(v1_skew)

        f2m = self.irrep_lin2(f2)
        out2_sh = self.aniso_readout(f2m)
        aniso_flat = torch.einsum("ij,bj->bi", self.sh2_to_cartesian, out2_sh)
        aniso_tensor = _flat_to_traceless_symmetric(aniso_flat)

        return iso_tensor + skew_tensor + aniso_tensor

class _ForwardMixin:
    def forward(self, node_features, x, edge_index, atomic_masses, batch,
                edge_attr=None, vec_feat=None, tensor_feat=None):
        B = int(batch.max().item()) + 1

        f0 = self.input_embedding(node_features).unsqueeze(-1)
        seeded = self._seed_equivariant_features(f0, x, atomic_masses, batch)
        f = {0: f0, 1: seeded[1], 2: seeded[2]}

        counts_per_graph = torch.bincount(batch, minlength=B)
        ptr = [0] + torch.cumsum(counts_per_graph, dim=0).tolist()

        num_bond_feats = edge_attr.shape[-1] if edge_attr is not None else 0

        per_graph = []
        K_global = 0
        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g = hi - lo

            pos_g = x[lo:hi]
            mask_e = (edge_index[0] >= lo) & (edge_index[0] < hi)
            ei_g = edge_index[:, mask_e] - lo
            edge_attr_g = edge_attr[mask_e] if edge_attr is not None else None

            nidx_local, nmask = self._build_radius_neighbor_info(pos_g, n_g)

            if self.use_bond_info and edge_attr_g is not None:
                bond_lookup = torch.zeros(
                    n_g, n_g, num_bond_feats, device=x.device, dtype=edge_attr_g.dtype
                )
                bond_lookup[ei_g[0], ei_g[1]] = edge_attr_g
                nbond_local = bond_lookup[
                    torch.arange(n_g, device=x.device).unsqueeze(1), nidx_local
                ]
                nbond_local = nbond_local * nmask.unsqueeze(-1).to(nbond_local.dtype)
            else:
                nbond_local = None

            per_graph.append((nidx_local + lo, nmask, nbond_local))
            K_global = max(K_global, nidx_local.shape[1])

        neighbor_idx_list, neighbor_mask_list, neighbor_bond_list = [], [], []
        for nidx, nmask, nbond in per_graph:
            K_g = nidx.shape[1]
            if K_g < K_global:
                pad_idx = torch.zeros(nidx.shape[0], K_global - K_g, dtype=torch.long, device=x.device)
                pad_mask = torch.zeros(nmask.shape[0], K_global - K_g, dtype=torch.bool, device=x.device)
                nidx = torch.cat([nidx, pad_idx], dim=1)
                nmask = torch.cat([nmask, pad_mask], dim=1)
                if nbond is not None:
                    pad_bond = torch.zeros(nbond.shape[0], K_global - K_g, num_bond_feats,
                                           dtype=nbond.dtype, device=x.device)
                    nbond = torch.cat([nbond, pad_bond], dim=1)
            neighbor_idx_list.append(nidx)
            neighbor_mask_list.append(nmask)
            if nbond is not None:
                neighbor_bond_list.append(nbond)

        neighbor_idx = torch.cat(neighbor_idx_list, dim=0)
        neighbor_mask = torch.cat(neighbor_mask_list, dim=0)
        neighbor_bond_attr = torch.cat(neighbor_bond_list, dim=0) if neighbor_bond_list else None

        f, _ = self.layer(
            f_in=f, x=x, neighbor_idx=neighbor_idx, neighbor_mask=neighbor_mask,
            neighbor_bond_attr=neighbor_bond_attr,
        )

        if getattr(self.head, "per_atom", False):
            return self.head(f[0], f[1], f[2])

        from torch_scatter import scatter_mean
        f0_pooled = scatter_mean(f[0], batch, dim=0, dim_size=B)
        f1_pooled = scatter_mean(f[1], batch, dim=0, dim_size=B)
        f2_pooled = scatter_mean(f[2], batch, dim=0, dim_size=B)

        return self.head(f0_pooled, f1_pooled, f2_pooled)

class SE3PolarizabilityIntraTransformer(_ForwardMixin, SE3IntraOnlyTransformer):
    def __init__(self, radial_net, in_features, max_degree=2,
                 feature_dim=32, hidden_dim=64, radius_cutoff=5.0, use_bond_info=True):
        super().__init__(
            radial_net=radial_net, in_features=in_features, max_degree=max_degree,
            feature_dim=feature_dim, hidden_dim=hidden_dim,
            radius_cutoff=radius_cutoff, scalar_out_dim=1, task=3,
            use_bond_info=use_bond_info,
        )
        self.head = PolarizabilityHead(feature_dim, hidden_dim)

class SE3DeDipoleIntraTransformer(_ForwardMixin, SE3IntraOnlyTransformer):
    def __init__(self, radial_net, in_features, max_degree=2,
                 feature_dim=32, hidden_dim=64, radius_cutoff=5.0, use_bond_info=True):
        super().__init__(
            radial_net=radial_net, in_features=in_features, max_degree=max_degree,
            feature_dim=feature_dim, hidden_dim=hidden_dim,
            radius_cutoff=radius_cutoff, scalar_out_dim=1, task=3,
            use_bond_info=use_bond_info,
        )
        self.head = DeDipoleHead(feature_dim, hidden_dim)

def _prepare_batch(batch, device, field="polar"):
    graph_batch = batch.batch.to(device)

    y = batch.y.to(device)
    target = y.view(-1, 3, 3) if field == "polar" else y

    return (
        one_hot_z(batch.z).to(device),
        batch.pos.to(device),
        batch.edge_index.to(device),
        get_atomic_masses(batch.z).to(device),
        target,
        graph_batch,
        batch.edge_attr.to(device) if getattr(batch, "edge_attr", None) is not None else None,
    )


def train_epoch(model, loader, optimizer, device, accum_steps, epoch,
                 monitor: SystemMonitor = None, field: str = "polar"):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    accum_loss = torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        node_feat, pos, edge_index, atomic_mass, target, graph_batch, edge_attr = _prepare_batch(
            batch, device, field=field
        )

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch, edge_attr=edge_attr)
        loss = nn.functional.mse_loss(pred, target) / accum_steps
        loss.backward()

        if monitor is not None:
            monitor.sample()

        accum_loss = accum_loss + loss.detach()

        B = pred.shape[0]
        total_loss += loss.item() * accum_steps * B
        n_graphs += B

        step_idx = i + 1
        if step_idx % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if monitor is not None:
                monitor.commit(step_idx, accum_loss.item())
            accum_loss = torch.tensor(0.0, device=device)

    remainder = len(loader) % accum_steps
    if remainder != 0:
        optimizer.step()
        optimizer.zero_grad()
        print(f"Flushed {remainder} remaining micro-batch(es).")
        if monitor is not None:
            monitor.commit(len(loader), accum_loss.item())

    epoch_loss = total_loss / max(n_graphs, 1)
    print(f"Epoch: {epoch + 1} | Epoch MSE: {epoch_loss:.4f}")
    return epoch_loss

@torch.no_grad()
def evaluate(model, loader, device, epoch=None, field: str = "polar"):
    model.eval()
    total_mse = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        node_feat, pos, edge_index, atomic_mass, target, graph_batch, edge_attr = _prepare_batch(
            batch, device, field=field
        )

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch, edge_attr=edge_attr)
        mse = nn.functional.mse_loss(pred, target).item()
        B = pred.shape[0]
        total_mse += mse * B
        n_graphs += B

    epoch_mse = total_mse / max(n_graphs, 1)
    if epoch is not None:
        print(f"Epoch: {epoch + 1} | Epoch Eval MSE: {epoch_mse:.4f}")
    else:
        print(f"Eval MSE: {epoch_mse:.4f}")
    return epoch_mse

def build_model(args, device):
    radial_net = RadialNetworkGSFB
    cls = SE3PolarizabilityIntraTransformer if args.field == "polar" else SE3DeDipoleIntraTransformer
    return cls(
        radial_net=radial_net,
        in_features=len(ATOM_TYPES),
        max_degree=args.max_degree,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        radius_cutoff=args.radius_cutoff,
    ).to(device)

def run_training(args, device, train_loader, val_loader, test_loader):
    print(f"\n=== Training: prediction_target={args.field!r} ===")
    torch.manual_seed(args.seed)

    model = build_model(args, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    monitor = SystemMonitor(device)

    checkpoint_path = os.path.join(args.metrics_dir, "best_model.pt")
    best_val_mse = float("inf")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        train_epoch(model, train_loader, optimizer, device,
                    accum_steps=args.accum_steps, epoch=epoch, monitor=monitor, field=args.field)
        val_mse = evaluate(model, val_loader, device, epoch, field=args.field)
        scheduler.step()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"  [New best] val MSE: {val_mse:.4f} -- saving checkpoint.")
            torch.save(model.state_dict(), checkpoint_path)

    monitor.save_csv(os.path.join(args.metrics_dir, "metrics.csv"))
    monitor.save_plots(
        os.path.join(args.metrics_dir, "metrics.png"),
        title_prefix=f"field={args.field}",
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_mse = evaluate(model, test_loader, device, field=args.field)
    print(f"Test MSE: {test_mse:.4f}")
    return {"best_val_mse": best_val_mse, "test_mse": test_mse}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="./data/QMe14S_single_point.h5")
    parser.add_argument("--field", type=str, default="polar", choices=["polar", "dedipole"])
    parser.add_argument("--radius_cutoff", type=float, default=5.0)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--rbf_type", type=str, default="grbf")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics_dir", type=str, default="./training_metrics_polarizability")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device:      {device}")
    print(f"HDF5 file:   {args.h5_path}")
    print(f"Field:       {args.field}")
    print(f"Batch size:  {args.batch_size}  x accum {args.accum_steps} = "
          f"{args.batch_size * args.accum_steps} effective")

    train_loader, val_loader, test_loader = load_data(args.h5_path, args.batch_size, args.field)

    result = run_training(args, device, train_loader, val_loader, test_loader)
    print(f"\n=== Summary === best val MSE: {result['best_val_mse']:.4f} | test MSE: {result['test_mse']:.4f}")

if __name__ == "__main__":
    main()
