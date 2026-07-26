import argparse
import os

import h5py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.se3_crossformer.model import (
    IrrepLinear,
    EquivariantReadout,
    SH1_TO_CARTESIAN,
    SH2_TO_CARTESIAN,
    SE3InterNeighborhoodTransformer,
    SE3IntraOnlyTransformer,
)
from src.se3_crossformer.spectral_partition import subgraph_center_of_mass
from src.se3_crossformer.se3_utils import RadialNetworkGSFB
from src.training_monitor import SystemMonitor

ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06
}
ATOM_TYPES = [1, 6, 7, 8, 9]


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
                self.samples.append(Data(edge_index=edge_index, pos=pos, z=z, y=target))

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
        alpha_iso = self.iso_linear(iso_in)
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
        self.skew_red = nn.Linear(feature_dim, 1)

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
        f1_skew_red = self.skew_red(f1_skew_emb.transpose(-1, -2)).transpose(-1, -2)
        v1_skew = torch.einsum("ij,bcj->bci", self.sh1_to_cartesian, f1_skew_red)
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

        ptr = [0]
        for g in range(B):
            ptr.append(int((batch <= g).sum().item()))

        node_to_subgraph_list, x_cm_list, subgraph_mask_list = [], [], []
        neighbor_idx_list, neighbor_mask_list = [], []

        subgraph_offset = 0
        K_global = 0
        per_graph = []
        partition_key = self._partition_config_key()

        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g = hi - lo

            pos_g = x[lo:hi]
            mass_g = atomic_masses[lo:hi]
            mask_e = (edge_index[0] >= lo) & (edge_index[0] < hi)
            ei_g = edge_index[:, mask_e] - lo
            edge_attr_g = edge_attr[mask_e] if edge_attr is not None else None

            cache_key = self._edge_hash(ei_g.cpu(), n_g, partition_key)
            if cache_key in self._graph_cache:
                n2s_local, xcm, smask, nidx_local, nmask = self._graph_cache[cache_key]
                xcm = subgraph_center_of_mass(pos_g, mass_g, n2s_local, self.num_parts)
            else:
                n2s_local, xcm, smask = self._build_subgraph_info(
                    ei_g, n_g, pos_g, mass_g, edge_attr=edge_attr_g
                )
                nidx_local, nmask = self._build_neighbor_info(ei_g, n2s_local, n_g)
                self._graph_cache[cache_key] = (n2s_local, xcm, smask, nidx_local, nmask)

            per_graph.append((
                n2s_local + subgraph_offset, xcm, smask,
                nidx_local + lo, nmask,
            ))
            subgraph_offset += self.num_parts
            K_global = max(K_global, nidx_local.shape[1])

        for g, (n2s, xcm, smask, nidx, nmask) in enumerate(per_graph):
            K_g = nidx.shape[1]
            if K_g < K_global:
                pad_idx = torch.zeros(nidx.shape[0], K_global - K_g, dtype=torch.long, device=x.device)
                pad_mask = torch.zeros(nmask.shape[0], K_global - K_g, dtype=torch.bool, device=x.device)
                nidx = torch.cat([nidx, pad_idx], dim=1)
                nmask = torch.cat([nmask, pad_mask], dim=1)
            node_to_subgraph_list.append(n2s)
            x_cm_list.append(xcm)
            subgraph_mask_list.append(smask)
            neighbor_idx_list.append(nidx)
            neighbor_mask_list.append(nmask)

        node_to_subgraph = torch.cat(node_to_subgraph_list, dim=0)
        x_cm_all = torch.cat(x_cm_list, dim=0)
        neighbor_idx = torch.cat(neighbor_idx_list, dim=0)
        neighbor_mask = torch.cat(neighbor_mask_list, dim=0)

        S_total = B * self.num_parts
        subgraph_mask_all = torch.zeros(S_total, S_total, dtype=torch.bool, device=x.device)
        for g, smask in enumerate(subgraph_mask_list):
            lo = g * self.num_parts
            hi = lo + self.num_parts
            subgraph_mask_all[lo:hi, lo:hi] = smask

        for layer in self.layers:
            f, _ = layer(
                f_in=f, x=x, neighbor_idx=neighbor_idx, neighbor_mask=neighbor_mask,
                x_cm=x_cm_all, node_to_subgraph=node_to_subgraph,
                subgraph_mask=subgraph_mask_all,
            )

        if getattr(self.head, "per_atom", False):
            return self.head(f[0], f[1], f[2])

        from torch_scatter import scatter_mean
        f0_pooled = scatter_mean(f[0], batch, dim=0, dim_size=B)
        f1_pooled = scatter_mean(f[1], batch, dim=0, dim_size=B)
        f2_pooled = scatter_mean(f[2], batch, dim=0, dim_size=B)

        return self.head(f0_pooled, f1_pooled, f2_pooled)

class SE3PolarizabilityInterTransformer(_ForwardMixin, SE3InterNeighborhoodTransformer):
    def __init__(self, radial_net, in_features, max_degree=2, num_layers=4,
                 feature_dim=32, hidden_dim=64, num_parts=4,
                 partition_type="spectral", knn_k=10, use_bond_info=True,
                 bond_order_power=1.0, use_connectivity_features=False):
        super().__init__(
            radial_net=radial_net, in_features=in_features, max_degree=max_degree,
            num_layers=num_layers, feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_parts=num_parts, scalar_out_dim=1, task=3,
            partition_type=partition_type, knn_k=knn_k, use_bond_info=use_bond_info,
            bond_order_power=bond_order_power, use_connectivity_features=use_connectivity_features,
        )
        self.head = PolarizabilityHead(feature_dim, hidden_dim)

class SE3PolarizabilityIntraTransformer(_ForwardMixin, SE3IntraOnlyTransformer):
    def __init__(self, radial_net, in_features, max_degree=2, num_layers=4,
                 feature_dim=32, hidden_dim=64, num_parts=4,
                 partition_type="spectral", knn_k=10, use_bond_info=True,
                 bond_order_power=1.0, use_connectivity_features=False):
        super().__init__(
            radial_net=radial_net, in_features=in_features, max_degree=max_degree,
            num_layers=num_layers, feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_parts=num_parts, scalar_out_dim=1, task=3,
            partition_type=partition_type, knn_k=knn_k, use_bond_info=use_bond_info,
            bond_order_power=bond_order_power, use_connectivity_features=use_connectivity_features,
        )
        self.head = PolarizabilityHead(feature_dim, hidden_dim)

class SE3DeDipoleInterTransformer(_ForwardMixin, SE3InterNeighborhoodTransformer):
    def __init__(self, radial_net, in_features, max_degree=2, num_layers=4,
                 feature_dim=32, hidden_dim=64, num_parts=4,
                 partition_type="spectral", knn_k=10, use_bond_info=True,
                 bond_order_power=1.0, use_connectivity_features=False):
        super().__init__(
            radial_net=radial_net, in_features=in_features, max_degree=max_degree,
            num_layers=num_layers, feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_parts=num_parts, scalar_out_dim=1, task=3,
            partition_type=partition_type, knn_k=knn_k, use_bond_info=use_bond_info,
            bond_order_power=bond_order_power, use_connectivity_features=use_connectivity_features,
        )
        self.head = DeDipoleHead(feature_dim, hidden_dim)

class SE3DeDipoleIntraTransformer(_ForwardMixin, SE3IntraOnlyTransformer):
    def __init__(self, radial_net, in_features, max_degree=2, num_layers=4,
                 feature_dim=32, hidden_dim=64, num_parts=4,
                 partition_type="spectral", knn_k=10, use_bond_info=True,
                 bond_order_power=1.0, use_connectivity_features=False):
        super().__init__(
            radial_net=radial_net, in_features=in_features, max_degree=max_degree,
            num_layers=num_layers, feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_parts=num_parts, scalar_out_dim=1, task=3,
            partition_type=partition_type, knn_k=knn_k, use_bond_info=use_bond_info,
            bond_order_power=bond_order_power, use_connectivity_features=use_connectivity_features,
        )
        self.head = DeDipoleHead(feature_dim, hidden_dim)

def _filter_small_graphs(batch, num_parts, device, field="polar"):
    graph_batch = batch.batch.to(device)
    counts = torch.bincount(graph_batch)
    valid_graph = counts >= num_parts

    y = batch.y.to(device)
    target_full = y.view(-1, 3, 3) if field == "polar" else y

    if valid_graph.all():
        return (
            one_hot_z(batch.z).to(device),
            batch.pos.to(device),
            batch.edge_index.to(device),
            get_atomic_masses(batch.z).to(device),
            target_full,
            graph_batch,
        )

    keep_nodes = valid_graph[graph_batch]
    keep_graphs = valid_graph.nonzero(as_tuple=True)[0]
    if keep_graphs.numel() == 0:
        return None

    remap = torch.full((valid_graph.shape[0],), -1, dtype=torch.long, device=device)
    remap[keep_graphs] = torch.arange(keep_graphs.shape[0], device=device)

    node_feat = one_hot_z(batch.z).to(device)[keep_nodes]
    pos = batch.pos.to(device)[keep_nodes]
    atomic_mass = get_atomic_masses(batch.z).to(device)[keep_nodes]
    graph_batch = remap[graph_batch[keep_nodes]]
    target = target_full[keep_graphs] if field == "polar" else target_full[keep_nodes]

    src, dst = batch.edge_index.to(device)
    edge_mask = keep_nodes[src] & keep_nodes[dst]
    old_to_new = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[keep_nodes.nonzero(as_tuple=True)[0]] = torch.arange(keep_nodes.sum(), device=device)
    edge_index = old_to_new[batch.edge_index.to(device)[:, edge_mask]]

    return node_feat, pos, edge_index, atomic_mass, target, graph_batch


def train_epoch(model, loader, optimizer, num_parts, device, accum_steps, epoch,
                 monitor: SystemMonitor = None, field: str = "polar"):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    accum_loss = torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts, device, field=field)
        if tensors is None:
            continue
        node_feat, pos, edge_index, atomic_mass, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)
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
def evaluate(model, loader, num_parts, device, epoch=None, field: str = "polar"):
    model.eval()
    total_mse = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, num_parts, device, field=field)
        if tensors is None:
            continue
        node_feat, pos, edge_index, atomic_mass, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue

        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch)
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

def build_model(model_type: str, args, device):
    radial_net = RadialNetworkGSFB
    if args.field == "polar":
        cls = SE3PolarizabilityInterTransformer if model_type == "inter" else SE3PolarizabilityIntraTransformer
    else:
        cls = SE3DeDipoleInterTransformer if model_type == "inter" else SE3DeDipoleIntraTransformer
    return cls(
        radial_net=radial_net,
        in_features=len(ATOM_TYPES),
        max_degree=args.max_degree,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_parts=args.num_parts,
        partition_type=args.partition_type,
    ).to(device)

def run_ablation_trial(model_type: str, args, device, train_loader, val_loader, test_loader):
    print(f"\n=== Ablation run: model_type={model_type!r}, prediction_target={args.field!r} ===")
    torch.manual_seed(args.seed)

    model = build_model(model_type, args, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    monitor = SystemMonitor(device)

    checkpoint_path = os.path.join(args.metrics_dir, f"best_model_{model_type}.pt")
    best_val_mse = float("inf")

    for epoch in range(args.epochs):
        print(f"[{model_type}] Epoch {epoch + 1}")
        train_epoch(model, train_loader, optimizer, args.num_parts, device,
                    accum_steps=args.accum_steps, epoch=epoch, monitor=monitor, field=args.field)
        val_mse = evaluate(model, val_loader, args.num_parts, device, epoch, field=args.field)
        scheduler.step()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"  [New best] val MSE: {val_mse:.4f} -- saving checkpoint.")
            torch.save(model.state_dict(), checkpoint_path)

    monitor.save_csv(os.path.join(args.metrics_dir, f"metrics_{model_type}.csv"))
    monitor.save_plots(
        os.path.join(args.metrics_dir, f"metrics_{model_type}.png"),
        title_prefix=f"field={args.field}, model_type={model_type}",
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_mse = evaluate(model, test_loader, args.num_parts, device, field=args.field)
    print(f"[{model_type}] Test MSE: {test_mse:.4f}")
    return {"model_type": model_type, "best_val_mse": best_val_mse, "test_mse": test_mse}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="./data/QMe14S_single_point.h5")
    parser.add_argument("--field", type=str, default="polar", choices=["polar", "dedipole"])
    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--partition_type", type=str, default="spectral")
    parser.add_argument("--rbf_type", type=str, default="grbf")
    parser.add_argument("--num_layers", type=int, default=4)
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

    results = []
    for model_type in ("inter", "intra"):
        results.append(
            run_ablation_trial(model_type, args, device, train_loader, val_loader, test_loader)
        )

    print("\n=== Ablation summary ===")
    for r in results:
        print(f"  {r['model_type']:>6s} | best val MSE: {r['best_val_mse']:.4f} | test MSE: {r['test_mse']:.4f}")

if __name__ == "__main__":
    main()