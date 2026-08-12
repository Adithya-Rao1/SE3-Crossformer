import time
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn

log = logging.getLogger("profiler.graph_construction")

ATOM_TYPES = [1, 6, 7, 8, 9]


def _one_hot_z(z):
    oh = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for i, a in enumerate(ATOM_TYPES):
        oh[:, i] = (z == a).float()
    return oh


def _atomic_masses(z):
    M = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}
    return torch.tensor([M.get(zi.item(), 12.0) for zi in z], dtype=torch.float32)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_graph_construction_experiment(
    loader,
    args,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Any]:
    log.info(f"  Graph construction experiment: {num_batches} batches")

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
    from src.se3_crossformer.model import SE3IntraOnlyTransformer
    from src.se3_crossformer.se3_utils import RadialNetworkGRBF

    model = SE3IntraOnlyTransformer(
        radial_net  = RadialNetworkGRBF,
        in_features = len(ATOM_TYPES),
        max_degree  = args.max_degree,
        feature_dim = args.feature_dim,
        hidden_dim  = args.hidden_dim,
        radius_cutoff = args.radius_cutoff,
        scalar_out_dim = 1,
        task        = 0,
        bond_feature_dim  = getattr(args, "bond_feature_dim", 0),
    ).to(device)
    model.eval()

    times_radius_build: List[float] = []
    times_forward:      List[float] = []

    loader_iter = iter(loader)

    for batch_idx in range(num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch       = batch.to(device)
        node_feat   = _one_hot_z(batch.x).to(device)
        pos         = batch.pos.to(device)
        edge_index  = batch.edge_index.to(device)
        edge_attr   = batch.edge_attr.to(device)
        am          = _atomic_masses(batch.x).to(device)
        graph_batch = batch.batch.to(device)
        graph_idx   = batch.idx.to(device)
        N           = pos.shape[0]
        B           = int(graph_batch.max().item()) + 1

        ptr = [0]
        for g in range(B):
            ptr.append(int((graph_batch <= g).sum().item()))

        build_start = time.perf_counter()
        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g    = hi - lo
            pos_g  = pos[lo:hi]
            _ = model._build_radius_neighbor_info(pos_g, n_g)
        _cuda_sync()
        build_end = time.perf_counter()

        times_radius_build.append(build_end - build_start)

        _cuda_sync()
        t_fwd0 = time.perf_counter()
        with torch.no_grad():
            _ = model(node_feat, pos, edge_index, am, graph_batch,
                      edge_attr=edge_attr, graph_idx=graph_idx)
        _cuda_sync()
        t_fwd1 = time.perf_counter()
        times_forward.append(t_fwd1 - t_fwd0)

        log.info(
            f"  batch {batch_idx+1:>2d}/"
            f"{num_batches} | "
            f"radius_build={times_radius_build[-1]*1e3:.1f}ms  "
            f"forward={times_forward[-1]*1e3:.1f}ms"
        )

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    mean_build   = _mean(times_radius_build)
    mean_forward = _mean(times_forward)
    build_fwd_ratio = mean_build / max(mean_forward, 1e-9)

    result = {
        "total_graph_build_times_s":    times_radius_build,
        "forward_times_s":              times_forward,
        "mean_total_graph_build_s":     mean_build,
        "mean_forward_s":               mean_forward,
        "graph_build_to_forward_ratio": build_fwd_ratio,
        "frequency_notes": {
            "radius_build":             "once per molecule per forward pass (cdist + mask, not cached)",
            "spherical_harmonics":      "once per edge per layer per forward pass",
            "basis_construction":       "once per edge per layer per forward pass",
        },
    }
    log.info(
        f"  Build/Forward ratio = {build_fwd_ratio:.2f}x  "
        f"({'graph build dominates' if build_fwd_ratio > 1 else 'forward dominates'})"
    )
    return result
