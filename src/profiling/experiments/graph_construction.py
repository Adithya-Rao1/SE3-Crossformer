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
    from src.se3_crossformer.spectral_partition import (
        spectral_partition,
        subgraph_center_of_mass,
        initial_message,
    )
    from src.se3_crossformer.model import SE3InterNeighborhoodTransformer

    model = SE3InterNeighborhoodTransformer(
        in_features = len(ATOM_TYPES),
        max_degree  = args.max_degree,
        num_layers  = args.num_layers,
        feature_dim = args.feature_dim,
        hidden_dim  = args.hidden_dim,
        num_parts   = args.num_parts,
        out_dim     = 19,
        task        = "regression",
    ).to(device)
    model.eval()

    times_spectral:  List[float] = []
    times_com:       List[float] = []
    times_neighbor:  List[float] = []
    times_total_build: List[float] = []
    times_forward:   List[float] = []

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
        am          = _atomic_masses(batch.x).to(device)
        graph_batch = batch.batch.to(device)
        N           = pos.shape[0]
        B           = int(graph_batch.max().item()) + 1

        build_start = time.perf_counter()

        ptr = [0]
        for g in range(B):
            ptr.append(int((graph_batch <= g).sum().item()))

        t_spectral_batch = 0.0
        t_com_batch      = 0.0
        t_neighbor_batch = 0.0

        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g    = hi - lo
            mask_e = (edge_index[0] >= lo) & (edge_index[0] < hi)
            ei_g   = edge_index[:, mask_e] - lo
            pos_g  = pos[lo:hi]
            am_g   = am[lo:hi]

            t0 = time.perf_counter()
            n2s = spectral_partition(ei_g, n_g, args.num_parts).to(device)
            _cuda_sync()
            t1 = time.perf_counter()
            t_spectral_batch += (t1 - t0)

            t0 = time.perf_counter()
            x_cm = subgraph_center_of_mass(pos_g, am_g, n2s, args.num_parts)
            _cuda_sync()
            t1 = time.perf_counter()
            t_com_batch += (t1 - t0)

            t0 = time.perf_counter()
            _ = model._build_neighbor_info(ei_g, n2s, n_g)
            _cuda_sync()
            t1 = time.perf_counter()
            t_neighbor_batch += (t1 - t0)

        build_end = time.perf_counter()

        times_spectral.append(t_spectral_batch)
        times_com.append(t_com_batch)
        times_neighbor.append(t_neighbor_batch)
        times_total_build.append(build_end - build_start)

        _cuda_sync()
        t_fwd0 = time.perf_counter()
        with torch.no_grad():
            _ = model(node_feat, pos, edge_index, am, graph_batch)
        _cuda_sync()
        t_fwd1 = time.perf_counter()
        times_forward.append(t_fwd1 - t_fwd0)

        log.info(
            f"  batch {batch_idx+1:>2d}/"
            f"{num_batches} | "
            f"spectral={t_spectral_batch*1e3:.1f}ms  "
            f"com={t_com_batch*1e3:.1f}ms  "
            f"neighbor={t_neighbor_batch*1e3:.1f}ms  "
            f"total_build={times_total_build[-1]*1e3:.1f}ms  "
            f"forward={times_forward[-1]*1e3:.1f}ms"
        )

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    mean_build   = _mean(times_total_build)
    mean_forward = _mean(times_forward)
    build_fwd_ratio = mean_build / max(mean_forward, 1e-9)

    result = {
        "spectral_partition_times_s":   times_spectral,
        "center_of_mass_times_s":       times_com,
        "neighbor_build_times_s":       times_neighbor,
        "total_graph_build_times_s":    times_total_build,
        "forward_times_s":              times_forward,
        "mean_spectral_s":              _mean(times_spectral),
        "mean_com_s":                   _mean(times_com),
        "mean_neighbor_s":              _mean(times_neighbor),
        "mean_total_graph_build_s":     mean_build,
        "mean_forward_s":               mean_forward,
        "graph_build_to_forward_ratio": build_fwd_ratio,
        "frequency_notes": {
            "spectral_partition":       "once per molecule per forward pass (not cached)",
            "center_of_mass":           "once per molecule per forward pass (not cached)",
            "neighbor_build":           "once per molecule per forward pass (Python loop)",
            "initial_message":          "once per layer per forward pass",
            "spherical_harmonics":      "once per edge per layer per forward pass",
            "basis_construction":       "once per edge per layer per forward pass",
        },
    }
    log.info(
        f"  Build/Forward ratio = {build_fwd_ratio:.2f}x  "
        f"({'graph build dominates' if build_fwd_ratio > 1 else 'forward dominates'})"
    )
    return result