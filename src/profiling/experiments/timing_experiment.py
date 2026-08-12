import time
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam

log = logging.getLogger("profiler.timing")

ATOM_TYPES = [1, 6, 7, 8, 9]


def _one_hot_z(z: torch.Tensor) -> torch.Tensor:
    one_hot = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for idx, a in enumerate(ATOM_TYPES):
        one_hot[:, idx] = (z == a).float()
    return one_hot


def _get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}
    return torch.tensor(
        [MASSES.get(zi.item(), 12.0) for zi in z], dtype=torch.float32
    )


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_timing_experiment(
    loader,
    args,
    device: torch.device,
    num_batches: int = 20,
) -> Dict[str, Any]:
    log.info(f"  Timing experiment: {num_batches} batches on {device}")

    from src.se3_crossformer.model import SE3IntraOnlyTransformer
    from src.se3_crossformer.se3_utils import RadialNetworkGRBF

    model = SE3IntraOnlyTransformer(
        radial_net=RadialNetworkGRBF,
        in_features=len(ATOM_TYPES),
        max_degree=args.max_degree,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        radius_cutoff=args.radius_cutoff,
        scalar_out_dim=1,
        task=0,
        bond_feature_dim=getattr(args, "bond_feature_dim", 0),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train()

    load_times:     list = []
    h2d_times:      list = []
    forward_times:  list = []
    backward_times: list = []
    optim_times:    list = []

    loader_iter = iter(loader)

    for batch_idx in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        t1 = time.perf_counter()
        load_times.append(t1 - t0)

        t2 = time.perf_counter()
        batch        = batch.to(device)
        node_feat    = _one_hot_z(batch.x).to(device)
        pos          = batch.pos.to(device)
        edge_index   = batch.edge_index.to(device)
        edge_attr    = batch.edge_attr.to(device)
        atomic_mass  = _get_atomic_masses(batch.x).to(device)
        target       = batch.y[:, args.target].to(device)
        graph_batch  = batch.batch.to(device)
        graph_idx    = batch.idx.to(device)
        _cuda_sync()
        t3 = time.perf_counter()
        h2d_times.append(t3 - t2)

        t4 = time.perf_counter()
        pred = model(node_feat, pos, edge_index, atomic_mass, graph_batch,
                     edge_attr=edge_attr, graph_idx=graph_idx)
        loss = nn.functional.l1_loss(pred.squeeze(-1), target)
        _cuda_sync()
        t5 = time.perf_counter()
        forward_times.append(t5 - t4)

        optimizer.zero_grad()
        t6 = time.perf_counter()
        loss.backward()
        _cuda_sync()
        t7 = time.perf_counter()
        backward_times.append(t7 - t6)

        t8 = time.perf_counter()
        optimizer.step()
        _cuda_sync()
        t9 = time.perf_counter()
        optim_times.append(t9 - t8)

        total = (t9 - t0)
        log.info(
            f"  batch {batch_idx+1:>3d}/{num_batches} | "
            f"load={load_times[-1]*1e3:.1f}ms  h2d={h2d_times[-1]*1e3:.1f}ms  "
            f"fwd={forward_times[-1]*1e3:.1f}ms  bwd={backward_times[-1]*1e3:.1f}ms  "
            f"opt={optim_times[-1]*1e3:.1f}ms  total={total*1e3:.1f}ms"
        )

    def _stats(lst):
        import statistics
        return {
            "mean": statistics.mean(lst),
            "std":  statistics.stdev(lst) if len(lst) > 1 else 0.0,
            "min":  min(lst),
            "max":  max(lst),
        }

    result = {
        "load_times_s":     load_times,
        "h2d_times_s":      h2d_times,
        "forward_times_s":  forward_times,
        "backward_times_s": backward_times,
        "optim_times_s":    optim_times,
        "mean_load_s":      sum(load_times)     / len(load_times),
        "mean_h2d_s":       sum(h2d_times)      / len(h2d_times),
        "mean_forward_s":   sum(forward_times)  / len(forward_times),
        "mean_backward_s":  sum(backward_times) / len(backward_times),
        "mean_optim_s":     sum(optim_times)    / len(optim_times),
        "stats_load":       _stats(load_times),
        "stats_h2d":        _stats(h2d_times),
        "stats_forward":    _stats(forward_times),
        "stats_backward":   _stats(backward_times),
        "stats_optim":      _stats(optim_times),
    }

    phases = {
        "load":     result["mean_load_s"],
        "h2d":      result["mean_h2d_s"],
        "forward":  result["mean_forward_s"],
        "backward": result["mean_backward_s"],
        "optim":    result["mean_optim_s"],
    }
    dominant = max(phases, key=phases.get)
    result["dominant_phase"] = dominant
    result["phase_means_s"]  = phases

    log.info(f"  Dominant phase: {dominant} ({phases[dominant]*1e3:.1f} ms mean)")
    return result