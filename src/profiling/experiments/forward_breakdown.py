import time
import logging
from typing import Dict, Any, List
from contextlib import contextmanager

import torch
import torch.nn as nn

log = logging.getLogger("profiler.forward_breakdown")

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


@contextmanager
def _timer(store: dict, key: str):
    _cuda_sync()
    t0 = time.perf_counter()
    yield
    _cuda_sync()
    t1 = time.perf_counter()
    store.setdefault(key, []).append(t1 - t0)


def run_forward_breakdown_experiment(
    loader,
    args,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Any]:
    log.info(f"  Forward breakdown experiment: {num_batches} batches")

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
    from src.se3_crossformer.model import SE3IntraOnlyTransformer
    from src.se3_crossformer.se3_utils import RadialNetworkGRBF

    # The monkeypatching below reaches into SE3IntraOnlyLayer's own
    # _intra_update method directly -- the model has exactly one such layer.
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

    timing_store: Dict[str, List[float]] = {}

    def make_patched_forward(orig_forward):
        def patched_forward(f_in, x, neighbor_idx, neighbor_mask, neighbor_bond_attr=None):
            with _timer(timing_store, "layer_0.intra_update"):
                f_out = orig_forward.__self__._intra_update(
                    f_in, x, neighbor_idx, neighbor_mask,
                    neighbor_bond_attr=neighbor_bond_attr,
                )
            m_out: Dict[int, torch.Tensor] = {}
            return f_out, m_out
        return patched_forward

    original_layer_forward = model.layer.forward
    model.layer.forward = make_patched_forward(model.layer.forward)

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
        B           = int(graph_batch.max().item()) + 1

        ptr = [0]
        for g in range(B):
            ptr.append(int((graph_batch <= g).sum().item()))

        with _timer(timing_store, "graph_build"):
            for g in range(B):
                lo, hi = ptr[g], ptr[g + 1]
                _ = model._build_radius_neighbor_info(pos[lo:hi], hi - lo)

        with _timer(timing_store, "input_embedding"):
            f0 = model.input_embedding(node_feat).unsqueeze(-1)

        _cuda_sync()
        t_total0 = time.perf_counter()

        with torch.no_grad():
            _ = model(node_feat, pos, edge_index, am, graph_batch,
                      edge_attr=edge_attr, graph_idx=graph_idx)

        _cuda_sync()
        t_total1 = time.perf_counter()
        timing_store.setdefault("total_forward", []).append(t_total1 - t_total0)

        log.info(
            f"  batch {batch_idx+1:>2d}/{num_batches} | "
            f"total_fwd={timing_store['total_forward'][-1]*1e3:.1f}ms"
        )

    model.layer.forward = original_layer_forward

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    means = {k: _mean(v) for k, v in timing_store.items()}
    total_mean = means.get("total_forward", 1.0)

    block_pct: Dict[str, float] = {}
    for k, v in means.items():
        if k == "total_forward":
            continue
        block_pct[k] = 100.0 * v / max(total_mean, 1e-9)

    block_pct = dict(sorted(block_pct.items(), key=lambda x: x[1], reverse=True))

    log.info("  Block breakdown (% of total forward):")
    for block, pct in block_pct.items():
        log.info(f"    {block:<40s}: {pct:5.1f}%  ({means[block]*1e3:.2f}ms)")

    return {
        "timing_lists_s":   {k: v for k, v in timing_store.items()},
        "mean_times_s":     means,
        "block_pct":        block_pct,
        "mean_total_forward_s": total_mean,
    }
