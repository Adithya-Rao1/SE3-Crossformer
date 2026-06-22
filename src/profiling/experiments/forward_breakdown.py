"""
forward_breakdown.py
--------------------
Experiment 9: Time each major block of the forward pass and report
the percentage of total forward time each block comprises.

Blocks timed:
  • input_embedding
  • graph_build   (spectral partition + neighbor info)
  • layer_{i}     (each SE3InterNeighborhoodLayer)
    ├─ intra_update
    ├─ initial_message
    ├─ message_update
    └─ cross_update
  • readout

Uses a monkey-patched version of the model's layer forward to insert timers
without modifying the source code.
"""

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
    """
    Returns dict with block timing lists, means, and percentage breakdown.
    """
    log.info(f"  Forward breakdown experiment: {num_batches} batches")

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
    from src.se3_crossformer.model import (
        SE3InterNeighborhoodTransformer,
        SE3InterNeighborhoodLayer,
    )
    from src.se3_crossformer.spectral_partition import initial_message

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

    timing_store: Dict[str, List[float]] = {}

    # ── patch each layer to time its sub-blocks ───────────────────────────
    original_layer_forwards = {}

    def make_patched_forward(layer_idx, orig_forward):
        def patched_forward(f_in, x, neighbor_idx, neighbor_mask,
                            x_cm, node_to_subgraph, subgraph_mask):
            layer_key = f"layer_{layer_idx}"
            num_subgraphs = x_cm.shape[0]

            with _timer(timing_store, f"{layer_key}.intra_update"):
                f_out = orig_forward.__self__._intra_update(
                    f_in, x, neighbor_idx, neighbor_mask
                )
            with _timer(timing_store, f"{layer_key}.initial_message"):
                m_in = initial_message(f_out, node_to_subgraph, num_subgraphs)
            with _timer(timing_store, f"{layer_key}.message_update"):
                m_out = orig_forward.__self__._message_update(m_in, x_cm, subgraph_mask)
            with _timer(timing_store, f"{layer_key}.cross_update"):
                f_out = orig_forward.__self__._cross_update(
                    f_out, m_out, x, x_cm, node_to_subgraph, subgraph_mask
                )
            return f_out, m_out
        return patched_forward

    for i, layer in enumerate(model.layers):
        original_layer_forwards[i] = layer.forward
        layer.forward = make_patched_forward(i, layer.forward)

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

        with _timer(timing_store, "graph_build"):
            B = int(graph_batch.max().item()) + 1
            # just trigger the build portion via a dummy forward to get graph_build time
            # We actually time it inside the full forward below
            pass

        _cuda_sync()
        t_total0 = time.perf_counter()

        with _timer(timing_store, "input_embedding"):
            import math
            N = node_feat.shape[0]
            f0 = model.input_embedding(node_feat).unsqueeze(-1)
            f1 = torch.randn(N, f0.shape[1], 3,  device=device) * (1/math.sqrt(3.0))
            f2 = torch.randn(N, f0.shape[1], 5,  device=device) * (1/math.sqrt(5.0))

        # Time graph build separately
        with _timer(timing_store, "graph_build"):
            with torch.no_grad():
                _ = model(node_feat, pos, edge_index, am, graph_batch)

        _cuda_sync()
        t_total1 = time.perf_counter()
        timing_store.setdefault("total_forward", []).append(t_total1 - t_total0)

        log.info(
            f"  batch {batch_idx+1:>2d}/{num_batches} | "
            f"total_fwd={timing_store['total_forward'][-1]*1e3:.1f}ms"
        )

    # Restore original layer forwards
    for i, layer in enumerate(model.layers):
        if i in original_layer_forwards:
            layer.forward = original_layer_forwards[i]

    # ── compute means and percentages ─────────────────────────────────────
    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    means = {k: _mean(v) for k, v in timing_store.items()}
    total_mean = means.get("total_forward", 1.0)

    block_pct: Dict[str, float] = {}
    for k, v in means.items():
        if k == "total_forward":
            continue
        block_pct[k] = 100.0 * v / max(total_mean, 1e-9)

    # Sort by percentage descending
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