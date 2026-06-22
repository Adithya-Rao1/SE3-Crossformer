"""
throughput_experiment.py
------------------------
Experiments 11 & 12: Track nodes/sec, edges/sec, samples/sec, and
spherical harmonic accesses/sec.

Also records per-batch graph statistics (nodes, edges, avg degree) to help
correlate throughput with graph size.
"""

import time
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn

log = logging.getLogger("profiler.throughput")

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


def _count_sh_accesses(num_edges: int, max_degree: int, num_layers: int) -> int:
    """
    Estimate: for each edge, for each (l, k) pair, for each J in |l-k|..l+k,
    we compute Y_J.  Per layer, per edge.
    """
    sh_per_edge = 0
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            for J in range(abs(l - k), l + k + 1):
                sh_per_edge += 1   # one Y_J call
    return sh_per_edge * num_edges * num_layers


def run_throughput_experiment(
    loader,
    args,
    device: torch.device,
    num_batches: int = 20,
) -> Dict[str, Any]:
    """
    Returns dict with throughput metrics.
    """
    log.info(f"  Throughput experiment: {num_batches} batches on {device}")

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
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

    # Per-batch stats
    batch_nodes:   List[int]   = []
    batch_edges:   List[int]   = []
    batch_samples: List[int]   = []
    batch_times:   List[float] = []
    batch_sh_est:  List[int]   = []

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

        N      = pos.shape[0]
        E      = edge_index.shape[1]
        B      = int(graph_batch.max().item()) + 1
        sh_est = _count_sh_accesses(E, args.max_degree, args.num_layers)

        _cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(node_feat, pos, edge_index, am, graph_batch)
        _cuda_sync()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        batch_nodes.append(N)
        batch_edges.append(E)
        batch_samples.append(B)
        batch_times.append(elapsed)
        batch_sh_est.append(sh_est)

        nps = N / elapsed
        eps = E / elapsed
        sps = B / elapsed
        sh_s = sh_est / elapsed

        log.info(
            f"  batch {batch_idx+1:>2d}/{num_batches} | "
            f"N={N:>5d}  E={E:>6d}  B={B:>3d}  "
            f"t={elapsed*1e3:.1f}ms  "
            f"N/s={nps:.0f}  E/s={eps:.0f}  samp/s={sps:.1f}  SH/s={sh_s:.0f}"
        )

    total_nodes   = sum(batch_nodes)
    total_edges   = sum(batch_edges)
    total_samples = sum(batch_samples)
    total_time    = sum(batch_times)
    total_sh      = sum(batch_sh_est)

    nodes_per_sec   = total_nodes   / total_time
    edges_per_sec   = total_edges   / total_time
    samples_per_sec = total_samples / total_time
    sh_per_sec      = total_sh      / total_time

    avg_nodes_per_graph = total_nodes   / max(total_samples, 1)
    avg_edges_per_graph = total_edges   / max(total_samples, 1)
    avg_degree          = total_edges   / max(total_nodes,   1)

    log.info(f"\n  ─── Throughput summary ───────────────────────────────")
    log.info(f"  nodes/s  : {nodes_per_sec:.0f}")
    log.info(f"  edges/s  : {edges_per_sec:.0f}")
    log.info(f"  samples/s: {samples_per_sec:.2f}")
    log.info(f"  SH/s     : {sh_per_sec:.0f}")
    log.info(f"  Avg nodes/graph: {avg_nodes_per_graph:.1f}")
    log.info(f"  Avg edges/graph: {avg_edges_per_graph:.1f}")
    log.info(f"  Avg degree     : {avg_degree:.2f}")

    return {
        "batch_nodes":              batch_nodes,
        "batch_edges":              batch_edges,
        "batch_samples":            batch_samples,
        "batch_times_s":            batch_times,
        "batch_sh_estimates":       batch_sh_est,
        # aggregate throughput
        "nodes_per_sec":            nodes_per_sec,
        "edges_per_sec":            edges_per_sec,
        "samples_per_sec":          samples_per_sec,
        "sh_accesses_per_sec":      sh_per_sec,
        # graph size stats
        "avg_nodes_per_graph":      avg_nodes_per_graph,
        "avg_edges_per_graph":      avg_edges_per_graph,
        "avg_degree":               avg_degree,
        "total_nodes_processed":    total_nodes,
        "total_edges_processed":    total_edges,
        "total_samples_processed":  total_samples,
        "total_sh_accesses_est":    total_sh,
        "total_wall_time_s":        total_time,
    }