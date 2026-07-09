"""
torch_profiler_exp.py
---------------------
Experiments 3 & 10: PyTorch profiler + kernel launch analysis.

Runs torch.profiler.profile on a small subset of training data and extracts:
  • Top-K slowest ops (by total CPU + CUDA time)
  • Scatter/gather ops
  • Attention kernel times
  • Python call overhead
  • Kernel launch count vs. useful-work time
  • GPU busy time vs idle time

The Chrome trace is saved to profiling_results/ for external inspection in
chrome://tracing.
"""

import time
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn

log = logging.getLogger("profiler.torch_profiler")

ATOM_TYPES = [1, 6, 7, 8, 9]


def _one_hot_z(z):
    oh = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for i, a in enumerate(ATOM_TYPES):
        oh[:, i] = (z == a).float()
    return oh


def _atomic_masses(z):
    M = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}
    return torch.tensor([M.get(zi.item(), 12.0) for zi in z], dtype=torch.float32)


def run_torch_profiler_experiment(
    loader,
    args,
    device: torch.device,
    num_batches: int = 3,
    kernel_mode: bool = False,   # if True, focus on kernel launch analysis
) -> Dict[str, Any]:
    """
    Returns dict with profiler summary data.
    """
    mode_label = "kernel_launch" if kernel_mode else "torch_profiler"
    log.info(f"  PyTorch profiler [{mode_label}]: {num_batches} batches on {device}")

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))

    RESULTS_DIR = ROOT / "profiling_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    trace_path = str(RESULTS_DIR / f"{mode_label}_trace.json")

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
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    use_cuda = device.type == "cuda"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    loader_iter = iter(loader)
    batches = []
    for _ in range(num_batches):
        try:
            b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            b = next(loader_iter)
        batches.append(b)

    def _step(batch):
        batch       = batch.to(device)
        node_feat   = _one_hot_z(batch.x).to(device)
        pos         = batch.pos.to(device)
        edge_index  = batch.edge_index.to(device)
        am          = _atomic_masses(batch.x).to(device)
        target      = batch.y.to(device)
        graph_batch = batch.batch.to(device)

        pred = model(node_feat, pos, edge_index, am, graph_batch)
        loss = nn.functional.l1_loss(pred.squeeze(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    schedule = torch.profiler.schedule(
        wait   = 0,
        warmup = 1,
        active = 1,
        repeat = 1,
    )

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(RESULTS_DIR / mode_label)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    ) as prof:
        for batch in batches:
            _step(batch)
            prof.step()

    # Also export Chrome trace
    try:
        prof.export_chrome_trace(trace_path)
        log.info(f"  Chrome trace saved → {trace_path}")
    except Exception as e:
        log.warning(f"  Chrome trace export failed: {e}")

    # ── extract key metrics ───────────────────────────────────────────────
    TOP_K = 20

    try:
        key_averages = prof.key_averages(group_by_input_shape=False)
        sorted_ops   = sorted(
            key_averages,
            key=lambda e: e.self_cuda_time_total if use_cuda else e.self_cpu_time_total,
            reverse=True,
        )

        top_k_ops: List[Dict] = []
        for entry in sorted_ops[:TOP_K]:
            top_k_ops.append({
                "name":           entry.key,
                "cpu_time_us":    entry.self_cpu_time_total,
                "cuda_time_us":   entry.self_cuda_time_total if use_cuda else 0,
                "count":          entry.count,
                "cpu_memory_b":   entry.cpu_memory_usage,
                "cuda_memory_b":  entry.cuda_memory_usage if use_cuda else 0,
                "flops":          getattr(entry, "flops", 0),
            })

        # Scatter / gather ops
        scatter_gather_ops = [
            e for e in key_averages
            if any(kw in e.key.lower() for kw in ["scatter", "gather", "index"])
        ]
        scatter_time_us = sum(
            e.self_cuda_time_total if use_cuda else e.self_cpu_time_total
            for e in scatter_gather_ops
        )

        # Attention-related kernels
        attn_ops = [
            e for e in key_averages
            if any(kw in e.key.lower() for kw in ["softmax", "bmm", "einsum", "matmul"])
        ]
        attn_time_us = sum(
            e.self_cuda_time_total if use_cuda else e.self_cpu_time_total
            for e in attn_ops
        )

        # Python/prim overhead
        python_ops = [
            e for e in key_averages
            if "python" in e.key.lower() or "call_function" in e.key.lower()
        ]
        python_time_us = sum(e.self_cpu_time_total for e in python_ops)

        # Kernel launch vs useful work (CUDA only)
        kernel_launch_count = 0
        total_cuda_us = 0
        if use_cuda:
            for e in key_averages:
                kernel_launch_count += e.count
                total_cuda_us       += e.self_cuda_time_total
            # Estimate GPU busy time from CUDA activities
            total_cpu_us_with_cuda = sum(e.self_cpu_time_total for e in key_averages)
            gpu_busy_pct = (total_cuda_us / max(total_cpu_us_with_cuda, 1)) * 100
        else:
            gpu_busy_pct = 0.0

        # Total FLOPS
        total_flops = sum(getattr(e, "flops", 0) or 0 for e in key_averages)

        result = {
            "mode":                     mode_label,
            "trace_path":               trace_path,
            "top_k_ops":                top_k_ops,
            "scatter_gather_time_us":   scatter_time_us,
            "attention_time_us":        attn_time_us,
            "python_overhead_time_us":  python_time_us,
            "kernel_launch_count":      kernel_launch_count,
            "total_cuda_time_us":       total_cuda_us,
            "gpu_busy_pct":             gpu_busy_pct,
            "gpu_idle_pct":             100.0 - gpu_busy_pct,
            "total_flops":              total_flops,
        }

        log.info(f"  Top-3 ops by CUDA time:")
        for op in top_k_ops[:3]:
            log.info(
                f"    {op['name']:<50s}  "
                f"cuda={op['cuda_time_us']/1e3:.2f}ms  "
                f"count={op['count']}"
            )
        log.info(f"  Scatter/Gather total: {scatter_time_us/1e3:.2f} ms")
        log.info(f"  Attention total:      {attn_time_us/1e3:.2f} ms")
        log.info(f"  Python overhead:      {python_time_us/1e3:.2f} ms")
        if use_cuda:
            log.info(f"  GPU busy: {gpu_busy_pct:.1f}%  idle: {100-gpu_busy_pct:.1f}%")
            log.info(f"  Kernel launches: {kernel_launch_count}")

        return result

    except Exception as e:
        log.error(f"  Profiler key_averages extraction failed: {e}")
        return {"mode": mode_label, "error": str(e), "trace_path": trace_path}