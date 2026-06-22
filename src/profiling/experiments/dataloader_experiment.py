"""
dataloader_experiment.py
------------------------
Experiment 4: Dataloader throughput.

Measures:
  (a) Pure data loading time (CPU only, no device transfer)
  (b) Data loading + host→device transfer time

If (b) >> (a) the bottleneck is the H2D transfer (PCIe bandwidth).
If (a) is already slow, the bottleneck is disk I/O or CPU preprocessing.
"""

import time
import logging
from typing import Dict, Any

import torch

log = logging.getLogger("profiler.dataloader")


def run_dataloader_experiment(
    loader,
    device: torch.device,
    num_batches: int = 30,
) -> Dict[str, Any]:
    """
    Returns dict with keys:
        load_only_times_s, load_h2d_times_s,
        mean_load_only_s, mean_load_h2d_s,
        samples_per_sec_load_only, samples_per_sec_load_h2d
    """
    log.info(f"  Dataloader experiment: {num_batches} batches")

    # ── pass 1: load-only ──────────────────────────────────────────────────
    load_only_times: list = []
    n_samples_load:  int  = 0
    loader_iter = iter(loader)

    for i in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        t1 = time.perf_counter()
        load_only_times.append(t1 - t0)
        n_samples_load += int(batch.batch.max().item()) + 1
        log.debug(f"  load-only batch {i+1}: {(t1-t0)*1e3:.1f} ms")

    # ── pass 2: load + H2D ────────────────────────────────────────────────
    load_h2d_times: list = []
    n_samples_h2d:  int  = 0
    loader_iter = iter(loader)

    for i in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        # transfer everything in the batch
        _ = batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        load_h2d_times.append(t1 - t0)
        n_samples_h2d += int(batch.batch.max().item()) + 1
        log.debug(f"  load+h2d batch {i+1}: {(t1-t0)*1e3:.1f} ms")

    mean_load     = sum(load_only_times) / len(load_only_times)
    mean_load_h2d = sum(load_h2d_times)  / len(load_h2d_times)

    h2d_overhead  = mean_load_h2d - mean_load

    total_time_load     = sum(load_only_times)
    total_time_load_h2d = sum(load_h2d_times)

    sps_load     = n_samples_load / total_time_load     if total_time_load     > 0 else 0.0
    sps_load_h2d = n_samples_h2d  / total_time_load_h2d if total_time_load_h2d > 0 else 0.0

    log.info(f"  Mean load-only : {mean_load*1e3:.2f} ms  ({sps_load:.1f} samples/s)")
    log.info(f"  Mean load+H2D  : {mean_load_h2d*1e3:.2f} ms  ({sps_load_h2d:.1f} samples/s)")
    log.info(f"  H2D overhead   : {h2d_overhead*1e3:.2f} ms per batch")

    return {
        "load_only_times_s":       load_only_times,
        "load_h2d_times_s":        load_h2d_times,
        "mean_load_only_s":        mean_load,
        "mean_load_h2d_s":         mean_load_h2d,
        "h2d_overhead_s":          h2d_overhead,
        "samples_per_sec_load_only": sps_load,
        "samples_per_sec_load_h2d":  sps_load_h2d,
        "bottleneck_is_h2d":       (h2d_overhead / max(mean_load, 1e-9)) > 0.5,
    }