import time
import logging
from typing import Dict, Any, List

import torch

log = logging.getLogger("profiler.num_workers")

WORKER_COUNTS = [0, 2, 4, 8, 16]


def run_num_workers_experiment(
    args,
    num_batches: int = 20,
) -> Dict[str, Any]:
    log.info("  num_workers sweep: " + str(WORKER_COUNTS))

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
    from src.load_data import CustomQM9Dataset
    from torch_geometric.loader import DataLoader

    dataset = CustomQM9Dataset(
        root     = args.data_root + "/qm1",
        sdf_file = args.data_root + "/qm9/raw/gdb9.sdf",
        csv_file = args.data_root + "/qm9/raw/gdb9.sdf.csv",
        device   = torch.device("cpu"),
    )
    dataset.y = dataset.y[:, args.target]
    # Use only a slice for speed
    subset_size = min(len(dataset), num_batches * args.batch_size * 4)
    dataset = dataset[:subset_size]

    results_sps:  List[float] = []
    results_tpb:  List[float] = []

    for nw in WORKER_COUNTS:
        loader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = nw,
            pin_memory  = False,
        )
        times: List[float] = []
        n_samples = 0
        loader_iter = iter(loader)

        for i in range(num_batches):
            t0 = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_samples += int(batch.batch.max().item()) + 1

        total_t = sum(times)
        sps     = n_samples / total_t if total_t > 0 else 0.0
        tpb     = total_t / len(times)
        results_sps.append(sps)
        results_tpb.append(tpb)
        log.info(f"  num_workers={nw:>2d}: {sps:.1f} samples/s  ({tpb*1e3:.1f} ms/batch)")

    best_idx = int(max(range(len(results_sps)), key=lambda i: results_sps[i]))
    return {
        "worker_counts":         WORKER_COUNTS,
        "samples_per_sec":       results_sps,
        "mean_time_per_batch_s": results_tpb,
        "best_num_workers":      WORKER_COUNTS[best_idx],
        "best_samples_per_sec":  results_sps[best_idx],
        "dataloader_limited":    (results_sps[-1] / max(results_sps[0], 1e-9)) > 1.5,
    }