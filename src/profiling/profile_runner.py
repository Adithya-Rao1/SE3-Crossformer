import argparse
import json
import os
import sys
import time
import subprocess
import threading
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.profiling.experiments.timing_experiment      import run_timing_experiment
from src.profiling.experiments.dataloader_experiment  import run_dataloader_experiment
from src.profiling.experiments.num_workers_experiment import run_num_workers_experiment
from src.profiling.experiments.batch_scaling          import run_batch_scaling_experiment
from src.profiling.experiments.graph_construction     import run_graph_construction_experiment
from src.profiling.experiments.forward_breakdown      import run_forward_breakdown_experiment
from src.profiling.experiments.torch_profiler_exp     import run_torch_profiler_experiment
from src.profiling.experiments.throughput_experiment  import run_throughput_experiment
from src.profiling.gpu_monitor                        import GpuMonitor
from src.profiling.plot_results                       import plot_all


RESULTS_DIR = ROOT / "profiling_results"
RESULTS_DIR.mkdir(exist_ok=True)

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path   = RESULTS_DIR / f"profile_{timestamp}.log"
json_path  = RESULTS_DIR / f"profile_{timestamp}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("profiler")


ALL_EXPERIMENTS = [
    "timing",
    "torch_profiler",
    "dataloader",
    "num_workers",
    "batch_scaling",
    "graph_construction",
    "forward_breakdown",
    "kernel_launch",
    "throughput",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target",          type=int,   default=0)
    p.add_argument("--num_parts",       type=int,   default=4)
    p.add_argument("--max_degree",      type=int,   default=2)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_layers",      type=int,   default=4)
    p.add_argument("--feature_dim",     type=int,   default=32)
    p.add_argument("--hidden_dim",      type=int,   default=64)
    p.add_argument("--data_root",       type=str,   default="./data")
    p.add_argument("--profile_batches", type=int,   default=20,
                   help="Number of batches used in per-batch timing experiments")
    p.add_argument("--experiments",     type=str,   default="all",
                   help="Comma-separated list of experiments, or 'all'")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(args, device):
    from src.se3_crossformer.model import SE3InterNeighborhoodTransformer
    ATOM_TYPES = [1, 6, 7, 8, 9]
    model = SE3InterNeighborhoodTransformer(
        in_features   = len(ATOM_TYPES),
        max_degree    = args.max_degree,
        num_layers    = args.num_layers,
        feature_dim   = args.feature_dim,
        hidden_dim    = args.hidden_dim,
        num_parts     = args.num_parts,
        out_dim       = 19,
        task          = "regression",
    ).to(device)
    return model


def build_loaders(args):
    """Thin wrapper – reuse load_qm9 from train.py."""
    sys.path.insert(0, str(ROOT))
    from train import load_qm9
    return load_qm9(
        target_idx = args.target,
        batch_size = args.batch_size,
        r          = args.data_root,
        device     = torch.device("cpu"),   
    )


def main():
    args   = parse_args()
    device = torch.device(args.device)

    selected = (
        ALL_EXPERIMENTS
        if args.experiments.strip().lower() == "all"
        else [e.strip() for e in args.experiments.split(",")]
    )

    log.info("=" * 70)
    log.info("SE3 Transformer Profiling Suite")
    log.info(f"  device          : {device}")
    log.info(f"  experiments     : {selected}")
    log.info(f"  profile_batches : {args.profile_batches}")
    log.info(f"  results dir     : {RESULTS_DIR}")
    log.info("=" * 70)

    results: dict = {"meta": vars(args), "timestamp": timestamp, "experiments": {}}

    gpu_monitor = None
    if device.type == "cuda":
        gpu_monitor = GpuMonitor(poll_interval=1.0)
        gpu_monitor.start()
        log.info("GPU monitor started (background thread).")

    train_loader, val_loader, _ = build_loaders(args)

    exp_map = {
        "timing":            (run_timing_experiment,
                              dict(loader=train_loader, args=args, device=device,
                                   num_batches=args.profile_batches)),
        "torch_profiler":    (run_torch_profiler_experiment,
                              dict(loader=train_loader, args=args, device=device,
                                   num_batches=min(5, args.profile_batches))),
        "dataloader":        (run_dataloader_experiment,
                              dict(loader=train_loader, device=device,
                                   num_batches=args.profile_batches)),
        "num_workers":       (run_num_workers_experiment,
                              dict(args=args, num_batches=args.profile_batches)),
        "batch_scaling":     (run_batch_scaling_experiment,
                              dict(args=args, device=device,
                                   num_batches=args.profile_batches)),
        "graph_construction":(run_graph_construction_experiment,
                              dict(loader=train_loader, args=args, device=device,
                                   num_batches=args.profile_batches)),
        "forward_breakdown": (run_forward_breakdown_experiment,
                              dict(loader=train_loader, args=args, device=device,
                                   num_batches=args.profile_batches)),
        "kernel_launch":     (run_torch_profiler_experiment,
                              dict(loader=train_loader, args=args, device=device,
                                   num_batches=min(5, args.profile_batches),
                                   kernel_mode=True)),
        "throughput":        (run_throughput_experiment,
                              dict(loader=train_loader, args=args, device=device,
                                   num_batches=args.profile_batches)),
    }

    for name in selected:
        if name not in exp_map:
            log.warning(f"Unknown experiment '{name}', skipping.")
            continue
        fn, kwargs = exp_map[name]
        log.info(f"\n{'─'*60}")
        log.info(f"Running experiment: {name}")
        log.info(f"{'─'*60}")
        try:
            t0   = time.perf_counter()
            data = fn(**kwargs)
            elapsed = time.perf_counter() - t0
            results["experiments"][name] = data
            log.info(f"  ✓ {name} completed in {elapsed:.1f}s")
            _log_summary(name, data)
        except Exception as exc:
            log.error(f"  ✗ {name} FAILED: {exc}", exc_info=True)
            results["experiments"][name] = {"error": str(exc)}

    if gpu_monitor is not None:
        gpu_monitor.stop()
        gpu_data = gpu_monitor.get_results()
        results["experiments"]["gpu_util"] = gpu_data
        log.info(f"\nGPU utilization samples collected: {len(gpu_data.get('util_pct', []))}")
        _log_summary("gpu_util", gpu_data)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serialise)
    log.info(f"\nResults saved → {json_path}")

    plot_dir = RESULTS_DIR / f"plots_{timestamp}"
    plot_dir.mkdir(exist_ok=True)
    try:
        plot_all(results["experiments"], plot_dir)
        log.info(f"Plots saved → {plot_dir}")
    except Exception as e:
        log.error(f"Plotting failed: {e}", exc_info=True)

    log.info(f"\nLog file → {log_path}")
    log.info("Profiling complete.")


def _json_serialise(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _log_summary(name: str, data: dict):
    """Print a compact summary for each experiment."""
    if not isinstance(data, dict) or "error" in data:
        return
    log.info(f"  Summary [{name}]:")
    SUMMARY_KEYS = {
        "timing":            ["mean_load_s", "mean_h2d_s", "mean_forward_s",
                              "mean_backward_s", "mean_optim_s"],
        "dataloader":        ["mean_load_only_s", "mean_load_h2d_s"],
        "num_workers":       ["best_num_workers", "best_samples_per_sec"],
        "batch_scaling":     ["best_batch_size", "best_samples_per_sec"],
        "graph_construction":["mean_total_graph_build_s", "mean_forward_s"],
        "forward_breakdown": ["block_pct"],
        "throughput":        ["nodes_per_sec", "edges_per_sec", "samples_per_sec"],
        "gpu_util":          ["mean_util_pct", "mean_mem_util_pct", "mean_power_w"],
    }
    keys = SUMMARY_KEYS.get(name, list(data.keys())[:6])
    for k in keys:
        if k in data:
            v = data[k]
            if isinstance(v, float):
                log.info(f"    {k}: {v:.4f}")
            elif isinstance(v, dict):
                log.info(f"    {k}: {json.dumps(v, default=str)}")
            else:
                log.info(f"    {k}: {v}")


if __name__ == "__main__":
    main()