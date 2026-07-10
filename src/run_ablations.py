"""
run_ablations.py
-----------------
Runs the three ablation sweeps (graph construction, model architecture, RBF)
sequentially against train.py, writing each run's metrics to its own
directory and logging a summary CSV across all runs.

Fixed-variable design per ablation group:
    graph_construction : model_type="inter",  rbf_type="gsfb"  | sweep partition_type
    model_architecture  : partition_type="spectral", rbf_type="gsfb" | sweep model_type
    rbf                  : model_type="inter",  partition_type="spectral" | sweep rbf_type

Usage:
    python run_ablations.py --epochs 100 --target 1 --batch_size 32 --accum_steps 8
    python run_ablations.py --groups graph_construction rbf   # run subset of groups
    python run_ablations.py --dry_run                          # print commands only
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

LAYERS = ["inter", "intra_only"]                # model architecture sweep
PARTITION_TYPES = ["spectral", "knn"]            # graph construction sweep
RBFS = ["grbf", "gsfb"]                   # RBF sweep


@dataclass
class AblationRun:
    group: str            # "graph_construction" | "model_architecture" | "rbf"
    swept_param: str      # name of the parameter being varied in this run
    partition_type: str
    model_type: str
    rbf_type: str

    @property
    def run_name(self) -> str:
        return (
            f"{self.group}"
            f"__pt-{self.partition_type}"
            f"__mt-{self.model_type}"
            f"__rbf-{self.rbf_type}"
        )


def build_ablation_plan() -> list:
    """Builds the full sequential list of ablation runs, in the fixed-vs-swept
    configuration requested:
      1) graph_construction: model_type=inter, rbf_type=gsfb, sweep partition_type
      2) model_architecture:  partition_type=spectral, rbf_type=gsfb, sweep model_type
      3) rbf:                  model_type=inter, partition_type=spectral, sweep rbf_type
    """
    plan = []

    for pt in PARTITION_TYPES:
        plan.append(AblationRun(
            group="graph_construction",
            swept_param=f"partition_type={pt}",
            partition_type=pt,
            model_type="inter",
            rbf_type="gsfb",
        ))

    for mt in LAYERS:
        plan.append(AblationRun(
            group="model_architecture",
            swept_param=f"model_type={mt}",
            partition_type="spectral",
            model_type=mt,
            rbf_type="gsfb",
        ))

    for rbf in RBFS:
        plan.append(AblationRun(
            group="rbf",
            swept_param=f"rbf_type={rbf}",
            partition_type="spectral",
            model_type="inter",
            rbf_type=rbf,
        ))

    return plan


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

TEST_MAE_RE = re.compile(
    r"Test MAE over \d+ trials?:\s*([-\d.eE]+)\s*±\s*([-\d.eE]+)"
)


def build_command(run: AblationRun, args: argparse.Namespace, metrics_dir: str) -> list:
    cmd = [
        sys.executable, args.train_script,
        "--target", str(args.target),
        "--num_parts", str(args.num_parts),
        "--max_degree", str(args.max_degree),
        "--batch_size", str(args.batch_size),
        "--accum_steps", str(args.accum_steps),
        "--partition_type", run.partition_type,
        "--model_type", run.model_type,
        "--rbf_type", run.rbf_type,
        "--num_layers", str(args.num_layers),
        "--feature_dim", str(args.feature_dim),
        "--hidden_dim", str(args.hidden_dim),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--trials", str(args.trials),
        "--data_root", args.data_root,
        "--metrics_dir", metrics_dir,
        "--device", args.device,
    ]
    return cmd


def parse_test_mae(log_text: str) -> Optional[tuple]:
    match = TEST_MAE_RE.search(log_text)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def run_single_ablation(run: AblationRun, args: argparse.Namespace) -> dict:
    metrics_dir = os.path.join(args.base_metrics_dir, run.group, run.run_name)
    os.makedirs(metrics_dir, exist_ok=True)

    cmd = build_command(run, args, metrics_dir)
    log_path = os.path.join(metrics_dir, "run.log")

    print(f"\n{'=' * 80}\n[{run.group}] {run.swept_param}\n"
          f"  partition_type={run.partition_type}  model_type={run.model_type}  "
          f"rbf_type={run.rbf_type}\n  metrics_dir={metrics_dir}\n"
          f"  cmd: {' '.join(cmd)}\n{'=' * 80}")

    result_row = {
        "group": run.group,
        "swept_param": run.swept_param,
        "partition_type": run.partition_type,
        "model_type": run.model_type,
        "rbf_type": run.rbf_type,
        "metrics_dir": metrics_dir,
        "returncode": None,
        "test_mae": None,
        "test_mae_ci95": None,
        "wall_time_sec": None,
        "status": "not_run",
    }

    if args.dry_run:
        result_row["status"] = "dry_run"
        return result_row

    start = time.time()
    with open(log_path, "w") as log_file:
        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log_file.write(process.stdout)
    elapsed = time.time() - start

    result_row["returncode"] = process.returncode
    result_row["wall_time_sec"] = round(elapsed, 1)

    if process.returncode != 0:
        result_row["status"] = "failed"
        print(f"  [FAILED] returncode={process.returncode}. See {log_path}")
        if args.stop_on_failure:
            raise RuntimeError(f"Ablation run {run.run_name} failed; see {log_path}")
    else:
        parsed = parse_test_mae(process.stdout)
        if parsed is not None:
            result_row["test_mae"], result_row["test_mae_ci95"] = parsed
        result_row["status"] = "success"
        print(f"  [OK] test_mae={result_row['test_mae']} "
              f"(±{result_row['test_mae_ci95']})  [{elapsed:.1f}s]")

    return result_row


def write_summary_csv(rows: list, path: str):
    fieldnames = [
        "group", "swept_param", "partition_type", "model_type", "rbf_type",
        "metrics_dir", "returncode", "test_mae", "test_mae_ci95",
        "wall_time_sec", "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSummary written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run SE3 transformer ablation sweeps.")

    parser.add_argument("--train_script", type=str, default="train.py",
                         help="Path to train.py entry point.")
    parser.add_argument("--groups", type=str, nargs="+",
                         default=["graph_construction", "model_architecture", "rbf"],
                         choices=["graph_construction", "model_architecture", "rbf"],
                         help="Which ablation groups to run, in order.")

    # Fixed training hyperparameters passed through to every run.
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--data_root", type=str,
                         default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--device", type=str,
                         default="cuda" if _cuda_available() else "cpu")

    parser.add_argument("--base_metrics_dir", type=str, default="./ablation_metrics",
                         help="Root directory; each run gets its own subdirectory "
                              "named after its group and swept parameter.")
    parser.add_argument("--stop_on_failure", action="store_true",
                         help="Abort the whole sweep on the first failed run "
                              "instead of continuing to the next one.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print planned commands without executing them.")

    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    args = parse_args()
    os.makedirs(args.base_metrics_dir, exist_ok=True)

    plan = [run for run in build_ablation_plan() if run.group in args.groups]

    print(f"Planned {len(plan)} ablation run(s) across group(s): {args.groups}")

    results = []
    for run in plan:
        row = run_single_ablation(run, args)
        results.append(row)
        write_summary_csv(results, os.path.join(args.base_metrics_dir, "summary.csv"))

    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone. {n_ok} succeeded, {n_failed} failed, "
          f"{len(results) - n_ok - n_failed} not run.")


if __name__ == "__main__":
    main()