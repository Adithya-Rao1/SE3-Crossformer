import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger("profiler.plots")

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
HAS_MPL = True

COLOURS = {
    "load":     "#4e79a7",
    "h2d":      "#f28e2b",
    "forward":  "#e15759",
    "backward": "#76b7b2",
    "optim":    "#59a14f",
    "default":  "#4e79a7",
}

def _savefig(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out.name}")


def plot_all(experiments: Dict[str, Any], plot_dir: Path):
    if not HAS_MPL:
        log.warning("Skipping all plots (matplotlib unavailable).")
        return

    dispatch = {
        "timing":             _plot_timing,
        "dataloader":         _plot_dataloader,
        "num_workers":        _plot_num_workers,
        "batch_scaling":      _plot_batch_scaling,
        "graph_construction": _plot_graph_construction,
        "forward_breakdown":  _plot_forward_breakdown,
        "torch_profiler":     _plot_torch_profiler,
        "kernel_launch":      _plot_torch_profiler,  
        "throughput":         _plot_throughput,
        "gpu_util":           _plot_gpu_util,
    }

    for name, data in experiments.items():
        if "error" in data:
            log.warning(f"  Skipping plot for {name}: error in experiment.")
            continue
        fn = dispatch.get(name)
        if fn is None:
            continue
        try:
            fn(data, plot_dir, name)
        except Exception as e:
            log.error(f"  Plot for {name} failed: {e}", exc_info=True)


def _plot_timing(data: Dict, plot_dir: Path, name: str):
    """Experiment 1: stacked per-batch timing."""
    batches = list(range(1, len(data["load_times_s"]) + 1))

    load = np.array(data["load_times_s"]) * 1e3
    h2d  = np.array(data["h2d_times_s"])  * 1e3
    fwd  = np.array(data["forward_times_s"])  * 1e3
    bwd  = np.array(data["backward_times_s"]) * 1e3
    opt  = np.array(data["optim_times_s"])    * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.stackplot(
        batches,
        load, h2d, fwd, bwd, opt,
        labels=["Load", "H2D", "Forward", "Backward", "Optim"],
        colors=[COLOURS["load"], COLOURS["h2d"], COLOURS["forward"],
                COLOURS["backward"], COLOURS["optim"]],
        alpha=0.85,
    )
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-batch timing breakdown (stacked)")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.set_xlim(1, max(batches))

    ax2 = axes[1]
    phases = ["load", "h2d", "forward", "backward", "optim"]
    means  = [
        data["mean_load_s"]     * 1e3,
        data["mean_h2d_s"]      * 1e3,
        data["mean_forward_s"]  * 1e3,
        data["mean_backward_s"] * 1e3,
        data["mean_optim_s"]    * 1e3,
    ]
    bars = ax2.bar(
        phases, means,
        color=[COLOURS[p] for p in phases],
        edgecolor="white", width=0.6,
    )
    for bar, val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}ms",
            ha="center", va="bottom", fontsize=9,
        )
    ax2.set_ylabel("Mean time (ms)")
    ax2.set_title(f"Mean phase times  [dominant: {data.get('dominant_phase','?')}]")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.tight_layout(pad=2)
    _savefig(fig, plot_dir, "01_timing_breakdown.png")


def _plot_dataloader(data: Dict, plot_dir: Path, name: str):
    """Experiment 4: load-only vs load+H2D per batch."""
    n = min(len(data["load_only_times_s"]), len(data["load_h2d_times_s"]))
    x = list(range(1, n + 1))
    lo = np.array(data["load_only_times_s"][:n]) * 1e3
    lh = np.array(data["load_h2d_times_s"][:n])  * 1e3

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, lo, label="Load only",    color=COLOURS["load"],  linewidth=1.5)
    ax.plot(x, lh, label="Load + H2D",  color=COLOURS["h2d"],   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("DataLoader: load-only vs load + host→device transfer")
    ax.legend()
    _savefig(fig, plot_dir, "04_dataloader.png")


def _plot_num_workers(data: Dict, plot_dir: Path, name: str):
    """Experiment 5: samples/sec vs num_workers."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        data["worker_counts"], data["samples_per_sec"],
        marker="o", color=COLOURS["forward"], linewidth=2,
    )
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Samples / second")
    ax.set_title("DataLoader throughput vs. num_workers")
    ax.set_xticks(data["worker_counts"])
    ax.grid(True, linestyle="--", alpha=0.5)
    _savefig(fig, plot_dir, "05_num_workers.png")


def _plot_batch_scaling(data: Dict, plot_dir: Path, name: str):
    """Experiment 6: samples/sec vs batch size."""
    bs   = data["batch_sizes"]
    sps  = data["samples_per_sec"]
    tfwd = np.array(data["mean_forward_time_s"]) * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(bs, sps, marker="o", color=COLOURS["forward"], linewidth=2)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Samples / second")
    ax1.set_title("GPU throughput vs. batch size")
    ax1.set_xticks(bs)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(bs, tfwd, marker="s", color=COLOURS["backward"], linewidth=2)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Forward time (ms)")
    ax2.set_title("Forward pass latency vs. batch size")
    ax2.set_xticks(bs)
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _savefig(fig, plot_dir, "06_batch_scaling.png")


def _plot_graph_construction(data: Dict, plot_dir: Path, name: str):
    """Experiment 8 & 12: graph build sub-steps vs forward time."""
    n = len(data["total_graph_build_times_s"])
    x = list(range(1, n + 1))

    spec = np.array(data["spectral_partition_times_s"]) * 1e3
    com  = np.array(data["center_of_mass_times_s"])     * 1e3
    nbr  = np.array(data["neighbor_build_times_s"])     * 1e3
    tot  = np.array(data["total_graph_build_times_s"])  * 1e3
    fwd  = np.array(data["forward_times_s"])            * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.stackplot(
        x, spec, com, nbr,
        labels=["spectral_partition", "center_of_mass", "neighbor_build"],
        alpha=0.85,
    )
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Graph construction sub-step times (stacked)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(x, tot, label="Total build", color=COLOURS["h2d"],     linewidth=2)
    ax2.plot(x, fwd, label="Forward",     color=COLOURS["forward"], linewidth=2, linestyle="--")
    ax2.set_xlabel("Batch index")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title(
        f"Graph build vs. forward  "
        f"(ratio={data.get('graph_build_to_forward_ratio', 0):.2f}x)"
    )
    ax2.legend()

    fig.tight_layout()
    _savefig(fig, plot_dir, "08_graph_construction.png")


def _plot_forward_breakdown(data: Dict, plot_dir: Path, name: str):
    """Experiment 9: percentage pie + bar chart per block."""
    block_pct = data.get("block_pct", {})
    if not block_pct:
        return

    labels = list(block_pct.keys())
    pcts   = [block_pct[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    explode = [0.05] * len(labels)
    ax1.pie(pcts, labels=labels, autopct="%1.1f%%", explode=explode,
            startangle=140, textprops={"fontsize": 7})
    ax1.set_title("Forward pass time distribution")

    y_pos = list(range(len(labels)))
    ax2.barh(y_pos, pcts, color=COLOURS["default"], edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("% of total forward time")
    ax2.set_title("Forward block breakdown")
    ax2.invert_yaxis()
    for i, v in enumerate(pcts):
        ax2.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)

    fig.tight_layout()
    _savefig(fig, plot_dir, "09_forward_breakdown.png")


def _plot_torch_profiler(data: Dict, plot_dir: Path, name: str):
    """Experiments 3 & 10: top-K ops bar chart."""
    top_k = data.get("top_k_ops", [])
    if not top_k:
        return

    names       = [op["name"][:50] for op in top_k[:15]]
    cuda_times  = [op["cuda_time_us"] / 1e3 for op in top_k[:15]]
    cpu_times   = [op["cpu_time_us"]  / 1e3 for op in top_k[:15]]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = list(range(len(names)))
    ax.barh(y, cuda_times, label="CUDA time (ms)", color=COLOURS["forward"],   alpha=0.8)
    ax.barh(y, cpu_times,  label="CPU time (ms)",  color=COLOURS["load"],      alpha=0.6,
            left=cuda_times)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"PyTorch profiler: top-{len(names)} ops [{data.get('mode','')}]")
    ax.legend()

    gpu_busy = data.get("gpu_busy_pct", None)
    if gpu_busy is not None:
        ax.text(
            0.98, 0.02,
            f"GPU busy: {gpu_busy:.1f}%  idle: {100-gpu_busy:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.tight_layout()
    safe_name = name.replace("/", "_")
    _savefig(fig, plot_dir, f"10_{safe_name}_top_ops.png")


def _plot_throughput(data: Dict, plot_dir: Path, name: str):
    """Experiment 11: per-batch throughput metrics."""
    n      = len(data["batch_times_s"])
    x      = list(range(1, n + 1))
    times  = np.array(data["batch_times_s"]) * 1e3
    nodes  = np.array(data["batch_nodes"])
    edges  = np.array(data["batch_edges"])

    nps = nodes / (np.array(data["batch_times_s"]) + 1e-12)
    eps = edges / (np.array(data["batch_times_s"]) + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(x, times, color=COLOURS["forward"])
    axes[0, 0].set_title("Forward time per batch (ms)")
    axes[0, 0].set_xlabel("Batch")
    axes[0, 0].set_ylabel("ms")

    axes[0, 1].plot(x, nps, color=COLOURS["h2d"])
    axes[0, 1].set_title("Nodes / second")
    axes[0, 1].set_xlabel("Batch")
    axes[0, 1].set_ylabel("nodes/s")

    axes[1, 0].plot(x, eps, color=COLOURS["backward"])
    axes[1, 0].set_title("Edges / second")
    axes[1, 0].set_xlabel("Batch")
    axes[1, 0].set_ylabel("edges/s")

    axes[1, 1].scatter(nodes, times, alpha=0.6, color=COLOURS["optim"])
    axes[1, 1].set_title("Forward time vs. #nodes")
    axes[1, 1].set_xlabel("#nodes in batch")
    axes[1, 1].set_ylabel("forward time (ms)")

    fig.suptitle(
        f"Throughput: {data['samples_per_sec']:.2f} samp/s  |  "
        f"{data['nodes_per_sec']:.0f} nodes/s  |  "
        f"{data['edges_per_sec']:.0f} edges/s",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, plot_dir, "11_throughput.png")


def _plot_gpu_util(data: Dict, plot_dir: Path, name: str):
    """GPU utilisation timeline."""
    ts   = data.get("timestamps", [])
    util = data.get("util_pct", [])
    mem  = data.get("mem_util_pct", [])
    pwr  = data.get("power_w", [])

    if not ts:
        log.info("  No GPU timeline data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].fill_between(ts, util, alpha=0.7, color=COLOURS["forward"])
    axes[0].set_ylabel("GPU utilisation (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title(
        f"GPU utilisation  [mean={data.get('mean_util_pct',0):.1f}%  "
        f"max={data.get('max_util_pct',0):.1f}%]"
    )

    axes[1].fill_between(ts, mem, alpha=0.7, color=COLOURS["h2d"])
    axes[1].set_ylabel("Memory util. (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title(f"GPU memory utilisation [mean={data.get('mean_mem_util_pct',0):.1f}%]")

    axes[2].plot(ts, pwr, color=COLOURS["optim"], linewidth=1.2)
    axes[2].set_ylabel("Power (W)")
    axes[2].set_xlabel("Wall time (s)")
    axes[2].set_title(f"Power draw [mean={data.get('mean_power_w',0):.1f}W  "
                       f"max={data.get('max_power_w',0):.1f}W]")

    fig.tight_layout()
    _savefig(fig, plot_dir, "02_gpu_utilisation.png")