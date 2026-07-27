import shutil
import subprocess
import warnings

import numpy as np
import psutil
import torch

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import pynvml

_warned_gpu_util_unavailable = False

def _gpu_index_from_device(device):
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        return device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return torch.cuda.current_device()

def _gpu_utilization_via_nvidia_smi(gpu_index):
    if shutil.which("nvidia-smi") is None:
        return float("nan")
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return float("nan")

def get_gpu_utilization(device=None):
    global _warned_gpu_util_unavailable

    if not torch.cuda.is_available():
        return float("nan")

    gpu_index = _gpu_index_from_device(device)
    if gpu_index is None:
        return float("nan")

    if hasattr(torch.cuda, "utilization"):
        try:
            return float(torch.cuda.utilization(gpu_index))
        except Exception:
            pass
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(rates.gpu)
    except Exception:
        pass

    val = _gpu_utilization_via_nvidia_smi(gpu_index)
    if np.isnan(val) and not _warned_gpu_util_unavailable:
        warnings.warn(
            "Could not read GPU utilization via torch.cuda.utilization, "
            "pynvml, or nvidia-smi. GPU utilization will be logged as NaN."
        )
        _warned_gpu_util_unavailable = True
    return val

def get_cpu_utilization():
    return float(psutil.cpu_percent(interval=None))

def get_memory_utilization():
    return float(psutil.virtual_memory().percent)

class SystemMonitor:
    def __init__(self, device=None):
        self.device = device
        self._gpu_buf = []
        self._cpu_buf = []
        self._mem_buf = []
        self.history = {
            "step": [], "loss": [], "gpu_util": [], "cpu_util": [], "mem_util": [],
        }
        psutil.cpu_percent(interval=None)

    def sample(self):
        self._gpu_buf.append(get_gpu_utilization(self.device))
        self._cpu_buf.append(get_cpu_utilization())
        self._mem_buf.append(get_memory_utilization())

    def commit(self, step_idx, loss_value):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            gpu_mean = float(np.nanmean(self._gpu_buf)) if self._gpu_buf else float("nan")
        cpu_mean = float(np.mean(self._cpu_buf)) if self._cpu_buf else float("nan")
        mem_mean = float(np.mean(self._mem_buf)) if self._mem_buf else float("nan")

        self.history["step"].append(step_idx)
        self.history["loss"].append(float(loss_value))
        self.history["gpu_util"].append(gpu_mean)
        self.history["cpu_util"].append(cpu_mean)
        self.history["mem_util"].append(mem_mean)

        self._gpu_buf.clear()
        self._cpu_buf.clear()
        self._mem_buf.clear()

    def save_csv(self, out_path):
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "gpu_util_pct", "cpu_util_pct", "mem_util_pct"])
            for row in zip(
                self.history["step"], self.history["loss"], self.history["gpu_util"],
                self.history["cpu_util"], self.history["mem_util"],
            ):
                writer.writerow(row)

    def save_plots(self, out_path, title_prefix=""):
        if plt is None:
            warnings.warn("matplotlib is not installed; skipping plot generation.")
            return
        if not self.history["step"]:
            warnings.warn("No monitor history to plot (0 accumulated steps recorded).")
            return

        steps = self.history["step"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        panels = [
            ("loss", "Loss (MAE)", axes[0, 0]),
            ("gpu_util", "GPU Utilization (%)", axes[0, 1]),
            ("cpu_util", "CPU Utilization (%)", axes[1, 0]),
            ("mem_util", "System Memory Utilization (%)", axes[1, 1]),
        ]
        for key, ylabel, ax in panels:
            ax.plot(steps, self.history[key], linewidth=1.2)
            ax.set_xlabel("Accumulated batch step")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(alpha=0.3)

        title = "Training resource usage"
        if title_prefix:
            title = f"{title_prefix} -- {title}"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)