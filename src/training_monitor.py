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
        self._grad_norm = float("nan")
        self._weight_norm = float("nan")
        self._grad_weight_ratio = float("nan")
        self._step_counter = 0
        self.history = {
            "step": [], "loss": [], "gpu_util": [], "cpu_util": [], "mem_util": [],
            "grad_norm": [], "weight_norm": [], "grad_weight_ratio": [],
        }
        psutil.cpu_percent(interval=None)

    def sample(self):
        self._gpu_buf.append(get_gpu_utilization(self.device))
        self._cpu_buf.append(get_cpu_utilization())
        self._mem_buf.append(get_memory_utilization())

    def record_grad_stats(self, model):
        grad_norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
        weight_norms = [p.detach().norm(2) for p in model.parameters()]

        grad_norm = float(torch.norm(torch.stack(grad_norms), 2)) if grad_norms else 0.0
        weight_norm = float(torch.norm(torch.stack(weight_norms), 2)) if weight_norms else 0.0

        self._grad_norm = grad_norm
        self._weight_norm = weight_norm
        self._grad_weight_ratio = grad_norm / (weight_norm + 1e-18)

    def commit(self, loss_value):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            gpu_mean = float(np.nanmean(self._gpu_buf)) if self._gpu_buf else float("nan")
        cpu_mean = float(np.mean(self._cpu_buf)) if self._cpu_buf else float("nan")
        mem_mean = float(np.mean(self._mem_buf)) if self._mem_buf else float("nan")

        self._step_counter += 1
        self.history["step"].append(self._step_counter)
        self.history["loss"].append(float(loss_value))
        self.history["gpu_util"].append(gpu_mean)
        self.history["cpu_util"].append(cpu_mean)
        self.history["mem_util"].append(mem_mean)
        self.history["grad_norm"].append(self._grad_norm)
        self.history["weight_norm"].append(self._weight_norm)
        self.history["grad_weight_ratio"].append(self._grad_weight_ratio)

        self._gpu_buf.clear()
        self._cpu_buf.clear()
        self._mem_buf.clear()
        self._grad_norm = float("nan")
        self._weight_norm = float("nan")
        self._grad_weight_ratio = float("nan")

    def save_csv(self, out_path):
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "loss", "gpu_util_pct", "cpu_util_pct", "mem_util_pct",
                "grad_norm", "weight_norm", "grad_weight_ratio",
            ])
            for row in zip(
                self.history["step"], self.history["loss"], self.history["gpu_util"],
                self.history["cpu_util"], self.history["mem_util"],
                self.history["grad_norm"], self.history["weight_norm"],
                self.history["grad_weight_ratio"],
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
        fig, axes = plt.subplots(2, 3, figsize=(16, 7))
        panels = [
            ("loss", "Loss (MAE)", axes[0, 0]),
            ("gpu_util", "GPU Utilization (%)", axes[0, 1]),
            ("cpu_util", "CPU Utilization (%)", axes[0, 2]),
            ("mem_util", "System Memory Utilization (%)", axes[1, 0]),
            ("grad_norm", "Gradient Norm (L2)", axes[1, 1]),
            ("grad_weight_ratio", "Grad Norm / Weight Norm", axes[1, 2]),
        ]
        for key, ylabel, ax in panels:
            ax.plot(steps, self.history[key], linewidth=1.2)
            ax.set_xlabel("Optimizer step")
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