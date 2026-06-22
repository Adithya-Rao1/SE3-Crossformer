import threading
import subprocess
import time
import logging
from typing import List, Dict, Any

log = logging.getLogger("profiler.gpu_monitor")


class GpuMonitor:
    """
    Runs ``nvidia-smi dmon`` in a subprocess and parses its output into
    time-stamped lists.  Use as::

        mon = GpuMonitor(poll_interval=1.0)
        mon.start()
        # ... do training ...
        mon.stop()
        results = mon.get_results()   # dict of lists
    """

    def __init__(self, poll_interval: float = 1.0, gpu_id: int = 0):
        self.poll_interval = poll_interval
        self.gpu_id        = gpu_id
        self._thread       = None
        self._stop_event   = threading.Event()

        self._timestamps:       List[float] = []
        self._util_pct:         List[float] = []
        self._mem_util_pct:     List[float] = []
        self._power_w:          List[float] = []
        self._mem_used_mib:     List[float] = []
        self._lock             = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def get_results(self) -> Dict[str, Any]:
        with self._lock:
            util      = list(self._util_pct)
            mem_util  = list(self._mem_util_pct)
            power     = list(self._power_w)
            mem_used  = list(self._mem_used_mib)
            ts        = list(self._timestamps)

        def safe_mean(lst):
            return float(sum(lst) / len(lst)) if lst else 0.0

        def safe_max(lst):
            return float(max(lst)) if lst else 0.0

        return {
            "timestamps":        ts,
            "util_pct":          util,
            "mem_util_pct":      mem_util,
            "power_w":           power,
            "mem_used_mib":      mem_used,
            # summary stats
            "mean_util_pct":     safe_mean(util),
            "max_util_pct":      safe_max(util),
            "mean_mem_util_pct": safe_mean(mem_util),
            "max_mem_util_pct":  safe_max(mem_util),
            "mean_power_w":      safe_mean(power),
            "max_power_w":       safe_max(power),
            "mean_mem_used_mib": safe_mean(mem_used),
            "peak_mem_used_mib": safe_max(mem_used),
        }

    # ── internal ──────────────────────────────────────────────────────────

    def _run(self):
        """Poll loop – tries dmon first, falls back to manual smi queries."""
        try:
            self._run_dmon()
        except Exception as e:
            log.warning(f"nvidia-smi dmon failed ({e}), falling back to manual polling.")
            self._run_manual()

    def _run_dmon(self):
        """
        Uses ``nvidia-smi dmon -s pum -d <interval>`` which prints:
          # gpu   pwr  gtemp  mtemp     sm    mem    enc    dec
        columns vary by driver; we pick 'sm', 'mem', 'pwr' by index.
        """
        cmd = [
            "nvidia-smi", "dmon",
            "-i", str(self.gpu_id),
            "-s", "pum",          # p=power, u=utilisation, m=memory
            "-d", str(max(1, int(self.poll_interval))),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        t0 = time.perf_counter()

        header_line = None
        for line in proc.stdout:
            if self._stop_event.is_set():
                break

            line = line.strip()
            if not line:
                continue

            # Capture header so we know column order
            if line.startswith("#"):
                if "sm" in line.lower():
                    header_line = line.lstrip("#").split()
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                now = time.perf_counter() - t0
                if header_line:
                    idx = {h.lower(): i for i, h in enumerate(header_line)}
                    util     = float(parts[idx.get("sm",  1)])
                    mem_util = float(parts[idx.get("mem", 2)])
                    power    = float(parts[idx.get("pwr", 0)])
                else:
                    # Fallback: assume gpu, pwr, sm, mem order
                    power    = float(parts[1])
                    util     = float(parts[3])
                    mem_util = float(parts[4])

                with self._lock:
                    self._timestamps.append(now)
                    self._util_pct.append(util)
                    self._mem_util_pct.append(mem_util)
                    self._power_w.append(power)
                    self._mem_used_mib.append(0.0)   # dmon pum doesn't give MiB directly
            except (ValueError, IndexError, KeyError):
                continue

        proc.terminate()

    def _run_manual(self):
        """
        Falls back to querying individual nvidia-smi fields every poll_interval.
        Works even if dmon is unsupported.
        """
        t0 = time.perf_counter()
        query = (
            "utilization.gpu,utilization.memory,power.draw,memory.used"
        )
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]

        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, timeout=5).strip()
                parts = [p.strip() for p in out.split(",")]
                util     = float(parts[0])
                mem_util = float(parts[1])
                power    = float(parts[2])
                mem_used = float(parts[3])

                now = time.perf_counter() - t0
                with self._lock:
                    self._timestamps.append(now)
                    self._util_pct.append(util)
                    self._mem_util_pct.append(mem_util)
                    self._power_w.append(power)
                    self._mem_used_mib.append(mem_used)
            except Exception as e:
                log.debug(f"nvidia-smi query failed: {e}")

            self._stop_event.wait(self.poll_interval)