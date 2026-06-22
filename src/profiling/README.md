# SE3 Transformer Profiling Suite

Diagnostic suite for identifying GPU under-utilisation and per-batch
slowdowns in the `SE3InterNeighborhoodTransformer` training loop.

---

## Quick start

```bash
# Full suite (all experiments, 20 batches each)
python profiling/profile_runner.py \
    --target 0 \
    --num_parts 4 \
    --max_degree 2 \
    --data_root ./data \
    --batch_size 16 \
    --profile_batches 20 \
    --experiments all

# Selected experiments only
python profiling/profile_runner.py \
    --experiments timing,forward_breakdown,batch_scaling

# Run GPU + CPU monitors in parallel (separate terminals)
chmod +x profiling/gpu_util_monitor.sh profiling/cpu_monitor.sh
./profiling/gpu_util_monitor.sh ./profiling_results 120 &
./profiling/cpu_monitor.sh      ./profiling_results 2 120
```

Results are written to `profiling_results/profile_<timestamp>.json` and
`profiling_results/profile_<timestamp>.log`.  Plots are saved to
`profiling_results/plots_<timestamp>/`.

---

## Different bottleneck areas:
- Data loading
- CPU preprocessing
- Graph construction
- Host → device transfer
- Python overhead
- Synchronization bottlenecks
- Inefficient kernels
- Memory bandwidth
- Model FLOPs

## All Experiments:
# Timing:
For each batch record the time for the following:
- Loading next batch
- Transferring batch data to device
- Getting model predictions (forward)
- Backprop
- Optimizer step
- Plot data loading time, H2D transfer time, forward time, backward time, and optimizer time per batch

# GPU Utilization timeline:
- Run nvidia-smi dmon
- Run watch -n 1 nvidia-smi
- Track GPU utilization, memory utilization, and power draw

# Pytorch profiler
- Use small subset of training data and run Pytorch profiler on it
Look for slowdown in:
- Scatter operations
- Gather operations
- Edge construction
- Attention kernels
- Tensor indexing
- Python calls

# Dataloader test:
Measure time to load data using loader only vs. when transferring data to device

# Num workers test:
- Run dataloading using a range of num_workers ([0, 16])
- Plot samples/sec
- If performance increases with workers, it is a CPU preprocessing bottleneck. 
- If not, the process isn’t dataloader limited

# Batch size scaling:
- Run model on small subset of data using a range of batch sizes [1, 512]
- Plot samples/sec
- If performance increases, GPU was not being utilized enough
- If not, some fixed overhead is dominating

# CPU utilization:
- 1 core at 100% → Python bottleneck
- All cores at 100% → Data processing bottleneck
- All cores idle → Could be a synchronization or GPU issue

# Graph construction:
- Measure the time for each subprocess in the graph construction
- Plot the graph build time vs forward time

# Forward process:
- Time each major block and plot the percentage of the total forward time each block comprises
- Use torch.profiler to examine the kernel launch constraint and kernel duration
- If kernel launch is much higher than useful work, GPU is spending all its time launching kernels
- Track nodes/sec, edges/sec, samples/sec, spherical harmonic accesses/sec
Next, for every graph-related step, record:
- Radius graph construction
- Neighbor search
- Edge feature computation
- Spherical harmonic computation
- Basis construction
To evaluate whether they occur once per molecule, epoch, batch, or forward pass
- Compute GPU busy time and GPU idle time from the profiler

## Experiments

| # | Key | What it measures |
|---|-----|-----------------|
| 1 | `timing` | Per-batch: load / H2D / forward / backward / optim times |
| 2 | `gpu_util` | GPU util %, memory util %, power draw (background thread) |
| 2 | `gpu_util_monitor.sh` | `nvidia-smi dmon` + snapshot timeline |
| 3 | `torch_profiler` | PyTorch op-level profiler; scatter/gather/attn breakdown |
| 4 | `dataloader` | Load-only vs load+H2D time |
| 5 | `num_workers` | Samples/sec across `num_workers ∈ {0,2,4,8,16}` |
| 6 | `batch_scaling` | Samples/sec across batch sizes `{1…512}` |
| 7 | `cpu_monitor.sh` | Per-core CPU utilisation via `mpstat` |
| 8 | `graph_construction` | Per sub-step timing: spectral partition / CoM / neighbor build |
| 9 | `forward_breakdown` | % of forward time per model block (intra/inter/cross/readout) |
| 10 | `kernel_launch` | Kernel launch count vs. useful-work time (via torch.profiler) |
| 11 | `throughput` | nodes/s, edges/s, samples/s, spherical-harmonic accesses/s |
| 12 | `graph_construction` | Frequency annotation per graph-pipeline step |
| 13 | `torch_profiler` | GPU busy time / GPU idle time from profiler data |

---

## Interpreting results

### Timing (Exp 1)
```
dominant_phase: "forward"   → model compute is the bottleneck
dominant_phase: "load"      → dataloader is too slow; increase num_workers
dominant_phase: "h2d"       → PCIe bandwidth saturated; use pin_memory=True
dominant_phase: "backward"  → gradient computation is slow; check for Python loops
```

### GPU utilisation (Exp 2)
```
mean_util_pct < 30%         → GPU severely under-utilised; check other phases
mean_util_pct > 80%         → GPU is the bottleneck (good!)
mean_power_w  ≈ TDP         → GPU is fully saturated
```

### DataLoader (Exp 4)
```
h2d_overhead_s >> mean_load_only_s → PCIe is the bottleneck
mean_load_only_s is large          → Disk I/O or CPU preprocessing
```

### num_workers (Exp 5)
```
sps increases with workers          → CPU preprocessing bottleneck
sps plateaus after 2-4 workers      → Not dataloader-limited
```

### Batch scaling (Exp 6)
```
sps rises steeply up to bs=64+     → GPU under-utilised at small batch sizes
sps plateaus early (bs=8)          → Memory-bandwidth or fixed-overhead bound
OOM at bs=64                       → GPU memory is the constraint
```

### CPU utilisation (Exp 7)
```
1 core at 100%, rest idle           → Python GIL / single-threaded bottleneck
All cores at 100%                   → CPU data preprocessing bottleneck
All cores idle + GPU low util       → Synchronisation bottleneck
```

### Graph construction (Exp 8)
```
graph_build_to_forward_ratio > 1.0  → Graph construction dominates; cache or vectorise
spectral_partition >> others        → Laplacian eigh + k-means is the bottleneck
neighbor_build  is large            → Python loop in _build_neighbor_info is slow
```

### Forward breakdown (Exp 9)
```
cross_update > 50%                  → _cross_update loop over subgraphs is the bottleneck
intra_update > 40%                  → equivariant_weight_matrix is slow (SH + CG)
initial_message > 10%               → Python loop in initial_message is slow
```

### Kernel launch (Exp 10)
```
kernel_launch_count >> useful work  → Too many small kernels; batch or fuse operations
gpu_busy_pct < 40%                  → GPU spends most time waiting for kernel launches
```

### Throughput (Exp 11)
```
nodes_per_sec                       → Absolute node throughput
sh_accesses_per_sec                 → Estimated spherical harmonic compute rate
```

---

## Output files

```
profiling_results/
├── profile_<ts>.json              # All numeric results
├── profile_<ts>.log               # Full console log
├── torch_profiler/                # TensorBoard trace files
├── torch_profiler_trace.json      # Chrome trace (open in chrome://tracing)
├── gpu_dmon_<ts>.log              # nvidia-smi dmon raw output
├── gpu_snapshots_<ts>.log         # nvidia-smi periodic snapshots
├── cpu_util_<ts>.log              # mpstat per-core output
└── plots_<ts>/
    ├── 01_timing_breakdown.png
    ├── 02_gpu_utilisation.png
    ├── 04_dataloader.png
    ├── 05_num_workers.png
    ├── 06_batch_scaling.png
    ├── 08_graph_construction.png
    ├── 09_forward_breakdown.png
    ├── 10_torch_profiler_top_ops.png
    └── 11_throughput.png
```

---

## Common fixes (based on expected findings)

| Finding | Likely fix |
|---------|-----------|
| `graph_build/forward_ratio > 1` | Cache spectral partition per molecule; it doesn't change across epochs |
| `_build_neighbor_info` is slow | Replace Python `adj` list loop with `torch.isin` + masked `edge_index` |
| `_cross_update` loop over S | Vectorise the `for b in range(S)` loop with batched einsum |
| `initial_message` slow | Replace Python loop with `scatter_mean` from `torch_scatter` |
| `gpu_busy_pct < 30%` | Increase batch size; enable `pin_memory=True`; prefetch |
| `num_workers` helps a lot | The DataLoader is CPU-preprocessing limited |
| H2D dominates | Set `pin_memory=True` and `non_blocking=True` in `.to(device)` |
| `x_to_alpha_beta` loop | Vectorise: use `torch.atan2` / `torch.acos` directly on the full tensor |