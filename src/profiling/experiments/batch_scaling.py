import time
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn

log = logging.getLogger("profiler.batch_scaling")

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ATOM_TYPES  = [1, 6, 7, 8, 9]


def _one_hot_z(z):
    one_hot = torch.zeros(z.shape[0], len(ATOM_TYPES))
    for idx, a in enumerate(ATOM_TYPES):
        one_hot[:, idx] = (z == a).float()
    return one_hot


def _atomic_masses(z):
    M = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}
    return torch.tensor([M.get(zi.item(), 12.0) for zi in z], dtype=torch.float32)


def run_batch_scaling_experiment(
    args,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Any]:
    log.info(f"  Batch scaling experiment on {device}")

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))
    from src.se3_crossformer.model import SE3IntraOnlyTransformer
    from src.se3_crossformer.se3_utils import RadialNetworkGRBF
    from src.load_data import CustomQM9Dataset
    from torch_geometric.loader import DataLoader

    dataset = CustomQM9Dataset(
        root     = args.data_root + "/qm1",
        sdf_file = args.data_root + "/qm9/raw/gdb9.sdf",
        csv_file = args.data_root + "/qm9/raw/gdb9.sdf.csv",
        device   = torch.device("cpu"),
    )
    dataset = dataset[:min(len(dataset), 2048)]

    model = SE3IntraOnlyTransformer(
        radial_net  = RadialNetworkGRBF,
        in_features = len(ATOM_TYPES),
        max_degree  = args.max_degree,
        feature_dim = args.feature_dim,
        hidden_dim  = args.hidden_dim,
        radius_cutoff = args.radius_cutoff,
        scalar_out_dim = 1,
        task        = 0,
        bond_feature_dim  = getattr(args, "bond_feature_dim", 0),
    ).to(device)
    model.eval()

    valid_batch_sizes: List[int]   = []
    sps_list:          List[float] = []
    fwd_time_list:     List[float] = []
    oom_at: int = -1

    for bs in BATCH_SIZES:
        if len(dataset) < bs:
            log.info(f"  bs={bs}: skipped (dataset too small)")
            continue

        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
        times:    List[float] = []
        n_total:  int = 0

        try:
            loader_iter = iter(loader)
            for _ in range(num_batches):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                batch       = batch.to(device)
                node_feat   = _one_hot_z(batch.x).to(device)
                pos         = batch.pos.to(device)
                edge_index  = batch.edge_index.to(device)
                edge_attr   = batch.edge_attr.to(device)
                am          = _atomic_masses(batch.x).to(device)
                graph_batch = batch.batch.to(device)
                graph_idx   = batch.idx.to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(node_feat, pos, edge_index, am, graph_batch,
                              edge_attr=edge_attr, graph_idx=graph_idx)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                B = int(graph_batch.max().item()) + 1
                times.append(t1 - t0)
                n_total += B

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_at = bs
                log.warning(f"  bs={bs}: OOM – stopping batch sweep here")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise

        total_t = sum(times)
        sps     = n_total / total_t if total_t > 0 else 0.0
        mean_t  = total_t / len(times) if times else 0.0
        valid_batch_sizes.append(bs)
        sps_list.append(sps)
        fwd_time_list.append(mean_t)
        log.info(
            f"  bs={bs:>4d}: {sps:.1f} samples/s  mean_fwd={mean_t*1e3:.1f}ms"
        )

    best_idx = int(max(range(len(sps_list)), key=lambda i: sps_list[i])) if sps_list else 0
    return {
        "batch_sizes":           valid_batch_sizes,
        "samples_per_sec":       sps_list,
        "mean_forward_time_s":   fwd_time_list,
        "oom_at":                oom_at,
        "best_batch_size":       valid_batch_sizes[best_idx] if valid_batch_sizes else -1,
        "best_samples_per_sec":  sps_list[best_idx] if sps_list else 0.0,
        "gpu_underutilised":     (
            len(sps_list) > 2 and sps_list[-1] / max(sps_list[0], 1e-9) > 2.0
        ),
    }