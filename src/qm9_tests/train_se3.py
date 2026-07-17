"""
train_se3_transformer_pytorch.py
----------------------------------
Trains the `se3-transformer-pytorch` (lucidrains) model on QM9, mirroring
train.py as closely as possible so the two models are directly comparable:

  - same load_qm9() split (110000 / 10000 / rest), same --target semantics
  - same gradient-accumulation loop (effective batch = batch_size * accum_steps)
  - same Adam + CosineAnnealingLR schedule
  - same 5-trial loop with per-trial seeding (torch.manual_seed(trial)) and
    best-val-MAE checkpointing
  - same confidence_interval_95 across trials

Checkpoints are written as `authors_trial{trial}.pt`, matching the default
`--checkpoint_template` in predict_se3_transformer_pytorch.py, so you can
run that script unmodified afterwards.

Model/data notes (see predict_se3_transformer_pytorch.py docstring for the
full explanation):
  - se3-transformer-pytorch has no built-in QM9 regression head, so this
    reuses SE3RegressionWrapper (backbone -> masked mean-pool -> linear)
    from predict_se3_transformer_pytorch.py.
  - Inputs are converted to dense (atoms, coors, mask, edges) tensors via
    batch_to_dense, also reused from that script, so both scripts stay in
    sync on the data adapter.
  - Dense edge tensors are O(n^2) per graph, so this defaults to a smaller
    physical batch size than the custom model and a larger accum_steps to
    reach the same effective batch size -- adjust --batch_size /
    --accum_steps together if you want to match the custom model's
    effective batch size exactly.

Usage:
    python train_se3_transformer_pytorch.py --target 0 --epochs 300 \
        --batch_size 8 --accum_steps 32 --data_root ./data

If you hit CUDA OOM (this library's dense O(n^2) kernel over per-batch
variable-size padding is memory-hungry and prone to allocator fragmentation),
try in this order:
    1. Lower --batch_size further (e.g. 4) and raise --accum_steps to
       compensate, to keep the effective batch size the same.
    2. Lower --dim / --heads / --dim_head / --num_degrees.
    3. Lower --empty_cache_every (e.g. to 1, the default) if you raised it.
    4. As a last resort, run with fewer --min_nodes filtering so the largest
       molecules (up to 29 atoms in QM9) are excluded, capping worst-case N.
The script already sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and
skips (rather than crashes on) any individual batch that still OOMs.
"""

import argparse
import math
import os

# Must be set before CUDA is initialized (i.e. before any tensor touches the
# GPU). This library pads every batch to that batch's own max atom count, so
# tensor shapes change batch-to-batch; the caching allocator fragments badly
# under that pattern, and this flag lets it recycle fragments instead of
# growing the pool indefinitely. Override by exporting the env var yourself
# before running if you want a different value.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import stats

from src.train import load_qm9, _filter_small_graphs
from src.tests.pred_se3_orig import SE3RegressionWrapper, batch_to_dense
from src.training_monitor import SystemMonitor


def confidence_interval_95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan")
    std = values.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    half_width = float(t_crit * std / math.sqrt(n))
    return mean, half_width


def train_epoch(model, loader, optimizer, device, min_nodes, accum_steps=1,
                 empty_cache_every=1, monitor: SystemMonitor = None):
    """Gradient-accumulation training epoch, mirroring train.py::train_epoch.

    Two additions vs. a naive port, both aimed at the OOM behavior this
    library exhibits on variable-size padded batches:
      - torch.cuda.empty_cache() every `empty_cache_every` optimizer steps,
        to return freed fragments to the allocator's shared pool instead of
        letting them pile up as batch shapes keep changing.
      - a per-batch OOM guard: if a single unlucky (large-N) batch overflows,
        we drop that batch, clear the cache, and keep training instead of
        losing the whole epoch/trial.

    monitor: optional SystemMonitor, sample()d once per micro-batch and
             commit()ted once per accumulated step (mean over that step's
             micro-batches), matching train.py's monitoring.
    """
    model.train()
    total_loss = 0.0
    n_graphs = 0
    accum_loss = torch.tensor(0.0, device=device)
    skipped = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, min_nodes, device)
        if tensors is None:
            continue
        _, _, _, _, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue

        atoms = coors = mask = edges = pred = loss = None
        try:
            atoms, coors, mask, edges = batch_to_dense(batch, device)
            pred = model(atoms, coors, mask, edges)
            if pred.shape[0] != target.shape[0]:
                            continue # Skipping last batch
            loss = nn.functional.l1_loss(pred, target) / accum_steps
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            n_atoms = atoms.shape[1] if atoms is not None else "?"
            print(f"[OOM] skipping batch {i + 1} (N={n_atoms}, "
                  f"B={target.shape[0]}) and clearing cache")
            optimizer.zero_grad()
            del atoms, coors, mask, edges, pred, loss
            torch.cuda.empty_cache()
            skipped += 1
            continue

        if monitor is not None:
            monitor.sample()

        micro_batch_mae = loss.item() * accum_steps
        accum_loss = accum_loss + loss.detach()

        B = pred.shape[0]
        total_loss += loss.item() * accum_steps * B
        n_graphs += B

        del atoms, coors, mask, edges, pred, loss

        step_idx = i + 1
        if step_idx % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Batch {step_idx} | accum MAE: {accum_loss.item():.4f}")
            if monitor is not None:
                monitor.commit(step_idx, accum_loss.item())
            accum_loss = torch.tensor(0.0, device=device)
            if (step_idx // accum_steps) % empty_cache_every == 0:
                torch.cuda.empty_cache()
        else:
            print(f"Batch {step_idx} | micro-batch MAE: {micro_batch_mae:.4f} "
                  f"({step_idx % accum_steps}/{accum_steps})")

    remainder = len(loader) % accum_steps
    if remainder != 0:
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        print(f"Flushed {remainder} remaining micro-batch(es).")
        if monitor is not None:
            monitor.commit(len(loader), accum_loss.item())

    if skipped:
        print(f"[epoch summary] skipped {skipped} batch(es) due to OOM.")

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, device, min_nodes, empty_cache_every=1):
    model.eval()
    total_mae = 0.0
    n_graphs = 0
    skipped = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        tensors = _filter_small_graphs(batch, min_nodes, device)
        if tensors is None:
            continue
        _, _, _, _, target, graph_batch = tensors
        if graph_batch.max() < 0:
            continue

        atoms = coors = mask = edges = pred = None
        try:
            atoms, coors, mask, edges = batch_to_dense(batch, device)
            pred = model(atoms, coors, mask, edges)
            if pred.shape[0] != target.shape[0]:
                continue # Skipping last batch
            mae = nn.functional.l1_loss(pred, target).item()
        except torch.cuda.OutOfMemoryError:
            n_atoms = atoms.shape[1] if atoms is not None else "?"
            print(f"[OOM] skipping eval batch {i + 1} (N={n_atoms}, "
                  f"B={target.shape[0]}) and clearing cache")
            del atoms, coors, mask, edges, pred
            torch.cuda.empty_cache()
            skipped += 1
            continue

        B = pred.shape[0]
        total_mae += mae * B
        n_graphs += B
        print(f"[Eval Batch {i + 1}] MAE: {mae:.4f}")

        del atoms, coors, mask, edges, pred
        if (i + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    if skipped:
        print(f"[eval summary] skipped {skipped} batch(es) due to OOM.")

    return total_mae / max(n_graphs, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--min_nodes", type=int, default=4,
                         help="Drop graphs with fewer atoms than this. Set equal to "
                              "the custom model's --num_parts to evaluate both models "
                              "on exactly the same set of molecules.")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Dense (b, n, n) edge tensors are memory-heavy; "
                              "keep this smaller than the custom model's batch size. "
                              "Lowered from an earlier default of 16 after OOMs on a "
                              "39GB GPU -- raise it back if you have headroom.")
    parser.add_argument("--accum_steps", type=int, default=32,
                         help="Gradient accumulation steps. Effective batch = "
                              "batch_size * accum_steps.")
    parser.add_argument("--empty_cache_every", type=int, default=1,
                         help="Call torch.cuda.empty_cache() every N optimizer "
                              "steps (train) / N batches (eval). This library pads "
                              "every batch to that batch's own max atom count, so "
                              "shapes vary batch-to-batch and the allocator "
                              "fragments; periodic clearing keeps that in check. "
                              "Set higher (e.g. 4-8) if this is slowing you down "
                              "and you're not near the memory ceiling.")
    parser.add_argument("--dim", type=int, default=32,
                         help="Lowered from an earlier default of 64 -- the dense "
                              "pairwise kernel scales with dim * heads * dim_head "
                              "per degree pair, so this is the biggest lever for "
                              "memory besides batch_size.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_degrees", type=int, default=2)
    parser.add_argument("--dim_head", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--data_root", type=str,
                         default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--checkpoint_dir", type=str, default=".")
    parser.add_argument("--checkpoint_template", type=str,
                         default="authors_trial{trial}.pt")
    parser.add_argument("--metrics_dir", type=str, default="./training_metrics",
                         help="Directory for per-trial loss/GPU/CPU/memory "
                              "utilization CSVs and plots.")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    device = torch.device(args.device)
    effective_batch = args.batch_size * args.accum_steps
    print(f"Device:           {device}")
    print(f"QM9 target:       {args.target}")
    print(f"Physical batch:   {args.batch_size}")
    print(f"Accum steps:      {args.accum_steps}")
    print(f"Effective batch:  {effective_batch}")

    train_loader, val_loader, test_loader = load_qm9(
        args.target, args.batch_size, args.data_root, args.device
    )

    trial_maes = []
    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1} / {args.trials} ---")
        torch.manual_seed(trial)

        model = SE3RegressionWrapper(
            dim=args.dim, depth=args.depth, num_degrees=args.num_degrees,
            dim_head=args.dim_head, heads=args.heads,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        monitor = SystemMonitor(device)

        ckpt_path = os.path.join(
            args.checkpoint_dir, args.checkpoint_template.format(trial=trial)
        )

        best_val_mae = float("inf")
        for epoch in range(args.epochs):
            print(f"[Epoch {epoch + 1}]")
            train_loss = train_epoch(
                model, train_loader, optimizer, device, args.min_nodes,
                accum_steps=args.accum_steps,
                empty_cache_every=args.empty_cache_every,
                monitor=monitor,
            )
            val_mae = evaluate(model, val_loader, device, args.min_nodes,
                                empty_cache_every=args.empty_cache_every)
            scheduler.step()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"  [New best] val MAE: {val_mae:.4f} -- saving checkpoint.")
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train MAE: {train_loss:.4f} | val MAE: {val_mae:.4f}")

        monitor.save_csv(os.path.join(args.metrics_dir, f"metrics_trial{trial}.csv"))
        monitor.save_plots(
            os.path.join(args.metrics_dir, f"metrics_trial{trial}.png"),
            title_prefix=f"se3-transformer-pytorch -- trial {trial}",
        )

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_mae = evaluate(model, test_loader, device, args.min_nodes,
                             empty_cache_every=args.empty_cache_every)
        print(f"  Trial {trial + 1} test MAE: {test_mae:.4f}")
        trial_maes.append(test_mae)

    mean_mae, half_width = confidence_interval_95(trial_maes)
    print(f"\nTest MAE over {args.trials} trials: {mean_mae:.4f} +/- {half_width:.4f}  (95% CI)")


if __name__ == "__main__":
    main()