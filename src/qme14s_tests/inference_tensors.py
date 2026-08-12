import argparse
import os

import h5py
import numpy as np
import torch

from src.qme14s_tests.train_qme14s import (
    ATOM_TYPES,
    one_hot_z,
    get_atomic_masses,
    build_model,
)

def load_trained_model(checkpoint_path: str, field: str, args, device):
    class _Args:
        pass
    a = _Args()
    a.field = field
    a.max_degree = args.max_degree
    a.feature_dim = args.feature_dim
    a.hidden_dim = args.hidden_dim
    a.radius_cutoff = args.radius_cutoff

    model = build_model(a, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def predict_single(pos, z, polar_model, dedipole_model, device, edge_index=None):
    pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z, dtype=torch.long, device=device)
    n = pos_t.shape[0]
    # edge_index is only ever used for optional bond-feature gathering (the
    # model builds its own radius-cutoff neighbor list from `pos` internally);
    # an empty edge_index is fine when no bond connectivity is available.
    if edge_index is None:
        edge_index_t = torch.zeros(2, 0, dtype=torch.long, device=device)
    else:
        edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    node_feat = one_hot_z(z_t.cpu()).to(device)
    atomic_mass = get_atomic_masses(z_t.cpu()).to(device)
    graph_batch = torch.zeros(n, dtype=torch.long, device=device)

    out = {}
    if polar_model is not None:
        pred = polar_model(node_feat, pos_t, edge_index_t, atomic_mass, graph_batch)
        out["polarizability"] = pred.squeeze(0).cpu().numpy()
    if dedipole_model is not None:
        pred = dedipole_model(node_feat, pos_t, edge_index_t, atomic_mass, graph_batch)
        out["dipole_derivative"] = pred.cpu().numpy()
    return out

def predict_dataset(h5_path, out_path, polar_model, dedipole_model, device):
    n_written = 0
    n_skipped = 0
    with h5py.File(h5_path, "r") as fin, h5py.File(out_path, "w") as fout:
        for group_name in fin.keys():
            group = fin[group_name]
            try:
                pos = np.asarray(group["pos"][:], dtype=np.float32)
                z = np.asarray(group["z"][:], dtype=np.int64)
                edge_index = np.asarray(group["edge_index"][:], dtype=np.int64)
            except KeyError as e:
                print(f"Skipping {group_name}: missing {e}")
                n_skipped += 1
                continue

            try:
                preds = predict_single(pos, z, polar_model, dedipole_model, device, edge_index=edge_index)
            except Exception as e:
                print(f"Skipping {group_name}: inference failed ({e})")
                n_skipped += 1
                continue

            g = fout.create_group(group_name)
            g.create_dataset("pos", data=pos)
            g.create_dataset("z", data=z)
            if "smile" in group.attrs:
                g.attrs["smile"] = group.attrs["smile"]

            if "polarizability" in preds:
                g.create_dataset("pred_polarizability", data=preds["polarizability"])
                if "polar" in group:
                    g.create_dataset(
                        "true_polarizability",
                        data=np.asarray(group["polar"][:], dtype=np.float32).reshape(3, 3),
                    )
            if "dipole_derivative" in preds:
                g.create_dataset("pred_dipole_derivative", data=preds["dipole_derivative"])
                if "dedipole" in group:
                    g.create_dataset(
                        "true_dipole_derivative",
                        data=np.asarray(group["dedipole"][:], dtype=np.float32).reshape(pos.shape[0], 3, 3),
                    )
            n_written += 1

    print(f"Wrote predictions for {n_written} molecule(s) to {out_path} "
          f"({n_skipped} skipped).")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", type=str, required=True,
                    help="Source HDF5 (same schema as HDF5Dataset) to run inference over.")
    p.add_argument("--out_path", type=str, default="./predictions.h5")
    p.add_argument("--polar_checkpoint", type=str, default=None,
                    help="Path to a checkpoint for the polar field. "
                         "Omit to skip polarizability prediction.")
    p.add_argument("--dedipole_checkpoint", type=str, default=None,
                    help="Path to a checkpoint for the dedipole field. "
                         "Omit to skip dipole-derivative prediction.")
    p.add_argument("--radius_cutoff", type=float, default=5.0,
                    help="Must match the radius_cutoff the checkpoint was trained with.")
    p.add_argument("--max_degree", type=int, default=2)
    p.add_argument("--feature_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.polar_checkpoint is None and args.dedipole_checkpoint is None:
        raise ValueError("Provide at least one of --polar_checkpoint / --dedipole_checkpoint.")

    polar_model = None
    if args.polar_checkpoint is not None:
        polar_model = load_trained_model(args.polar_checkpoint, "polar", args, device)
        print(f"Loaded polarizability model from {args.polar_checkpoint}")

    dedipole_model = None
    if args.dedipole_checkpoint is not None:
        dedipole_model = load_trained_model(args.dedipole_checkpoint, "dedipole", args, device)
        print(f"Loaded dipole-derivative model from {args.dedipole_checkpoint}")

    predict_dataset(args.h5_path, args.out_path, polar_model, dedipole_model, device)

if __name__ == "__main__":
    main()
