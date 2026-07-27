import argparse
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference_tensors import build_radius_graph, load_trained_model, predict_single
from train_polarizability_dipole import get_atomic_masses

HARTREE_BOHR2_AMU_TO_CM1 = 5140.487  # sqrt(Hartree / (bohr^2 * amu)) in cm^-1

def normal_modes_from_hessian(hessian, z, unit_factor=HARTREE_BOHR2_AMU_TO_CM1,
                                n_zero_modes=None):
    from train_polarizability_dipole import ATOMIC_MASSES
    n = len(z)
    masses = np.array([ATOMIC_MASSES.get(int(zi), 12.0) for zi in z])
    if n_zero_modes is None:
        n_zero_modes = 5 if n == 2 else 6  # diatomics have only 1 rotational dof

    m3 = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m3)
    mw_hessian = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
    mw_hessian = 0.5 * (mw_hessian + mw_hessian.T) 

    eigvals, eigvecs = np.linalg.eigh(mw_hessian)

    order = np.argsort(np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    keep = slice(n_zero_modes, None)
    eigvals_vib = eigvals[keep]
    eigvecs_vib = eigvecs[:, keep]

    sign = np.sign(eigvals_vib)
    freqs_cm = sign * np.sqrt(np.abs(eigvals_vib)) * unit_factor

    modes_cart = eigvecs_vib * inv_sqrt_m[:, None]
    modes_cart /= np.linalg.norm(modes_cart, axis=0, keepdims=True)

    return freqs_cm, modes_cart

def raman_activity_from_invariants(dalpha_dQ):
    a = dalpha_dQ
    alpha_bar = np.trace(a) / 3.0
    gamma2 = 0.5 * (
        (a[0, 0] - a[1, 1]) ** 2 + (a[1, 1] - a[2, 2]) ** 2 + (a[2, 2] - a[0, 0]) ** 2
        + 6.0 * (a[0, 1] ** 2 + a[1, 2] ** 2 + a[2, 0] ** 2)
    )
    return 45.0 * alpha_bar ** 2 + 7.0 * gamma2

def raman_spectrum(pos, z, modes_cart, polar_model, device, delta=0.01, cutoff=5.0, num_parts=4):
    n = pos.shape[0]
    activities = []
    for k in range(modes_cart.shape[1]):
        disp = modes_cart[:, k].reshape(n, 3)

        pos_plus = pos + delta * disp
        pos_minus = pos - delta * disp

        pred_plus = predict_single(pos_plus, z, polar_model, None, device,
                                     cutoff=cutoff, num_parts=num_parts)["polarizability"]
        pred_minus = predict_single(pos_minus, z, polar_model, None, device,
                                      cutoff=cutoff, num_parts=num_parts)["polarizability"]

        dalpha_dQ = (pred_plus - pred_minus) / (2.0 * delta)
        activities.append(raman_activity_from_invariants(dalpha_dQ))
    return np.array(activities)

def ir_spectrum(pos, z, modes_cart, dedipole_model, device, cutoff=5.0, num_parts=4):
    n = pos.shape[0]
    pred = predict_single(pos, z, None, dedipole_model, device,
                           cutoff=cutoff, num_parts=num_parts)["dipole_derivative"]  

    intensities = []
    for k in range(modes_cart.shape[1]):
        disp = modes_cart[:, k].reshape(n, 3)              
        dmu_dQ = np.einsum("nij,nj->i", pred, disp)          
        intensities.append(float(np.sum(dmu_dQ ** 2)))
    return np.array(intensities)

def load_reference_spectrum(h5_path, group_name, kind):
    freq_key = f"{kind}_freq_cm"
    val_key = "ir_intensity" if kind == "ir" else "raman_activity"
    with h5py.File(h5_path, "r") as f:
        if group_name not in f or freq_key not in f[group_name] or val_key not in f[group_name]:
            return None, None
        freqs = np.asarray(f[group_name][freq_key][:])
        vals = np.asarray(f[group_name][val_key][:])
    return freqs, vals

def match_by_frequency(pred_freq, pred_val, ref_freq, ref_val, tol_cm=25.0):
    matched_pred, matched_ref = [], []
    for rf, rv in zip(ref_freq, ref_val):
        diffs = np.abs(pred_freq - rf)
        j = np.argmin(diffs)
        if diffs[j] <= tol_cm:
            matched_pred.append(pred_val[j])
            matched_ref.append(rv)
    return np.array(matched_pred), np.array(matched_ref)

def plot_lsrl(ax, true, pred, title):
    slope, intercept = np.polyfit(true, pred, 1)
    fit = slope * true + intercept
    ss_res = np.sum((pred - fit) ** 2)
    ss_tot = np.sum((pred - pred.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(pred - true))

    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    xs = np.linspace(lo, hi, 100)
    ax.scatter(true, pred, s=14, alpha=0.5, color="#4C72B0", linewidths=0)
    ax.plot(xs, xs, "k--", linewidth=1, label="y = x")
    ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=1.5,
            label=f"LSRL (R²={r2:.3f})")
    ax.set_xlabel("QMe14S reference")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nMAE={mae:.4g}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return {"slope": slope, "intercept": intercept, "r2": r2, "mae": mae}

def compute_predicted_spectra_for_molecule(hessian_h5, group_name, polar_model,
                                             dedipole_model, device, delta=0.01,
                                             cutoff=5.0, num_parts=4,
                                             hessian_unit_factor=HARTREE_BOHR2_AMU_TO_CM1):
    with h5py.File(hessian_h5, "r") as f:
        group = f[group_name]
        pos = np.asarray(group["pos"][:], dtype=np.float64)
        z = np.asarray(group["z"][:], dtype=np.int64)
        hessian = np.asarray(group["hessian"][:], dtype=np.float64)

    freqs_cm, modes_cart = normal_modes_from_hessian(hessian, z, unit_factor=hessian_unit_factor)

    ir_int = (ir_spectrum(pos, z, modes_cart, dedipole_model, device, cutoff, num_parts)
              if dedipole_model is not None else None)
    raman_act = (raman_spectrum(pos, z, modes_cart, polar_model, device, delta, cutoff, num_parts)
                 if polar_model is not None else None)

    return freqs_cm, ir_int, raman_act

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hessian_h5", type=str, required=True,
                    help="Hessian_single_point.h5-style file with pos/z/hessian per group.")
    p.add_argument("--reference_h5", type=str, required=True,
                    help="QMe14S HDF5 with reference IR/Raman spectra -- see "
                         "load_reference_spectrum() docstring for the expected schema, "
                         "adjust field names there to match your actual file.")
    p.add_argument("--polar_checkpoint", type=str, default=None)
    p.add_argument("--dedipole_checkpoint", type=str, default=None)
    p.add_argument("--polar_model_type", type=str, default="inter", choices=["inter", "intra"])
    p.add_argument("--dedipole_model_type", type=str, default="inter", choices=["inter", "intra"])
    p.add_argument("--num_parts", type=int, default=4)
    p.add_argument("--max_degree", type=int, default=2)
    p.add_argument("--feature_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--partition_type", type=str, default="spectral")
    p.add_argument("--delta", type=float, default=0.01,
                    help="Finite-difference step (angstrom) for the Raman derivative.")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--freq_match_tol", type=float, default=25.0,
                    help="cm^-1 tolerance for matching predicted modes to reference peaks.")
    p.add_argument("--hessian_unit_factor", type=float, default=HARTREE_BOHR2_AMU_TO_CM1,
                    help="Unit-conversion constant applied to sqrt(eigenvalue) to get cm^-1. "
                         "Default assumes a Hessian in Hartree/bohr^2 with masses in amu; "
                         "override if your Hessian files use different units.")
    p.add_argument("--out_dir", type=str, default="./spectra_figures")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    polar_model = None
    if args.polar_checkpoint:
        polar_model = load_trained_model(args.polar_checkpoint, "polar", args.polar_model_type, args, device)
    dedipole_model = None
    if args.dedipole_checkpoint:
        dedipole_model = load_trained_model(args.dedipole_checkpoint, "dedipole", args.dedipole_model_type, args, device)

    if polar_model is None and dedipole_model is None:
        raise ValueError("Provide at least one of --polar_checkpoint / --dedipole_checkpoint.")

    with h5py.File(args.hessian_h5, "r") as f:
        group_names = list(f.keys())
    print(f"Found {len(group_names)} molecule(s) in {args.hessian_h5}")

    ir_pred_all, ir_ref_all = [], []
    raman_pred_all, raman_ref_all = [], []
    n_used = 0

    for group_name in group_names:
        try:
            freqs_cm, ir_int, raman_act = compute_predicted_spectra_for_molecule(
                args.hessian_h5, group_name, polar_model, dedipole_model, device,
                delta=args.delta, cutoff=args.cutoff, num_parts=args.num_parts,
                hessian_unit_factor=args.hessian_unit_factor,
            )
        except Exception as e:
            print(f"Skipping {group_name}: {e}")
            continue

        if ir_int is not None:
            ref_freq, ref_val = load_reference_spectrum(args.reference_h5, group_name, "ir")
            if ref_freq is not None:
                mp, mr = match_by_frequency(freqs_cm, ir_int, ref_freq, ref_val, args.freq_match_tol)
                ir_pred_all.append(mp)
                ir_ref_all.append(mr)

        if raman_act is not None:
            ref_freq, ref_val = load_reference_spectrum(args.reference_h5, group_name, "raman")
            if ref_freq is not None:
                mp, mr = match_by_frequency(freqs_cm, raman_act, ref_freq, ref_val, args.freq_match_tol)
                raman_pred_all.append(mp)
                raman_ref_all.append(mr)

        n_used += 1

    print(f"Computed spectra for {n_used}/{len(group_names)} molecule(s).")

    n_panels = int(bool(ir_pred_all)) + int(bool(raman_pred_all))
    if n_panels == 0:
        print("No reference spectra were found (check load_reference_spectrum's schema "
              "against your actual QMe14S file) -- nothing to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    ax_idx = 0

    if ir_pred_all:
        pred = np.concatenate(ir_pred_all)
        ref = np.concatenate(ir_ref_all)
        stats = plot_lsrl(axes[ax_idx], ref, pred, f"IR intensity (n={len(pred)} matched peaks)")
        print(f"IR:    R²={stats['r2']:.4f}  MAE={stats['mae']:.4g}")
        ax_idx += 1

    if raman_pred_all:
        pred = np.concatenate(raman_pred_all)
        ref = np.concatenate(raman_ref_all)
        stats = plot_lsrl(axes[ax_idx], ref, pred, f"Raman activity (n={len(pred)} matched peaks)")
        print(f"Raman: R²={stats['r2']:.4f}  MAE={stats['mae']:.4g}")

    fig.suptitle("Predicted vs. QMe14S reference spectra")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(args.out_dir, "ir_raman_lsrl.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()