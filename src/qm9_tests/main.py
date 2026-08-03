import argparse
import json
import os
import subprocess
import sys

DEFAULT_TARGETS = [1, 4, 3, 2, 0, 11]  # alpha, gap, homo, lumo, mu, Cv
TARGET_NAMES = {
    0: "mu", 1: "alpha", 2: "homo", 3: "lumo", 4: "gap",
    5: "R2", 6: "zpve", 7: "U0", 8: "U", 9: "H", 10: "G", 11: "Cv",
}


def run(cmd, dry_run):
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def train_se3_cross(target, args, dry_run):
    """src/qm9_tests/train.py already names checkpoints
    best_model_trial{trial}_target{target}.pt internally, so we only need
    to give it its own --metrics_dir per target to keep runs isolated."""
    out_dir = os.path.join(args.out_root, "se3-cross", f"target_{target}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.qm9_tests.train",
        "--target", str(target),
        "--trials", str(args.trials),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.se3_cross_batch_size),
        "--accum_steps", str(args.se3_cross_accum_steps),
        "--num_parts", str(args.num_parts),
        "--data_root", args.data_root,
        "--metrics_dir", out_dir,
        "--device", args.device,
    ]
    if args.dummy_data:
        cmd += ["--dummy_data", "--dummy_batches", str(args.dummy_batches)]
    run(cmd, dry_run)
    return out_dir


def train_se3_trans(target, args, dry_run):
    """src/qm9_tests/train_se3.py takes --checkpoint_dir/--checkpoint_template as
    args, so we build a target-qualified template ourselves."""
    out_dir = os.path.join(args.out_root, "se3-trans", f"target_{target}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_template = f"se3_trans_trial{{trial}}_target{target}.pt"
    cmd = [
        sys.executable, "-m", "src.qm9_tests.train_se3",
        "--target", str(target),
        "--trials", str(args.trials),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.se3_trans_batch_size),
        "--accum_steps", str(args.se3_trans_accum_steps),
        "--min_nodes", str(args.num_parts),
        "--data_root", args.data_root,
        "--checkpoint_dir", out_dir,
        "--checkpoint_template", checkpoint_template,
        "--metrics_dir", out_dir,
        "--device", args.device,
    ]
    if args.dummy_data:
        cmd += ["--dummy_data", "--dummy_batches", str(args.dummy_batches)]
    run(cmd, dry_run)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS,
                         help="QM9 target indices to train on.")
    parser.add_argument("--models", type=str, nargs="+", default=["se3-cross", "se3-trans"],
                         choices=["se3-cross", "se3-trans"])
    parser.add_argument("--trials", type=int, default=5,
                         help="Number of trials (checkpoints) per (model, target). "
                              "Keep at 5 to match the 95%% CI sample size used downstream.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_parts", type=int, default=4,
                         help="Passed as --num_parts to the se3_cross model and --min_nodes "
                              "to the se3_trans model, so both are trained/evaluated on the "
                              "same set of molecules.")
    parser.add_argument("--se3_cross_batch_size", type=int, default=32)
    parser.add_argument("--se3_cross_accum_steps", type=int, default=8)
    parser.add_argument("--se3_trans_batch_size", type=int, default=32)
    parser.add_argument("--se3_trans_accum_steps", type=int, default=8)
    parser.add_argument("--data_root", type=str, default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--out_root", type=str, default="./qm9_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print the commands without running them.")
    parser.add_argument("--dummy_data", action="store_true",
                         help="Use synthetic in-memory molecules instead of load_qm9 in "
                              "both training scripts, for smoke-testing the whole pipeline "
                              "without real QM9 data on disk.")
    parser.add_argument("--dummy_batches", type=int, default=6,
                         help="Number of training batches to synthesize when --dummy_data "
                              "is set (val/test get roughly a third of this each).")
    args = parser.parse_args()

    print(f"Targets:  {args.targets}  ({[TARGET_NAMES.get(t, '?') for t in args.targets]})")
    print(f"Models:   {args.models}")
    print(f"Trials:   {args.trials}  |  Epochs: {args.epochs}")

    manifest = []
    for target in args.targets:
        tname = TARGET_NAMES.get(target, str(target))
        print(f"\n=== Target {target} ({tname}) ===")
        if "se3-cross" in args.models:
            out_dir = train_se3_cross(target, args, args.dry_run)
            manifest.append({"model": "se3-cross", "target": target, "target_name": tname, "dir": out_dir})
        if "se3-trans" in args.models:
            out_dir = train_se3_trans(target, args, args.dry_run)
            manifest.append({"model": "se3-trans", "target": target, "target_name": tname, "dir": out_dir})

    os.makedirs(args.out_root, exist_ok=True)
    manifest_path = os.path.join(args.out_root, "train_manifest.json")
    if not args.dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nWrote manifest of all runs to {manifest_path}")


if __name__ == "__main__":
    main()