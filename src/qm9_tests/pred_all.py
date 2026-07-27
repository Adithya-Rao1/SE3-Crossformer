import argparse
import os
import subprocess
import sys

DEFAULT_TARGETS = [1, 4, 3, 2, 0, 11]
TARGET_NAMES = {
    0: "mu", 1: "alpha", 2: "homo", 3: "lumo", 4: "gap",
    5: "R2", 6: "zpve", 7: "U0", 8: "U", 9: "H", 10: "G", 11: "Cv",
}


def run(cmd, dry_run):
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def predict_se3_cross(target, args, dry_run):
    ckpt_dir = os.path.join(args.ckpt_root, "se3-cross", f"target_{target}")
    out_dir = os.path.join(args.out_root, "se3-cross", f"target_{target}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.tests.pred_custom",
        "--target", str(target),
        "--trials", str(args.trials),
        "--num_parts", str(args.num_parts),
        "--checkpoint_dir", ckpt_dir,
        "--data_root", args.data_root,
        "--out_dir", out_dir,
        "--device", args.device,
    ]
    run(cmd, dry_run)
    return out_dir


def predict_se3_trans(target, args, dry_run):
    ckpt_dir = os.path.join(args.ckpt_root, "se3-trans", f"target_{target}")
    out_dir = os.path.join(args.out_root, "se3-trans", f"target_{target}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.tests.pred_se3_orig",
        "--target", str(target),
        "--trials", str(args.trials),
        "--min_nodes", str(args.num_parts),
        "--checkpoint_dir", ckpt_dir,
        "--data_root", args.data_root,
        "--out_dir", out_dir,
        "--device", args.device,
    ]
    run(cmd, dry_run)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--models", type=str, nargs="+", default=["se3-cross", "se3-trans"],
                         choices=["se3-cross", "se3-trans"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--num_parts", type=int, default=4,
                         help="Must match what train_all_targets.py used, so both "
                              "models are evaluated on the same set of molecules.")
    parser.add_argument("--data_root", type=str, default="/home/ubuntu/se3-crossformer-data/data")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints",
                         help="Root passed as --out_root to train_all_targets.py.")
    parser.add_argument("--out_root", type=str, default="./predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for target in args.targets:
        tname = TARGET_NAMES.get(target, str(target))
        print(f"\n=== Target {target} ({tname}) ===")
        if "se3-cross" in args.models:
            predict_se3_cross(target, args, args.dry_run)
        if "se3-trans" in args.models:
            predict_se3_trans(target, args, args.dry_run)

    print(f"\nDone. Per-target/per-model summaries are under {args.out_root}/<model>/target_<t>/summary.json")
    print("Run aggregate_results.py to build the cross-target comparison CSV and plot.")


if __name__ == "__main__":
    main()