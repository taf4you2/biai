import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(command):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def run_loo(args):
    metadata = pd.read_csv(Path(args.dataset_dir) / "metadata.csv")
    participants = sorted(str(value) for value in metadata["participant"].dropna().unique())
    results_parent = Path(args.results_parent)
    results_parent.mkdir(parents=True, exist_ok=True)

    for participant in participants:
        output_dir = results_parent / f"test_{participant}"
        summary_path = output_dir / "eegnet_summary.json"
        if summary_path.exists() and not args.force:
            print(f"Skipping {participant}: existing {summary_path}", flush=True)
            continue

        command = [
            sys.executable,
            str(SCRIPT_DIR / "train_eegnet.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--output-dir",
            str(output_dir),
            "--split",
            "participant",
            "--test-participant",
            participant,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--val-size",
            str(args.val_size),
            "--val-split",
            args.val_split,
            "--label-control",
            args.label_control,
            "--normalization",
            args.normalization,
            "--balanced-sampler",
            args.balanced_sampler,
        ]
        if args.only_qc_accepted:
            command.append("--only-qc-accepted")
        if args.cpu:
            command.append("--cpu")
        if args.no_preload:
            command.append("--no-preload")
        run_command(command)

    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "aggregate_participant_loo_results.py"),
            "--results-dir",
            str(results_parent),
            "--output",
            str(results_parent / "participant_loo_summary.csv"),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run EEGNet leave-one-participant evaluation.")
    parser.add_argument("--dataset-dir", default="event_epoch_multisession_image_on_0_0p8")
    parser.add_argument("--results-parent", default="eegnet_multisession_results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--val-split", choices=["auto", "random", "series", "participant"], default="auto")
    parser.add_argument("--label-control", choices=["none", "permute"], default="none")
    parser.add_argument("--normalization", choices=["global", "participant", "epoch"], default="global")
    parser.add_argument("--balanced-sampler", choices=["none", "category", "participant", "category_participant"], default="none")
    parser.add_argument("--only-qc-accepted", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_loo(parse_args())
