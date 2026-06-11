import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def run_sweep(args):
    dataset_parent = Path(args.dataset_parent)
    results_parent = Path(args.results_parent)
    results_parent.mkdir(parents=True, exist_ok=True)

    datasets = sorted(path for path in dataset_parent.iterdir() if (path / "metadata.csv").exists())
    if not datasets:
        raise RuntimeError(f"No window datasets found in {dataset_parent}")

    for dataset_dir in datasets:
        output_dir = results_parent / dataset_dir.name
        summary_path = output_dir / "eegnet_summary.json"
        if summary_path.exists() and not args.force:
            print(f"Skipping {dataset_dir.name}: existing {summary_path}", flush=True)
            continue

        command = [
            sys.executable,
            "train_eegnet.py",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--split",
            args.split,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.cpu:
            command.append("--cpu")
        run_command(command)

    run_command(
        [
            sys.executable,
            "aggregate_eegnet_window_results.py",
            "--results-dir",
            str(results_parent),
            "--output",
            str(results_parent / "window_comparison_summary.csv"),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet across all window datasets in a directory.")
    parser.add_argument("--dataset-parent", default="event_epoch_window_grid_fine")
    parser.add_argument("--results-parent", default="eegnet_window_results_fine")
    parser.add_argument("--split", choices=["series", "random"], default="series")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
