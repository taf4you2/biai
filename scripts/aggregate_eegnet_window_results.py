import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate(results_dir):
    rows = []
    for summary_path in sorted(Path(results_dir).glob("*/eegnet_summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        result_dir = summary_path.parent
        rows.append(
            {
                "window_name": result_dir.name,
                "dataset": summary["dataset"],
                "samples": summary["samples"],
                "train_samples": summary["train_samples"],
                "validation_samples": summary.get("validation_samples", ""),
                "test_samples": summary["test_samples"],
                "input_shape": "x".join(str(dim) for dim in summary["input_shape"]),
                "best_epoch": summary.get("best_epoch", ""),
                "best_validation_accuracy": summary.get("best_validation_accuracy", ""),
                "best_test_accuracy": summary.get("best_test_accuracy", summary["final_test_accuracy"]),
                "final_test_accuracy": summary["final_test_accuracy"],
                "label_control": summary.get("label_control", "none"),
                "split": summary["split"],
                "validation_split": summary.get("validation_split", ""),
                "test_selected_by": summary.get("test_selected_by", "test_accuracy"),
            }
        )
    return pd.DataFrame(rows).sort_values("final_test_accuracy", ascending=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate EEGNet window-comparison summaries.")
    parser.add_argument("--results-dir", default="eegnet_window_results")
    parser.add_argument("--output", default="eegnet_window_results/window_comparison_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = aggregate(args.results_dir)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {len(results)} rows to {args.output}")
