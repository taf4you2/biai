import argparse
import json
from pathlib import Path

import pandas as pd


def extract_test_participant(split_description, result_dir_name):
    marker = "test participant:"
    if marker in split_description:
        return split_description.split(marker, 1)[1].strip()
    if result_dir_name.startswith("test_"):
        return result_dir_name[5:]
    return result_dir_name


def aggregate(results_dir):
    rows = []
    for summary_path in sorted(Path(results_dir).glob("test_*/eegnet_summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        result_dir = summary_path.parent
        participant = extract_test_participant(summary["split"], result_dir.name)
        rows.append(
            {
                "test_participant": participant,
                "samples": summary["samples"],
                "train_samples": summary["train_samples"],
                "validation_samples": summary.get("validation_samples", ""),
                "test_samples": summary["test_samples"],
                "input_shape": "x".join(str(dim) for dim in summary["input_shape"]),
                "best_epoch": summary.get("best_epoch", ""),
                "best_validation_accuracy": summary.get("best_validation_accuracy", ""),
                "best_test_accuracy": summary.get("best_test_accuracy", summary["final_test_accuracy"]),
                "final_test_accuracy": summary["final_test_accuracy"],
                "normalization": summary.get("normalization", ""),
                "balanced_sampler": summary.get("balanced_sampler", ""),
                "label_control": summary.get("label_control", "none"),
                "split": summary["split"],
                "validation_split": summary.get("validation_split", ""),
                "test_selected_by": summary.get("test_selected_by", "test_accuracy"),
            }
        )

    results = pd.DataFrame(rows).sort_values("test_participant")
    if not results.empty:
        mean_row = {
            "test_participant": "MEAN",
            "samples": results["samples"].iloc[0],
            "train_samples": "",
            "validation_samples": "",
            "test_samples": "",
            "input_shape": results["input_shape"].iloc[0],
            "best_epoch": "",
            "best_validation_accuracy": pd.to_numeric(results["best_validation_accuracy"], errors="coerce").mean(),
            "best_test_accuracy": results["best_test_accuracy"].mean(),
            "final_test_accuracy": results["final_test_accuracy"].mean(),
            "normalization": ",".join(sorted(set(str(value) for value in results["normalization"]))),
            "balanced_sampler": ",".join(sorted(set(str(value) for value in results["balanced_sampler"]))),
            "label_control": ",".join(sorted(set(str(value) for value in results["label_control"]))),
            "split": "mean over participants",
            "validation_split": ",".join(sorted(set(str(value) for value in results["validation_split"]))),
            "test_selected_by": ",".join(sorted(set(str(value) for value in results["test_selected_by"]))),
        }
        results = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate EEGNet leave-one-participant summaries.")
    parser.add_argument("--results-dir", default="eegnet_multisession_results")
    parser.add_argument("--output", default="eegnet_multisession_results/participant_loo_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = aggregate(args.results_dir)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {len(results)} rows to {args.output}")
