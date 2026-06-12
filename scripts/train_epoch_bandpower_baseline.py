import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_eegnet import resolve_epoch_path, split_metadata, truthy_series


BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
]


def load_metadata(dataset_dir, only_qc_accepted=False):
    metadata = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    metadata = metadata[metadata["image_category"].notna()].copy()
    metadata = metadata[metadata["epoch_path"].notna()].copy()
    if only_qc_accepted:
        if "qc_accepted" not in metadata.columns:
            raise ValueError("--only-qc-accepted requires a metadata.csv column named 'qc_accepted'")
        metadata = metadata[truthy_series(metadata["qc_accepted"])].copy()
    return metadata


def compute_bandpower_features(epoch, sfreq):
    epoch = np.nan_to_num(epoch.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    freqs = np.fft.rfftfreq(epoch.shape[1], d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(epoch, axis=1)) ** 2 / max(1, epoch.shape[1])

    features = []
    for _, fmin, fmax in BANDS:
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            band_power = np.zeros(epoch.shape[0], dtype=np.float32)
        else:
            band_power = power[:, mask].mean(axis=1)
        features.append(np.log10(band_power + 1e-20))
    return np.concatenate(features).astype(np.float32)


def load_feature_matrix(metadata, dataset_dir):
    features = []
    for _, row in metadata.iterrows():
        epoch_path = resolve_epoch_path(row["epoch_path"], dataset_dir)
        with np.load(epoch_path, allow_pickle=True) as data:
            epoch = data["epoch"].astype(np.float32)
            sfreq = float(data["sfreq"][0])
        features.append(compute_bandpower_features(epoch, sfreq))

    if not features:
        raise RuntimeError(f"No epochs found under {dataset_dir}")
    return np.vstack(features)


def write_report(output_dir, labels, y_test, pred, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_ids = np.arange(len(labels))
    matrix = confusion_matrix(y_test, pred, labels=label_ids)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    with (output_dir / "bandpower_baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    matrix_df.to_csv(output_dir / "bandpower_baseline_confusion_matrix.csv")
    with (output_dir / "bandpower_baseline_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {summary['dataset']}\n")
        f.write(f"Samples: {summary['samples']}\n")
        f.write(f"Train/test: {summary['train_samples']}/{summary['test_samples']}\n")
        f.write(f"Features per sample: {summary['features_per_sample']}\n")
        f.write(f"Split: {summary['split']}\n")
        f.write(f"Only QC accepted: {summary['only_qc_accepted']}\n")
        f.write(f"Dummy accuracy: {summary['dummy_accuracy']:.4f}\n")
        f.write(f"Logistic regression accuracy: {summary['logistic_accuracy']:.4f}\n\n")
        f.write(classification_report(y_test, pred, labels=label_ids, target_names=labels, zero_division=0))
        f.write("\nConfusion matrix:\n")
        f.write(matrix_df.to_string())
        f.write("\n")


def run_baseline(args):
    metadata = load_metadata(args.dataset_dir, args.only_qc_accepted)
    train_df, test_df, split_description = split_metadata(metadata, args)

    label_encoder = LabelEncoder()
    label_encoder.fit(metadata["image_category"])
    labels = label_encoder.classes_

    x_train = load_feature_matrix(train_df, args.dataset_dir)
    x_test = load_feature_matrix(test_df, args.dataset_dir)
    y_train = label_encoder.transform(train_df["image_category"])
    y_test = label_encoder.transform(test_df["image_category"])

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            class_weight="balanced",
            max_iter=args.max_iter,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    summary = {
        "dataset": args.dataset_dir,
        "samples": int(len(metadata)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "features_per_sample": int(x_train.shape[1]),
        "bands": [name for name, _, _ in BANDS],
        "classes": labels.tolist(),
        "split": split_description,
        "only_qc_accepted": bool(args.only_qc_accepted),
        "dummy_accuracy": float(accuracy_score(y_test, dummy_pred)),
        "logistic_accuracy": float(accuracy_score(y_test, pred)),
        "c": float(args.c),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
    }
    write_report(args.output_dir, labels, y_test, pred, summary)

    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {len(metadata)}")
    print(f"Features per sample: {x_train.shape[1]}")
    print(f"Classes: {len(labels)} -> {', '.join(labels)}")
    print(f"Train/test: {len(train_df)}/{len(test_df)}")
    print(f"Split: {split_description}")
    print(f"Only QC accepted: {args.only_qc_accepted}")
    print()
    print(f"Dummy accuracy: {summary['dummy_accuracy']:.4f}")
    print(f"Logistic regression accuracy: {summary['logistic_accuracy']:.4f}")
    print(f"Saved reports to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a bandpower baseline on trigger-aligned raw EEG epochs.")
    parser.add_argument("--dataset-dir", default="event_epoch_multisession_image_on_0_0p8")
    parser.add_argument("--output-dir", default="baseline_epoch_bandpower_results")
    parser.add_argument("--split", choices=["series", "random", "participant", "image", "participant_image"], default="series")
    parser.add_argument("--test-participant", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--only-qc-accepted", action="store_true")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
