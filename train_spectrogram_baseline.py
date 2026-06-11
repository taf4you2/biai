import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["image_category"].notna()].copy()

    features = []
    labels = []
    kept_rows = []
    for _, row in metadata.iterrows():
        tensor_path = Path(row["tensor_path"])
        if not tensor_path.is_absolute():
            tensor_path = Path(row["tensor_path"])
        if not tensor_path.exists():
            tensor_path = dataset_dir / "tensors" / Path(row["tensor_path"]).name
        if not tensor_path.exists():
            continue

        with np.load(tensor_path, allow_pickle=True) as data:
            spectrogram = data["spectrogram"].astype(np.float32)

        features.append(spectrogram.ravel())
        labels.append(row["image_category"])
        kept_rows.append(row)

    if not features:
        raise RuntimeError(f"No tensors found for {metadata_path}")

    return np.vstack(features), np.array(labels), pd.DataFrame(kept_rows)


def run_baseline(args):
    x, y_text, metadata = load_dataset(args.dataset_dir)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    if args.split == "series" and "series_id" in metadata.columns:
        groups = metadata["series_id"].fillna(-1).astype(int).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_series = [int(series_id) for series_id in sorted(set(groups[test_idx]))]
        split_description = f"grouped by series_id; test series: {test_series}"
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        split_description = "random stratified"

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

    labels = label_encoder.classes_
    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {len(x)}")
    print(f"Features per sample: {x.shape[1]}")
    print(f"Classes: {len(labels)} -> {', '.join(labels)}")
    print(f"Train/test: {len(x_train)}/{len(x_test)}")
    print(f"Split: {split_description}")
    print()
    print(f"Dummy accuracy: {accuracy_score(y_test, dummy_pred):.4f}")
    print(f"Logistic regression accuracy: {accuracy_score(y_test, pred):.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, pred, target_names=labels, zero_division=0))
    print("Confusion matrix:")
    matrix = confusion_matrix(y_test, pred)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    print(matrix_df.to_string())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "spectrogram_baseline_summary.txt"
    matrix_path = output_dir / "spectrogram_baseline_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"Samples: {len(x)}\n")
        f.write(f"Features per sample: {x.shape[1]}\n")
        f.write(f"Classes: {len(labels)} -> {', '.join(labels)}\n")
        f.write(f"Train/test: {len(x_train)}/{len(x_test)}\n")
        f.write(f"Split: {split_description}\n")
        f.write(f"Dummy accuracy: {accuracy_score(y_test, dummy_pred):.4f}\n")
        f.write(f"Logistic regression accuracy: {accuracy_score(y_test, pred):.4f}\n\n")
        f.write(classification_report(y_test, pred, target_names=labels, zero_division=0))
        f.write("\nConfusion matrix:\n")
        f.write(matrix_df.to_string())
        f.write("\n")
    print()
    print(f"Saved summary to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple baseline on trigger-aligned EEG spectrograms.")
    parser.add_argument("--dataset-dir", default="event_spectrogram_dataset")
    parser.add_argument("--output-dir", default="baseline_results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split", choices=["random", "series"], default="series")
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
