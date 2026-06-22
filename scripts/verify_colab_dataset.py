import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


def portable_epoch_path(raw_path, dataset_dir):
    dataset_dir = Path(dataset_dir)
    normalized = str(raw_path).replace("\\", "/")
    parts = list(PurePosixPath(normalized).parts)
    if "epochs" not in parts:
        return None
    return dataset_dir.joinpath(*parts[parts.index("epochs") :])


def verify(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Brak pliku: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"epoch_path", "image_category", "participant"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"Brak wymaganych kolumn: {missing_columns}")

    missing_paths = []
    resolved_paths = []
    for raw_path in metadata["epoch_path"]:
        resolved = portable_epoch_path(raw_path, dataset_dir)
        if resolved is None or not resolved.is_file():
            missing_paths.append(str(raw_path))
        else:
            resolved_paths.append(resolved)

    if missing_paths:
        examples = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"Nie znaleziono {len(missing_paths)} plików epok. Przykłady:\n{examples}"
        )

    sample_shapes = []
    for epoch_path in resolved_paths[: min(5, len(resolved_paths))]:
        with np.load(epoch_path, allow_pickle=True) as data:
            if "epoch" not in data:
                raise KeyError(f"Brak tablicy 'epoch' w {epoch_path}")
            sample_shapes.append(list(data["epoch"].shape))

    summary = {
        "dataset_dir": str(dataset_dir),
        "metadata_rows": int(len(metadata)),
        "epoch_files_found": int(len(resolved_paths)),
        "participants": sorted(metadata["participant"].dropna().astype(str).unique().tolist()),
        "categories": sorted(metadata["image_category"].dropna().astype(str).unique().tolist()),
        "sample_shapes": sample_shapes,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a portable EEG epoch dataset.")
    parser.add_argument("--dataset-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args.dataset_dir)
