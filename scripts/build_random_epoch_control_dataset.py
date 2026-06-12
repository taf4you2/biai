import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_event_spectrogram_dataset import prepare_raw


def resolve_source_column(metadata):
    if "source_edf" in metadata.columns:
        return "source_edf"
    if "edf_path" in metadata.columns:
        return "edf_path"
    raise ValueError("Reference metadata must contain source_edf or edf_path.")


def sample_random_epoch(raw, expected_samples, rng):
    max_start = raw.n_times - expected_samples
    if max_start <= 0:
        raise ValueError("Recording is shorter than the requested epoch length.")

    for _ in range(100):
        start = int(rng.integers(0, max_start + 1))
        stop = start + expected_samples
        epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="omit").astype(np.float32)
        if epoch.shape[1] == expected_samples:
            return start, stop, epoch

    raise RuntimeError("Could not sample a full random epoch after 100 attempts.")


def build_dataset(args):
    rng = np.random.default_rng(args.seed)
    reference_dir = Path(args.reference_dataset_dir)
    output_dir = Path(args.output_dir)
    epoch_root = output_dir / "epochs"
    epoch_root.mkdir(parents=True, exist_ok=True)
    for old_file in epoch_root.rglob("*.npz"):
        old_file.unlink()

    metadata = pd.read_csv(reference_dir / "metadata.csv")
    metadata = metadata[metadata["epoch_path"].notna()].copy()
    metadata = metadata[metadata["image_category"].notna()].copy()
    if args.max_rows:
        metadata = metadata.head(args.max_rows).copy()

    source_col = resolve_source_column(metadata)
    if "epoch_shape" in metadata.columns and metadata["epoch_shape"].notna().any():
        expected_samples = int(str(metadata["epoch_shape"].dropna().iloc[0]).split("x")[-1])
    else:
        if "tmin" not in metadata.columns or "tmax" not in metadata.columns:
            raise ValueError("Reference metadata must contain epoch_shape or tmin/tmax.")
        first_epoch_path = reference_dir / "epochs" / Path(metadata["epoch_path"].iloc[0]).name
        with np.load(first_epoch_path, allow_pickle=True) as data:
            expected_samples = int(data["epoch"].shape[1])

    all_rows = []
    session_summaries = []
    session_keys = [source_col]
    if "participant" in metadata.columns:
        session_keys.insert(0, "participant")

    for session_values, session_df in metadata.groupby(session_keys, dropna=False):
        if not isinstance(session_values, tuple):
            session_values = (session_values,)
        session_info = dict(zip(session_keys, session_values))
        source_edf = str(session_info[source_col])
        participant = str(session_info.get("participant", Path(source_edf).stem.replace("_raw", "")))
        participant_dir = epoch_root / participant
        participant_dir.mkdir(parents=True, exist_ok=True)

        impedance_csv = ""
        if "impedance_csv" in session_df.columns and session_df["impedance_csv"].notna().any():
            impedance_csv = str(session_df["impedance_csv"].dropna().iloc[0])

        raw = prepare_raw(
            source_edf,
            impedance_csv,
            args.reject_threshold,
            args.l_freq,
            args.h_freq,
            args.notch_freq,
        )
        sfreq = raw.info["sfreq"]
        times = np.arange(expected_samples, dtype=np.float32) / sfreq
        saved = 0

        for row_idx, (_, row) in enumerate(session_df.reset_index(drop=True).iterrows()):
            start, stop, epoch = sample_random_epoch(raw, expected_samples, rng)
            stem = f"{participant}_random_{row_idx:05d}"
            epoch_path = participant_dir / f"{stem}.npz"
            np.savez_compressed(
                epoch_path,
                epoch=epoch,
                times=times,
                channels=np.array(raw.ch_names),
                sfreq=np.array([sfreq], dtype=np.float32),
                random_start_sample=np.array([start], dtype=np.int64),
                random_stop_sample=np.array([stop], dtype=np.int64),
            )

            new_row = row.to_dict()
            new_row.update(
                {
                    "control_type": "random_epoch",
                    "reference_epoch_path": row["epoch_path"],
                    "epoch_path": str(epoch_path),
                    "random_start_sample": start,
                    "random_stop_sample": stop,
                    "tmin": 0.0,
                    "tmax": expected_samples / sfreq,
                    "epoch_shape": "x".join(str(dim) for dim in epoch.shape),
                    "source_edf": source_edf,
                }
            )
            if "participant" not in new_row:
                new_row["participant"] = participant
            all_rows.append(new_row)
            saved += 1

        session_summaries.append(
            {
                "participant": participant,
                "source_edf": source_edf,
                "saved_epochs": saved,
                "epoch_samples": expected_samples,
                "sfreq": sfreq,
            }
        )
        print(f"{participant}: saved {saved} random control epochs")

    pd.DataFrame(all_rows).to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(session_summaries).to_csv(output_dir / "session_summary.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved {len(all_rows)} random control epochs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build random-window EEG control epochs matched to an event epoch dataset.")
    parser.add_argument("--reference-dataset-dir", default="event_epoch_multisession_image_on_0_0p8")
    parser.add_argument("--output-dir", default="event_epoch_random_control")
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
