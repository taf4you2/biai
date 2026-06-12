import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_event_spectrogram_dataset import (
    attach_event_metadata,
    events_from_annotations,
    prepare_raw,
    write_event_alignment_qc,
)


def make_stem(row_idx, event_code, row):
    trial_id = row.get("trial_id")
    image_category = row.get("image_category")
    stem = f"{row_idx:05d}_code{event_code}"
    if pd.notna(trial_id):
        stem += f"_trial{int(trial_id):04d}"
    if isinstance(image_category, str) and image_category:
        stem += f"_{image_category}"
    return stem


def build_dataset(args):
    output_dir = Path(args.output_dir)
    epoch_dir = output_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for old_file in epoch_dir.glob("*.npz"):
        old_file.unlink()

    raw = prepare_raw(
        args.edf,
        args.impedance_csv,
        args.reject_threshold,
        args.l_freq,
        args.h_freq,
        args.notch_freq,
    )

    annotation_events = events_from_annotations(raw)
    events = attach_event_metadata(annotation_events, args.events_csv)
    write_event_alignment_qc(
        annotation_events,
        args.events_csv,
        events,
        output_dir / "event_alignment_qc.csv",
        args.event_code,
        strict=args.strict_event_qc,
    )
    target_events = events[events["event_code"] == args.event_code].copy()
    if args.max_trials:
        target_events = target_events.head(args.max_trials)

    sfreq = raw.info["sfreq"]
    start_offset = int(round(args.tmin * sfreq))
    stop_offset = int(round(args.tmax * sfreq))
    expected_samples = stop_offset - start_offset
    times = np.arange(expected_samples, dtype=np.float32) / sfreq + args.tmin
    metadata_rows = []

    for row_idx, row in target_events.reset_index(drop=True).iterrows():
        start = int(row["sample"]) + start_offset
        stop = int(row["sample"]) + stop_offset
        if start < 0 or stop > raw.n_times or stop <= start:
            continue

        epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="omit").astype(np.float32)
        if epoch.shape[1] != expected_samples:
            continue

        stem = make_stem(row_idx, args.event_code, row)
        epoch_path = epoch_dir / f"{stem}.npz"
        np.savez_compressed(
            epoch_path,
            epoch=epoch,
            times=times,
            channels=np.array(raw.ch_names),
            sfreq=np.array([sfreq], dtype=np.float32),
        )

        metadata = row.to_dict()
        metadata.update(
            {
                "epoch_path": str(epoch_path),
                "tmin": args.tmin,
                "tmax": args.tmax,
                "epoch_shape": "x".join(str(dim) for dim in epoch.shape),
            }
        )
        metadata_rows.append(metadata)

    pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved {len(metadata_rows)} event epochs to {output_dir}")
    print(f"Epoch shape example: channels x samples = {metadata_rows[0]['epoch_shape'] if metadata_rows else 'n/a'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build trigger-aligned raw EEG epoch tensors.")
    parser.add_argument("--edf", default="dane/mole_/mole_0004_raw.edf")
    parser.add_argument("--events-csv", default="dane/mole_/mole_EEGBasedVisualRecall_Events_Rep2_2026-05-22_11-27-28.csv")
    parser.add_argument("--impedance-csv", default="dane/mole_/mole_0004_imp.csv")
    parser.add_argument("--output-dir", default="event_epoch_dataset")
    parser.add_argument("--event-code", type=int, default=12, help="Default 12 = IMAGE_ON.")
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.5)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--strict-event-qc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
