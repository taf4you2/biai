import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_event_epoch_dataset import add_epoch_qc_args, compute_epoch_qc, make_stem, write_epoch_qc_reports
from build_event_spectrogram_dataset import (
    attach_event_metadata,
    events_from_annotations,
    prepare_raw,
    write_event_alignment_qc,
)


INITIAL_WINDOW_SPECS = [
    {"name": "image_on_pre_m08_0", "event_code": 12, "tmin": -0.8, "tmax": 0.0},
    {"name": "image_on_0_0p5", "event_code": 12, "tmin": 0.0, "tmax": 0.5},
    {"name": "image_on_0_0p8", "event_code": 12, "tmin": 0.0, "tmax": 0.8},
    {"name": "image_on_m02_1p0", "event_code": 12, "tmin": -0.2, "tmax": 1.0},
    {"name": "image_on_m05_1p5", "event_code": 12, "tmin": -0.5, "tmax": 1.5},
]

FINE_WINDOW_SPECS = [
    {"name": "image_on_0_0p6", "event_code": 12, "tmin": 0.0, "tmax": 0.6},
    {"name": "image_on_0_0p7", "event_code": 12, "tmin": 0.0, "tmax": 0.7},
    {"name": "image_on_0_0p8", "event_code": 12, "tmin": 0.0, "tmax": 0.8},
    {"name": "image_on_0_0p9", "event_code": 12, "tmin": 0.0, "tmax": 0.9},
    {"name": "image_on_0_1p0", "event_code": 12, "tmin": 0.0, "tmax": 1.0},
    {"name": "image_on_0p1_0p8", "event_code": 12, "tmin": 0.1, "tmax": 0.8},
    {"name": "image_on_0p1_0p9", "event_code": 12, "tmin": 0.1, "tmax": 0.9},
    {"name": "image_on_0p2_0p8", "event_code": 12, "tmin": 0.2, "tmax": 0.8},
]

WINDOW_PRESETS = {
    "initial": INITIAL_WINDOW_SPECS,
    "fine": FINE_WINDOW_SPECS,
}


def write_window_dataset(raw, annotation_events, events, spec, output_parent, args):
    output_dir = Path(output_parent) / spec["name"]
    epoch_dir = output_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for old_file in epoch_dir.glob("*.npz"):
        old_file.unlink()

    write_event_alignment_qc(
        annotation_events,
        args.events_csv,
        events,
        output_dir / "event_alignment_qc.csv",
        spec["event_code"],
        strict=args.strict_event_qc,
    )

    target_events = events[events["event_code"] == spec["event_code"]].copy()
    if args.max_trials:
        target_events = target_events.head(args.max_trials)

    sfreq = raw.info["sfreq"]
    start_offset = int(round(spec["tmin"] * sfreq))
    stop_offset = int(round(spec["tmax"] * sfreq))
    expected_samples = stop_offset - start_offset
    times = np.arange(expected_samples, dtype=np.float32) / sfreq + spec["tmin"]
    metadata_rows = []
    qc_rows = []

    for row_idx, row in target_events.reset_index(drop=True).iterrows():
        start = int(row["sample"]) + start_offset
        stop = int(row["sample"]) + stop_offset
        if start < 0 or stop > raw.n_times or stop <= start:
            continue

        epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="omit").astype(np.float32)
        if epoch.shape[1] != expected_samples:
            continue

        qc = compute_epoch_qc(epoch, raw.ch_names, args)
        stem = make_stem(row_idx, spec["event_code"], row)
        epoch_path = epoch_dir / f"{stem}.npz"
        metadata = row.to_dict()
        metadata.update(
            {
                "window_name": spec["name"],
                "tmin": spec["tmin"],
                "tmax": spec["tmax"],
                "epoch_shape": "x".join(str(dim) for dim in epoch.shape),
            }
        )
        metadata.update(qc)
        qc_metadata = metadata.copy()
        qc_metadata.update({"epoch_path": str(epoch_path), "qc_saved": False})

        if args.drop_rejected and not qc["qc_accepted"]:
            qc_rows.append(qc_metadata)
            continue

        np.savez_compressed(
            epoch_path,
            epoch=epoch,
            times=times,
            channels=np.array(raw.ch_names),
            sfreq=np.array([sfreq], dtype=np.float32),
        )

        metadata.update({"epoch_path": str(epoch_path), "qc_saved": True})
        metadata_rows.append(metadata)
        qc_metadata.update({"qc_saved": True})
        qc_rows.append(qc_metadata)

    pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
    write_epoch_qc_reports(qc_rows, output_dir)
    config = vars(args).copy()
    config.update(spec)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    shape = metadata_rows[0]["epoch_shape"] if metadata_rows else "n/a"
    rejected = sum(not row["qc_accepted"] for row in qc_rows)
    print(f"{spec['name']}: saved {len(metadata_rows)} epochs, shape {shape}, QC rejected {rejected}/{len(qc_rows)}")
    return {
        "window_name": spec["name"],
        "samples": len(metadata_rows),
        "qc_candidate_epochs": len(qc_rows),
        "qc_rejected_epochs": rejected,
        "epoch_shape": shape,
        **spec,
    }


def build_grid(args):
    output_parent = Path(args.output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)

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

    rows = []
    specs = WINDOW_PRESETS[args.preset]
    if args.only:
        selected = set(args.only)
        specs = [spec for spec in specs if spec["name"] in selected]
        missing = selected - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"Unknown window names for preset {args.preset}: {sorted(missing)}")

    for spec in specs:
        rows.append(write_window_dataset(raw, annotation_events, events, spec, output_parent, args))

    pd.DataFrame(rows).to_csv(output_parent / "window_grid_manifest.csv", index=False)
    print(f"Saved window grid manifest to {output_parent / 'window_grid_manifest.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build five trigger-aligned epoch datasets for window comparison.")
    parser.add_argument("--edf", default="dane/Wyniki/mole_0004_raw.edf")
    parser.add_argument("--events-csv", default="dane/Wyniki/mole_EEGBasedVisualRecall_Events_Rep2_2026-05-22_11-27-28.csv")
    parser.add_argument("--impedance-csv", default="dane/Wyniki/mole_0004_imp.csv")
    parser.add_argument("--output-parent", default="event_epoch_window_grid")
    parser.add_argument("--preset", choices=sorted(WINDOW_PRESETS), default="initial")
    parser.add_argument("--only", nargs="*", default=None, help="Optional window names from the chosen preset.")
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--strict-event-qc", action="store_true")
    add_epoch_qc_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    build_grid(parse_args())
