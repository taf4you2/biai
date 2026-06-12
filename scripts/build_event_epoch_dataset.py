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


def add_epoch_qc_args(parser):
    parser.add_argument("--epoch-reject-ptp-uv", type=float, default=150.0)
    parser.add_argument("--epoch-reject-max-abs-uv", type=float, default=None)
    parser.add_argument("--flatline-ptp-uv", type=float, default=0.5)
    parser.add_argument("--drop-rejected", action="store_true")


def compute_epoch_qc(epoch, channels, args):
    finite_mask = np.isfinite(epoch)
    nonfinite_count = int((~finite_mask).sum())
    safe_epoch = np.nan_to_num(epoch, nan=0.0, posinf=0.0, neginf=0.0)
    epoch_uv = safe_epoch * 1e6

    if epoch_uv.size:
        ptp_by_channel = np.ptp(epoch_uv, axis=1)
        max_abs_by_channel = np.max(np.abs(epoch_uv), axis=1)
    else:
        ptp_by_channel = np.array([], dtype=np.float32)
        max_abs_by_channel = np.array([], dtype=np.float32)

    flat_mask = ptp_by_channel <= args.flatline_ptp_uv
    flat_channel_count = int(flat_mask.sum())
    flat_channels = [str(channels[idx]) for idx in np.flatnonzero(flat_mask)]

    ptp_max = float(ptp_by_channel.max()) if ptp_by_channel.size else 0.0
    ptp_mean = float(ptp_by_channel.mean()) if ptp_by_channel.size else 0.0
    max_abs = float(max_abs_by_channel.max()) if max_abs_by_channel.size else 0.0

    reject_reasons = []
    if nonfinite_count:
        reject_reasons.append("nonfinite")
    if args.epoch_reject_ptp_uv is not None and ptp_max > args.epoch_reject_ptp_uv:
        reject_reasons.append("ptp")
    if args.epoch_reject_max_abs_uv is not None and max_abs > args.epoch_reject_max_abs_uv:
        reject_reasons.append("max_abs")
    if flat_channel_count == len(channels) and len(channels) > 0:
        reject_reasons.append("flatline_all")

    return {
        "qc_accepted": not reject_reasons,
        "qc_reject_reason": ";".join(reject_reasons) if reject_reasons else "none",
        "qc_nonfinite_count": nonfinite_count,
        "qc_ptp_max_uv": ptp_max,
        "qc_ptp_mean_uv": ptp_mean,
        "qc_max_abs_uv": max_abs,
        "qc_flat_channel_count": flat_channel_count,
        "qc_flat_channels": ";".join(flat_channels),
    }


def write_epoch_qc_reports(qc_rows, output_dir):
    output_dir = Path(output_dir)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(output_dir / "epoch_qc.csv", index=False)

    if qc.empty:
        pd.DataFrame(
            [
                {
                    "group": "all",
                    "candidate_epochs": 0,
                    "saved_epochs": 0,
                    "accepted_epochs": 0,
                    "rejected_epochs": 0,
                    "rejection_rate": 0.0,
                }
            ]
        ).to_csv(output_dir / "epoch_qc_summary.csv", index=False)
        return

    qc["qc_accepted"] = qc["qc_accepted"].astype(bool)
    qc["qc_saved"] = qc["qc_saved"].astype(bool)
    summaries = []

    def add_summary(group_name, group):
        rejected = ~group["qc_accepted"]
        candidate_count = int(len(group))
        rejected_count = int(rejected.sum())
        summaries.append(
            {
                "group": group_name,
                "candidate_epochs": candidate_count,
                "saved_epochs": int(group["qc_saved"].sum()),
                "accepted_epochs": int(group["qc_accepted"].sum()),
                "rejected_epochs": rejected_count,
                "rejection_rate": rejected_count / candidate_count if candidate_count else 0.0,
                "nonfinite_epochs": int((group["qc_nonfinite_count"] > 0).sum()),
                "ptp_rejected_epochs": int(group["qc_reject_reason"].fillna("").str.contains("ptp").sum()),
                "max_abs_rejected_epochs": int(group["qc_reject_reason"].fillna("").str.contains("max_abs").sum()),
                "flatline_all_epochs": int(group["qc_reject_reason"].fillna("").str.contains("flatline_all").sum()),
                "qc_ptp_max_uv_median": float(group["qc_ptp_max_uv"].median()),
                "qc_ptp_max_uv_p95": float(group["qc_ptp_max_uv"].quantile(0.95)),
                "qc_ptp_max_uv_max": float(group["qc_ptp_max_uv"].max()),
                "qc_max_abs_uv_p95": float(group["qc_max_abs_uv"].quantile(0.95)),
            }
        )

    add_summary("all", qc)
    if "participant" in qc.columns:
        for participant, group in qc.groupby(qc["participant"].astype(str)):
            add_summary(f"participant:{participant}", group)

    pd.DataFrame(summaries).to_csv(output_dir / "epoch_qc_summary.csv", index=False)


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
        metadata = row.to_dict()
        metadata.update(
            {
                "tmin": args.tmin,
                "tmax": args.tmax,
                "epoch_shape": "x".join(str(dim) for dim in epoch.shape),
            }
        )
        metadata.update(qc)

        stem = make_stem(row_idx, args.event_code, row)
        epoch_path = epoch_dir / f"{stem}.npz"
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
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved {len(metadata_rows)} event epochs to {output_dir} ({len(qc_rows)} QC candidates)")
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
    add_epoch_qc_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
