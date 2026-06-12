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


def build_session_epochs(session, output_dir, args):
    participant = str(session["participant"])
    participant_dir = output_dir / "epochs" / participant
    qc_dir = output_dir / "event_alignment_qc"
    participant_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    for old_file in participant_dir.glob("*.npz"):
        old_file.unlink()

    raw = prepare_raw(
        session["edf_path"],
        session["impedance_csv"],
        args.reject_threshold,
        args.l_freq,
        args.h_freq,
        args.notch_freq,
    )
    annotation_events = events_from_annotations(raw)
    events = attach_event_metadata(annotation_events, session["events_csv"])
    alignment_qc = write_event_alignment_qc(
        annotation_events,
        session["events_csv"],
        events,
        qc_dir / f"{participant}_rep{int(session['rep'])}.csv",
        args.event_code,
        strict=args.strict_event_qc,
    )
    target_events = events[events["event_code"] == args.event_code].copy()
    if args.max_trials_per_session:
        target_events = target_events.head(args.max_trials_per_session)

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

        epoch_qc = compute_epoch_qc(epoch, raw.ch_names, args)
        stem = f"{participant}_{make_stem(row_idx, args.event_code, row)}"
        epoch_path = participant_dir / f"{stem}.npz"
        metadata = row.to_dict()
        metadata.update(
            {
                "participant": participant,
                "rep": int(session["rep"]),
                "source_edf": session["edf_path"],
                "source_events_csv": session["events_csv"],
                "tmin": args.tmin,
                "tmax": args.tmax,
                "epoch_shape": "x".join(str(dim) for dim in epoch.shape),
            }
        )
        metadata.update(epoch_qc)
        qc_metadata = metadata.copy()
        qc_metadata.update({"epoch_path": str(epoch_path), "qc_saved": False})

        if args.drop_rejected and not epoch_qc["qc_accepted"]:
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

    rejected = sum(not row["qc_accepted"] for row in qc_rows)
    print(f"{participant}: saved {len(metadata_rows)} epochs ({rejected}/{len(qc_rows)} rejected by QC)")
    target_qc = alignment_qc[alignment_qc["event_code"] == args.event_code].iloc[0].to_dict()
    return metadata_rows, qc_rows, target_qc


def build_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "epochs").mkdir(exist_ok=True)

    manifest = pd.read_csv(args.sessions_manifest)
    all_rows = []
    all_qc_rows = []
    session_summaries = []
    for _, session in manifest.iterrows():
        rows, qc_rows, target_qc = build_session_epochs(session, output_dir, args)
        all_rows.extend(rows)
        all_qc_rows.extend(qc_rows)
        session_summaries.append(
            {
                "participant": session["participant"],
                "rep": int(session["rep"]),
                "saved_epochs": len(rows),
                "qc_candidate_epochs": len(qc_rows),
                "qc_rejected_epochs": sum(not row["qc_accepted"] for row in qc_rows),
                "image_on_events": int(session["image_on_events"]),
                "target_event_code": args.event_code,
                "target_edf_annotation_count": int(target_qc["edf_annotation_count"]),
                "target_csv_event_count": int(target_qc["csv_event_count"]),
                "target_missing_metadata_rows": int(target_qc["missing_metadata_rows"] or 0),
                "target_event_qc_status": target_qc["status"],
                "edf_path": session["edf_path"],
            }
        )

    metadata = pd.DataFrame(all_rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(session_summaries).to_csv(output_dir / "session_summary.csv", index=False)
    write_epoch_qc_reports(all_qc_rows, output_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved {len(metadata)} total epochs to {output_dir}")
    print(pd.DataFrame(session_summaries).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Build a multi-session raw EEG epoch dataset from a sessions manifest.")
    parser.add_argument("--sessions-manifest", default="manifests/visual_recall_sessions_manifest.csv")
    parser.add_argument("--output-dir", default="event_epoch_multisession_image_on_0_0p8")
    parser.add_argument("--event-code", type=int, default=12)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--max-trials-per-session", type=int, default=None)
    parser.add_argument("--strict-event-qc", action="store_true")
    add_epoch_qc_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
