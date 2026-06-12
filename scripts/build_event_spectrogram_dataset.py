import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import spectrogram


EVENT_NAME_BY_CODE = {
    10: "FIXATION",
    11: "BLACK_SCREEN_PRE_IMAGE",
    12: "IMAGE_ON",
    13: "BLACK_SCREEN_POST_IMAGE",
    14: "DESCRIBE_SCREEN",
    15: "SPACE_PRESSED",
    16: "BLACK_SCREEN_POST_SPACE",
}


def clean_channel_name(channel_name):
    channel_name = channel_name.replace("EEG ", "")
    return channel_name.split("-")[0].split(":")[0].strip()


def load_bad_channels(impedance_csv, reject_threshold):
    if not impedance_csv or not Path(impedance_csv).exists():
        return []

    df_imp = pd.read_csv(impedance_csv, skiprows=6)
    bad_channels = []
    for col in df_imp.columns:
        if col == "Time":
            continue
        low_snr_ratio = (df_imp[col].astype(str).str.strip() == "Low_SNR").sum() / len(df_imp)
        if low_snr_ratio > reject_threshold:
            bad_channels.append(clean_channel_name(col))
    return bad_channels


def prepare_raw(edf_path, impedance_csv, reject_threshold, l_freq, h_freq, notch_freq):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})

    ch_types = {}
    for ch in raw.ch_names:
        if ch in {"Trigger", "Event"}:
            ch_types[ch] = "stim"
        elif ch.startswith("Imp ") or ch in {"X1", "X2", "X3", "CM", "Ax", "Ay", "Az", "Battery", "Packet Counter"}:
            ch_types[ch] = "misc"
        else:
            ch_types[ch] = "eeg"
    raw.set_channel_types(ch_types, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    bad_channels = load_bad_channels(impedance_csv, reject_threshold)
    raw.info["bads"] = [ch for ch in bad_channels if ch in raw.ch_names]

    raw.pick(picks="eeg", exclude=[])
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=False, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(notch_freq, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def events_from_annotations(raw):
    rows = []
    sfreq = raw.info["sfreq"]
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        try:
            event_code = int(str(description))
        except ValueError:
            continue
        if event_code not in EVENT_NAME_BY_CODE:
            continue
        rows.append(
            {
                "sample": int(round(onset * sfreq)),
                "onset_s": float(onset),
                "event_code": event_code,
                "event_name": EVENT_NAME_BY_CODE[event_code],
            }
        )
    return pd.DataFrame(rows, columns=["sample", "onset_s", "event_code", "event_name"])


def load_csv_events(events_csv):
    if not events_csv or not Path(events_csv).exists():
        return pd.DataFrame(columns=["event_code"])

    csv_events = pd.read_csv(events_csv)
    if "event_code" not in csv_events.columns:
        raise ValueError(f"Missing event_code column in {events_csv}")

    csv_events = csv_events[csv_events["event_code"].notna()].copy()
    csv_events["event_code"] = csv_events["event_code"].astype(int)
    return csv_events


def attach_event_metadata(annotation_events, events_csv):
    csv_events = load_csv_events(events_csv)
    if csv_events.empty:
        return annotation_events.copy()

    csv_events["_event_order"] = csv_events.groupby("event_code").cumcount()

    annotation_events = annotation_events.copy()
    annotation_events["_event_order"] = annotation_events.groupby("event_code").cumcount()

    metadata_cols = [
        "event_code",
        "_event_order",
        "participant_id",
        "session_id",
        "series_id",
        "trial_id",
        "image_id",
        "image_file",
        "image_category",
        "notes",
    ]
    metadata_cols = [col for col in metadata_cols if col in csv_events.columns]
    merged = annotation_events.merge(csv_events[metadata_cols], on=["event_code", "_event_order"], how="left")
    return merged.drop(columns=["_event_order"])


def build_event_alignment_qc(annotation_events, events_csv, merged_events=None, event_codes=None):
    csv_events = load_csv_events(events_csv)
    if event_codes is None:
        event_codes = sorted(EVENT_NAME_BY_CODE)

    rows = []
    for event_code in event_codes:
        if "event_code" in annotation_events.columns:
            annotation_count = int((annotation_events["event_code"] == event_code).sum())
        else:
            annotation_count = 0
        csv_count = int((csv_events["event_code"] == event_code).sum()) if not csv_events.empty else 0

        merged_count = None
        missing_metadata_rows = None
        if merged_events is not None and "event_code" in merged_events.columns:
            merged_for_code = merged_events[merged_events["event_code"] == event_code]
            merged_count = int(len(merged_for_code))
            metadata_cols = [
                col
                for col in ["participant_id", "series_id", "trial_id", "image_id", "image_category"]
                if col in merged_for_code.columns
            ]
            if metadata_cols:
                missing_metadata_rows = int(merged_for_code[metadata_cols].isna().any(axis=1).sum())
            else:
                missing_metadata_rows = merged_count

        if annotation_count == csv_count:
            status = "ok"
        elif annotation_count == 0:
            status = "missing_edf_annotation"
        elif csv_count == 0:
            status = "missing_csv_event"
        else:
            status = "count_mismatch"

        rows.append(
            {
                "event_code": event_code,
                "event_name": EVENT_NAME_BY_CODE.get(event_code, ""),
                "edf_annotation_count": annotation_count,
                "csv_event_count": csv_count,
                "count_delta_edf_minus_csv": annotation_count - csv_count,
                "merged_count": merged_count,
                "missing_metadata_rows": missing_metadata_rows,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def write_event_alignment_qc(annotation_events, events_csv, merged_events, output_path, target_event_code, strict=False):
    qc = build_event_alignment_qc(annotation_events, events_csv, merged_events)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(output_path, index=False)

    target_rows = qc[qc["event_code"] == target_event_code]
    if not target_rows.empty:
        target = target_rows.iloc[0]
        has_missing_metadata = int(target.get("missing_metadata_rows") or 0) > 0
        has_bad_count = target["status"] != "ok"
        if has_bad_count or has_missing_metadata:
            message = (
                f"Event QC warning for code {target_event_code}: "
                f"status={target['status']}, missing_metadata_rows={target['missing_metadata_rows']}"
            )
            if strict:
                raise ValueError(message)
            print(message)

    return qc


def compute_epoch_spectrogram(epoch_data, sfreq, fmin, fmax, nperseg, noverlap):
    channel_specs = []
    selected_freqs = None
    selected_times = None

    for channel_data in epoch_data:
        freqs, times, power = spectrogram(
            channel_data,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        selected_freqs = freqs[freq_mask]
        selected_times = times
        channel_specs.append(np.log10(power[freq_mask] + 1e-20))

    return np.stack(channel_specs).astype(np.float32), selected_freqs.astype(np.float32), selected_times.astype(np.float32)


def save_preview(spec, output_path):
    # Preview only: the model should consume the .npz tensor, not this lossy image.
    image = spec.mean(axis=0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-12)
    plt.figure(figsize=(4, 3), dpi=120)
    plt.imshow(image, aspect="auto", origin="lower", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def build_dataset(args):
    output_dir = Path(args.output_dir)
    tensor_dir = output_dir / "tensors"
    preview_dir = output_dir / "previews"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old_file in tensor_dir.glob("*.npz"):
        old_file.unlink()
    for old_file in preview_dir.glob("*.png"):
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
    metadata_rows = []

    for row_idx, row in target_events.reset_index(drop=True).iterrows():
        start = int(row["sample"]) + start_offset
        stop = int(row["sample"]) + stop_offset
        if start < 0 or stop > raw.n_times or stop <= start:
            continue

        epoch_data = raw.get_data(start=start, stop=stop, reject_by_annotation="omit")
        if epoch_data.shape[1] != stop - start:
            continue

        spec, freqs, times = compute_epoch_spectrogram(
            epoch_data,
            sfreq,
            args.fmin,
            args.fmax,
            args.nperseg,
            args.noverlap,
        )

        trial_id = row.get("trial_id")
        image_category = row.get("image_category")
        stem = f"{row_idx:05d}_code{args.event_code}"
        if pd.notna(trial_id):
            stem += f"_trial{int(trial_id):04d}"
        if isinstance(image_category, str) and image_category:
            stem += f"_{image_category}"

        tensor_path = tensor_dir / f"{stem}.npz"
        preview_path = preview_dir / f"{stem}.png"
        np.savez_compressed(
            tensor_path,
            spectrogram=spec,
            freqs=freqs,
            times=times + args.tmin,
            channels=np.array(raw.ch_names),
        )
        save_preview(spec, preview_path)

        metadata = row.to_dict()
        metadata.update(
            {
                "tensor_path": str(tensor_path),
                "preview_path": str(preview_path),
                "tmin": args.tmin,
                "tmax": args.tmax,
                "spectrogram_shape": "x".join(str(dim) for dim in spec.shape),
            }
        )
        metadata_rows.append(metadata)

    pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved {len(metadata_rows)} event spectrograms to {output_dir}")
    print(f"Tensor shape example: channels x freqs x times = {metadata_rows[0]['spectrogram_shape'] if metadata_rows else 'n/a'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build trigger-aligned EEG spectrogram tensors.")
    parser.add_argument("--edf", default="dane/mole_/mole_0004_raw.edf")
    parser.add_argument("--events-csv", default="dane/mole_/mole_EEGBasedVisualRecall_Events_Rep2_2026-05-22_11-27-28.csv")
    parser.add_argument("--impedance-csv", default="dane/mole_/mole_0004_imp.csv")
    parser.add_argument("--output-dir", default="event_spectrogram_dataset")
    parser.add_argument("--event-code", type=int, default=12, help="Default 12 = IMAGE_ON.")
    parser.add_argument("--tmin", type=float, default=-0.5, help="Seconds before trigger.")
    parser.add_argument("--tmax", type=float, default=1.5, help="Seconds after trigger.")
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=224)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--strict-event-qc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
