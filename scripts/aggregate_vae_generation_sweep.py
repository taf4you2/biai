import argparse
import json
from pathlib import Path

import pandas as pd


def read_summary(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def make_row(participant, resnet, dinov2, ensemble):
    return {
        "participant": participant,
        "images": ensemble["images"],
        "weight_resnet": ensemble["weight_first"],
        "weight_dinov2": ensemble["weight_second"],
        "resnet_l1": resnet["eeg_embedding"]["l1"],
        "resnet_psnr": resnet["eeg_embedding"]["psnr"],
        "resnet_ssim": resnet["eeg_embedding"]["ssim"],
        "dinov2_l1": dinov2["eeg_embedding"]["l1"],
        "dinov2_psnr": dinov2["eeg_embedding"]["psnr"],
        "dinov2_ssim": dinov2["eeg_embedding"]["ssim"],
        "ensemble_l1": ensemble["l1"],
        "ensemble_psnr": ensemble["psnr"],
        "ensemble_ssim": ensemble["ssim"],
    }


def parse_external_source(value):
    parts = value.split("|", maxsplit=3)
    if len(parts) != 4:
        raise ValueError(
            "--external-source must use PARTICIPANT|RESNET_DIR|DINOV2_DIR|ENSEMBLE_DIR"
        )
    participant, resnet_dir, dinov2_dir, ensemble_dir = parts
    return participant, Path(resnet_dir), Path(dinov2_dir), Path(ensemble_dir)


def aggregate(args):
    root = Path(args.results_root)
    rows = []
    for participant in args.participants:
        participant_dir = root / participant.lower()
        resnet = read_summary(participant_dir / "generation_resnet" / "vae_generation_summary.json")
        dinov2 = read_summary(participant_dir / "generation_dinov2" / "vae_generation_summary.json")
        ensemble = read_summary(
            participant_dir / "generation_ensemble" / "ensemble_generation_summary.json"
        )
        rows.append(make_row(participant, resnet, dinov2, ensemble))

    for value in args.external_source:
        participant, resnet_dir, dinov2_dir, ensemble_dir = parse_external_source(value)
        rows.append(
            make_row(
                participant,
                read_summary(resnet_dir / "vae_generation_summary.json"),
                read_summary(dinov2_dir / "vae_generation_summary.json"),
                read_summary(ensemble_dir / "ensemble_generation_summary.json"),
            )
        )

    frame = pd.DataFrame(rows)
    numeric_columns = frame.select_dtypes("number").columns
    aggregate_rows = []
    for label, reducer in (("MEAN", "mean"), ("STD", "std")):
        values = getattr(frame[numeric_columns], reducer)(ddof=1) if reducer == "std" else getattr(frame[numeric_columns], reducer)()
        aggregate_rows.append({"participant": label, **values.to_dict()})
    report = pd.concat([frame, pd.DataFrame(aggregate_rows)], ignore_index=True)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(
            {
                "participants": args.participants,
                "rows": rows,
                "mean": aggregate_rows[0],
                "std": aggregate_rows[1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(f"Saved CSV to {output}")
    print(f"Saved JSON to {json_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate ResNet, DINOv2 and ensemble VAE generations across participants."
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--participants", nargs="+", required=True)
    parser.add_argument(
        "--external-source",
        action="append",
        default=[],
        help="Additional source: PARTICIPANT|RESNET_DIR|DINOV2_DIR|ENSEMBLE_DIR",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())
