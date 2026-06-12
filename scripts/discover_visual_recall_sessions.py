import argparse
import re
from pathlib import Path

import pandas as pd


EVENTS_PATTERN = re.compile(r"(?P<participant>.+)_EEGBasedVisualRecall_Events_Rep(?P<rep>\d+)_.+\.csv$")


def discover_sessions(results_dir):
    results_dir = Path(results_dir)
    rows = []
    for events_path in sorted(results_dir.glob("*_EEGBasedVisualRecall_Events_Rep*.csv")):
        match = EVENTS_PATTERN.match(events_path.name)
        if not match:
            continue

        participant = match.group("participant")
        rep = int(match.group("rep"))
        edf_candidates = sorted(results_dir.glob(f"{participant}*_raw.edf"))
        imp_candidates = sorted(results_dir.glob(f"{participant}*_imp.csv"))
        config_candidates = sorted(results_dir.glob(f"{participant}_EEGBasedVisualRecall_SessionConfig_Rep{rep}_*.json"))

        events = pd.read_csv(events_path, usecols=["event_code"])
        image_on_count = int((events["event_code"] == 12).sum())

        rows.append(
            {
                "participant": participant,
                "rep": rep,
                "edf_path": str(edf_candidates[0]) if edf_candidates else "",
                "impedance_csv": str(imp_candidates[0]) if imp_candidates else "",
                "events_csv": str(events_path),
                "session_config_json": str(config_candidates[0]) if config_candidates else "",
                "image_on_events": image_on_count,
                "edf_size_mb": round(edf_candidates[0].stat().st_size / 1024 / 1024, 2) if edf_candidates else "",
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Discover EEGBasedVisualRecall EDF/event/impedance session triplets.")
    parser.add_argument("--results-dir", default="dane/Wyniki")
    parser.add_argument("--output", default="manifests/visual_recall_sessions_manifest.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = discover_sessions(args.results_dir)
    manifest.to_csv(args.output, index=False)
    print(manifest.to_string(index=False))
    print(f"Saved {len(manifest)} sessions to {args.output}")
