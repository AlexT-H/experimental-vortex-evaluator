#!/usr/bin/env python3
"""Run the E-V-E quality-ranked case-isolated processing sequence."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from case_config import CASES, case_dir, resolve_case

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"

DEFAULT_STAGES = [
    "download_nexrad_event.py",
    "detect_tvs_candidates.py",
    "track_circulations.py",
    "build_ml_features.py",
    "train_track_model.py",
    "render_radar_frames.py",
    "load_postgis.py",
    "export_api_preview.py",
]


def run_stage(script: str, extra: list[str]) -> None:
    path = PIPELINE / script
    if not path.exists():
        raise FileNotFoundError(path)
    cmd = [sys.executable, str(path), *extra]
    print("\n" + "=" * 80)
    print(f"E-V-E — running {script}")
    print("=" * 80)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def clean_case_outputs(event_id: str, include_radar_frames: bool = False) -> None:
    path = case_dir(event_id, ROOT)
    if path.exists():
        print(f"Removing case outputs: {path}")
        shutil.rmtree(path)

    if include_radar_frames:
        frames = ROOT / "outputs" / "web" / "radar_frames" / event_id
        if frames.exists():
            print(f"Removing radar frames: {frames}")
            shutil.rmtree(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated E-V-E quality-ranked event case.")
    parser.add_argument("--event-id", default="test_case_1", choices=sorted(CASES.keys()))
    parser.add_argument("--event-name", default=None)
    parser.add_argument("--radar-site", default=None)
    parser.add_argument("--date-folder", default=None)
    parser.add_argument("--start-hhmm", default=None)
    parser.add_argument("--end-hhmm", default=None)
    parser.add_argument("--max-downloads", type=int, default=None)
    parser.add_argument("--sweep", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--stop-before-labels", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--stop-before-db", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--clean-case", action="store_true")
    parser.add_argument("--clean-radar-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = resolve_case(
        args.event_id,
        event_name=args.event_name,
        radar_site=args.radar_site,
        date_folder=args.date_folder,
        start_hhmm=args.start_hhmm,
        end_hhmm=args.end_hhmm,
        max_downloads=args.max_downloads,
        sweep=args.sweep,
    )

    if args.clean_case:
        clean_case_outputs(case.event_id, include_radar_frames=args.clean_radar_frames)

    common = [
        "--event-id", case.event_id,
        "--radar-site", case.radar_site,
        "--date-folder", case.date_folder,
    ]

    detect_args = [*common, "--sweep", str(case.sweep)]

    download_args = [
        "--event-id", case.event_id,
        "--radar-site", case.radar_site,
        "--date-folder", case.date_folder,
        "--start-hhmm", case.start_hhmm,
        "--end-hhmm", case.end_hhmm,
        "--max-downloads", str(case.max_downloads),
    ]

    stages = DEFAULT_STAGES.copy()
    if args.skip_download:
        stages.remove("download_nexrad_event.py")
    if args.skip_training:
        stages.remove("train_track_model.py")
    if args.stop_before_db:
        stages = [s for s in stages if s not in {"load_postgis.py", "export_api_preview.py"}]
    if args.stop_before_labels:
        stages = ["download_nexrad_event.py", "detect_tvs_candidates.py", "track_circulations.py", "build_ml_features.py"]
        if args.skip_download:
            stages.remove("download_nexrad_event.py")

    print("=" * 80)
    print("E-V-E — Quality-Ranked Case-Isolated Pipeline")
    print("=" * 80)
    print(f"Case:        {case.event_id}")
    print(f"Radar:       {case.radar_site}")
    print(f"Date:        {case.date_folder}")
    print(f"UTC window:  {case.start_hhmm}-{case.end_hhmm}")
    print(f"Sweep:       {case.sweep}")
    print(f"Stages:      {', '.join(stages)}")

    for stage in stages:
        if stage == "download_nexrad_event.py":
            run_stage(stage, download_args)
        elif stage == "detect_tvs_candidates.py":
            run_stage(stage, detect_args)
        elif stage == "load_postgis.py":
            extra = [*common]
            if args.append:
                extra.append("--append")
            run_stage(stage, extra)
        elif stage == "render_radar_frames.py":
            run_stage(stage, detect_args)
        else:
            run_stage(stage, ["--event-id", case.event_id])

    if args.stop_before_labels:
        print("\nStopped before model training/load.")
        print("Label this file, then rerun without --clean-case:")
        print(case_dir(case.event_id, ROOT) / "manual" / "track_labels.csv")


if __name__ == "__main__":
    main()
