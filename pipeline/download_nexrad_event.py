#!/usr/bin/env python3
"""Download NEXRAD Level II radar files for one E-V-E event case."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from case_config import resolve_case, project_root

BUCKET_NAME = "unidata-nexrad-level2"


def extract_hhmm_from_key(key: str) -> str | None:
    filename = Path(key).name
    if "_" not in filename:
        return None
    parts = filename.split("_")
    if len(parts) < 2:
        return None
    time_part = parts[1]
    return time_part[:4] if len(time_part) >= 4 else None


def parse_args() -> argparse.Namespace:
    default_case = os.getenv("EVENT_ID", "test_case_1")
    parser = argparse.ArgumentParser(description="Download NEXRAD radar files for an E-V-E case.")
    parser.add_argument("--event-id", default=default_case, help="Case ID, for example test_case_2.")
    parser.add_argument("--radar-site", default=os.getenv("RADAR_SITE"), help="Radar site, for example KBMX.")
    parser.add_argument("--date-folder", default=os.getenv("DATE_FOLDER"), help="UTC date folder, YYYY-MM-DD.")
    parser.add_argument("--start-hhmm", default=os.getenv("START_HHMM"), help="UTC start time, HHMM.")
    parser.add_argument("--end-hhmm", default=os.getenv("END_HHMM"), help="UTC end time, HHMM.")
    parser.add_argument("--max-downloads", type=int, default=None, help="Maximum radar files to download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = resolve_case(
        args.event_id,
        radar_site=args.radar_site,
        date_folder=args.date_folder,
        start_hhmm=args.start_hhmm,
        end_hhmm=args.end_hhmm,
        max_downloads=args.max_downloads,
    )

    root = project_root()
    out_dir = root / "data" / "raw" / "nexrad" / case.radar_site / case.date_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{case.year}/{case.month}/{case.day}/{case.radar_site}/"

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("=" * 80)
    print("E-V-E — NEXRAD Downloader")
    print("=" * 80)
    print(f"Event ID:          {case.event_id}")
    print(f"Radar site:        {case.radar_site}")
    print(f"Archive date UTC:  {case.date_folder}")
    print(f"Target UTC window: {case.start_hhmm} to {case.end_hhmm}")
    print(f"Output folder:     {out_dir}")
    print()

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if "Contents" not in response:
        print("No files found.")
        return

    selected_keys: list[str] = []
    for obj in response["Contents"]:
        key = obj["Key"]
        filename = Path(key).name
        if filename.endswith("_MDM"):
            continue
        hhmm = extract_hhmm_from_key(key)
        if hhmm is None:
            continue
        if case.start_hhmm <= hhmm <= case.end_hhmm:
            selected_keys.append(key)

    selected_keys = sorted(selected_keys)[: case.max_downloads]

    print(f"Selected radar files: {len(selected_keys)}")
    if not selected_keys:
        print("No radar files matched the requested time window.")
        return

    for index, key in enumerate(selected_keys, start=1):
        filename = Path(key).name
        output_path = out_dir / filename
        if output_path.exists():
            print(f"[{index}/{len(selected_keys)}] Already exists: {filename}")
            continue
        print(f"[{index}/{len(selected_keys)}] Downloading: {filename}")
        s3.download_file(BUCKET_NAME, key, str(output_path))

    print("Done.")


if __name__ == "__main__":
    main()
