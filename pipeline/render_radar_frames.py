#!/usr/bin/env python3
"""
E-V-E

Purpose:
    Render one georeferenced PNG image per NEXRAD scan for a selected event.
    The frontend can display these PNGs with Leaflet's L.imageOverlay().

Outputs:
    outputs/web/radar_frames/{event_id}/
        KINX20240507_020733_V06_sweep1_velocity.png
        ...
        radar_frames.csv
        radar_frames.geojson

CSV columns:
    event_id
    scan_time
    radar_site
    date_folder
    radar_file
    sweep
    product
    field_name
    image_path
    image_url_path
    south
    west
    north
    east
    vmin_kt
    vmax_kt
    display_range_mi
    status
    message

Example:
    docker compose run --rm pipeline python pipeline/13_render_radar_frames.py \
        --event-id test_case_1 \
        --radar-site KINX \
        --date-folder 2024-05-07 \
        --sweep 1 \
        --display-range-mi 120 \
        --vmax-kt 60
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

try:
    import pyart
except ImportError as exc:
    raise ImportError(
        "Missing dependency: pyart.\n\n"
        "This script must run in the same pipeline environment as your radar "
        "detection scripts. If needed, rebuild the pipeline container."
    ) from exc

from case_config import resolve_case


MPS_TO_KT = 1.943844
KM_TO_MI = 0.621371


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_scan_time(path: Path) -> Optional[datetime]:
    """
    Parse NEXRAD-style names such as:
        KINX20240507_020733_V06
    """
    match = re.search(r"([A-Z]{4})(\d{8})[_-](\d{6})", path.name)
    if not match:
        return None

    return datetime.strptime(
        match.group(2) + match.group(3),
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=timezone.utc)


def iso_utc(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def radar_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Radar input directory does not exist: {input_dir}")

    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and not path.name.endswith("_MDM")
        and not path.name.startswith(".")
    )


def clean_field_name(fields: Iterable[str], preferred: List[str]) -> Optional[str]:
    fields_set = set(fields)

    for name in preferred:
        if name in fields_set:
            return name

    lowered = {field.lower(): field for field in fields}
    for name in preferred:
        if name.lower() in lowered:
            return lowered[name.lower()]

    return None


def to_nan(data: Any) -> np.ndarray:
    if np.ma.isMaskedArray(data):
        return data.filled(np.nan).astype(float)
    return np.asarray(data, dtype=float)


def get_velocity_colormap():
    """
    Symmetric green/gray/red Doppler velocity colormap.

    Negative/inbound velocity renders green, 0 kt renders neutral gray, and
    positive/outbound velocity renders red. There are no purple extremes.
    """
    cmap = LinearSegmentedColormap.from_list(
        "eve_green_gray_red_centered",
        [
            (0.00, "#064e3b"),
            (0.18, "#15803d"),
            (0.35, "#86efac"),
            (0.50, "#e5e7eb"),
            (0.65, "#fca5a5"),
            (0.82, "#dc2626"),
            (1.00, "#7f1d1d"),
        ],
    )
    cmap.set_under("#022c22")
    cmap.set_over("#450a0a")
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def sweep_data(
    radar: Any,
    sweep: int,
    field_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return velocity, latitude, longitude, and range arrays for a radar sweep.
    """
    nsweeps = int(getattr(radar, "nsweeps", 0))
    if sweep < 0 or sweep >= nsweeps:
        raise ValueError(f"Requested sweep {sweep}, but radar file only has {nsweeps} sweeps.")

    start = int(radar.sweep_start_ray_index["data"][sweep])
    end = int(radar.sweep_end_ray_index["data"][sweep])

    values = to_nan(radar.fields[field_name]["data"][start:end + 1, :])

    gate_x_m, gate_y_m, _ = radar.get_gate_x_y_z(sweep)
    range_km = np.sqrt((gate_x_m / 1000.0) ** 2 + (gate_y_m / 1000.0) ** 2)

    lats, lons, _ = radar.get_gate_lat_lon_alt(sweep)

    return values, np.asarray(lats, dtype=float), np.asarray(lons, dtype=float), range_km


def finite_bounds(lats: np.ndarray, lons: np.ndarray, mask: np.ndarray, pad_deg: float) -> Optional[Dict[str, float]]:
    if not np.any(mask):
        return None

    valid_lats = lats[mask]
    valid_lons = lons[mask]

    south = float(np.nanmin(valid_lats)) - pad_deg
    north = float(np.nanmax(valid_lats)) + pad_deg
    west = float(np.nanmin(valid_lons)) - pad_deg
    east = float(np.nanmax(valid_lons)) + pad_deg

    if not all(math.isfinite(value) for value in [south, north, west, east]):
        return None

    if south >= north or west >= east:
        return None

    return {
        "south": south,
        "west": west,
        "north": north,
        "east": east,
    }


def render_png(
    output_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    vel_kt: np.ndarray,
    bounds: Dict[str, float],
    vmin_kt: float,
    vmax_kt: float,
    dpi: int,
    image_size_in: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = get_velocity_colormap()

    fig = plt.figure(
        figsize=(image_size_in, image_size_in),
        dpi=dpi,
        frameon=False,
    )
    fig.patch.set_alpha(0)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0, 0, 0, 0))

    abs_limit = max(abs(float(vmin_kt)), abs(float(vmax_kt)), 1.0)
    norm = TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit)

    ax.pcolormesh(
        lons,
        lats,
        np.ma.masked_invalid(vel_kt),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlim(bounds["west"], bounds["east"])
    ax.set_ylim(bounds["south"], bounds["north"])
    ax.axis("off")

    fig.savefig(
        output_path,
        transparent=True,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)


def footprint_feature(row: Dict[str, Any]) -> Dict[str, Any]:
    west = float(row["west"])
    east = float(row["east"])
    south = float(row["south"])
    north = float(row["north"])

    props = dict(row)
    for key in ["west", "east", "south", "north"]:
        props.pop(key, None)

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]],
        },
        "properties": props,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_id",
        "scan_time",
        "radar_site",
        "date_folder",
        "radar_file",
        "sweep",
        "product",
        "field_name",
        "image_path",
        "image_url_path",
        "south",
        "west",
        "north",
        "east",
        "vmin_kt",
        "vmax_kt",
        "display_range_mi",
        "status",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_geojson(path: Path, rows: List[Dict[str, Any]]) -> None:
    features = [
        footprint_feature(row)
        for row in rows
        if row.get("status") == "ok"
        and row.get("south") not in {"", None}
        and row.get("west") not in {"", None}
        and row.get("north") not in {"", None}
        and row.get("east") not in {"", None}
    ]

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def render_frame(path: Path, args: argparse.Namespace, output_event_dir: Path) -> Dict[str, Any]:
    scan_dt = parse_scan_time(path)
    scan_iso = iso_utc(scan_dt)

    base_row: Dict[str, Any] = {
        "event_id": args.event_id,
        "scan_time": scan_iso,
        "radar_site": args.radar_site,
        "date_folder": args.date_folder,
        "radar_file": path.name,
        "sweep": args.sweep,
        "product": args.product,
        "field_name": "",
        "image_path": "",
        "image_url_path": "",
        "south": "",
        "west": "",
        "north": "",
        "east": "",
        "vmin_kt": -float(args.vmax_kt),
        "vmax_kt": float(args.vmax_kt),
        "display_range_mi": float(args.display_range_mi),
        "status": "error",
        "message": "",
    }

    try:
        radar = pyart.io.read(str(path))
        field = clean_field_name(
            radar.fields.keys(),
            [
                args.field_name,
                "velocity",
                "corrected_velocity",
                "VEL",
                "VEL2",
                "v",
            ],
        )

        if not field:
            raise ValueError(f"No velocity field found in {path.name}")

        velocity_mps, lats, lons, range_km = sweep_data(radar, args.sweep, field)
        velocity_kt = velocity_mps * MPS_TO_KT
        range_mi = range_km * KM_TO_MI

        valid = (
            np.isfinite(velocity_kt)
            & np.isfinite(lats)
            & np.isfinite(lons)
            & (range_mi <= float(args.display_range_mi))
        )

        if np.sum(valid) < int(args.min_valid_gates):
            raise ValueError(
                f"Too few valid gates after filtering: {int(np.sum(valid))}"
            )

        # Transparent outside requested range/valid data.
        velocity_kt = np.where(valid, velocity_kt, np.nan)

        bounds = finite_bounds(lats, lons, valid, float(args.pad_deg))
        if bounds is None:
            raise ValueError("Could not calculate valid lat/lon bounds for image overlay")

        png_name = f"{path.stem}_sweep{args.sweep}_{args.product}.png"
        png_path = output_event_dir / png_name

        render_png(
            output_path=png_path,
            lats=lats,
            lons=lons,
            vel_kt=velocity_kt,
            bounds=bounds,
            vmin_kt=-float(args.vmax_kt),
            vmax_kt=float(args.vmax_kt),
            dpi=int(args.dpi),
            image_size_in=float(args.image_size_in),
        )

        base_row.update(
            {
                "field_name": field,
                "image_path": str(png_path.relative_to(project_root())).replace("\\", "/"),
                "image_url_path": f"/radar_frames/{args.event_id}/{png_name}",
                "south": bounds["south"],
                "west": bounds["west"],
                "north": bounds["north"],
                "east": bounds["east"],
                "status": "ok",
                "message": "",
            }
        )

    except Exception as exc:
        base_row["status"] = "error"
        base_row["message"] = str(exc)

    return base_row


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Render Doppler velocity PNG overlays for E-V-E frontend radar scan slider."
    )

    parser.add_argument("--event-id", default=os.getenv("EVENT_ID", "test_case_1"))
    parser.add_argument("--radar-site", default=os.getenv("RADAR_SITE"))
    parser.add_argument("--date-folder", default=os.getenv("DATE_FOLDER"))
    parser.add_argument("--sweep", type=int, default=int(os.getenv("SWEEP", "1")))

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Raw NEXRAD directory. Defaults to data/raw/nexrad/{radar_site}/{date_folder}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "web" / "radar_frames",
        help="Base output folder. Event-specific subfolder is created inside it.",
    )

    parser.add_argument("--product", default="velocity")
    parser.add_argument(
        "--field-name",
        default="velocity",
        help="Preferred radar field name. Script falls back to common velocity names.",
    )

    parser.add_argument("--display-range-mi", type=float, default=120.0)
    parser.add_argument("--vmax-kt", type=float, default=60.0)
    parser.add_argument("--pad-deg", type=float, default=0.03)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--image-size-in", type=float, default=8.0)
    parser.add_argument("--min-valid-gates", type=int, default=100)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete outputs/web/radar_frames/{event_id} before rendering new frames.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = resolve_case(args.event_id, radar_site=args.radar_site, date_folder=args.date_folder, sweep=args.sweep)
    args.event_id = case.event_id
    args.radar_site = case.radar_site
    args.date_folder = case.date_folder
    args.sweep = case.sweep

    input_dir = args.input_dir or (
        project_root() / "data" / "raw" / "nexrad" / args.radar_site / args.date_folder
    )

    output_event_dir = args.output_dir / args.event_id
    if args.clean_output and output_event_dir.exists():
        import shutil
        shutil.rmtree(output_event_dir)
    output_event_dir.mkdir(parents=True, exist_ok=True)

    files = radar_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No radar files found in {input_dir}")

    print("=" * 100)
    print("E-V-E — Render Doppler Radar Frames")
    print("=" * 100)
    print(f"Event ID:        {args.event_id}")
    print(f"Radar site:      {args.radar_site}")
    print(f"Date folder:     {args.date_folder}")
    print(f"Sweep:           {args.sweep}")
    print(f"Input dir:       {input_dir}")
    print(f"Output dir:      {output_event_dir}")
    print(f"Frames to render:{len(files)}")
    print()

    rows: List[Dict[str, Any]] = []

    for idx, path in enumerate(files, start=1):
        print(f"[{idx:03d}/{len(files):03d}] {path.name}")
        row = render_frame(path, args, output_event_dir)
        rows.append(row)

        if row["status"] == "ok":
            print(f"  OK  → {row['image_path']}")
        else:
            print(f"  ERR → {row['message']}")

    csv_path = output_event_dir / "radar_frames.csv"
    geojson_path = output_event_dir / "radar_frames.geojson"

    write_csv(csv_path, rows)
    write_geojson(geojson_path, rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    error_count = len(rows) - ok_count

    print()
    print("Saved:")
    print(f"  CSV metadata:       {csv_path}")
    print(f"  GeoJSON footprints: {geojson_path}")
    print()
    print(f"Rendered OK: {ok_count}")
    print(f"Errors:      {error_count}")

    if error_count:
        print()
        print("Frames with errors:")
        for row in rows:
            if row["status"] != "ok":
                print(f"  {row['radar_file']}: {row['message']}")

    print()
    print("Next backend step:")
    print("  1. Load radar_frames.csv into a radar_frames PostGIS table.")
    print("  2. Mount outputs/web/radar_frames at /radar_frames in FastAPI.")
    print("  3. Add GET /events/{event_id}/radar-frame?time=...")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
