from pathlib import Path
import argparse
import os
import shutil
import csv
import math
import re
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pyart
from scipy import ndimage
from scipy.spatial import cKDTree

from case_config import case_dir, resolve_case


# =============================================================================
# E-V-E
# Storm-Relative Banded TVS Candidate Detection
#
# Detects vortex-like velocity signatures as compact, storm-relative dipoles:
#   bright inbound + bright outbound lobes
#   weaker same-sign bands around the lobes
#   gray / near-zero transition zone near the center
#   semi-circular / opposing lobe geometry
#
# This intentionally avoids treating a simple straight shear line as sufficient.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_ID = os.getenv("EVENT_ID", "test_case_1")
RADAR_SITE = os.getenv("RADAR_SITE", "KINX")
DATE_FOLDER = os.getenv("DATE_FOLDER", "2024-05-07")
SWEEP = int(os.getenv("SWEEP", "1"))
SWEEP_INDEX = SWEEP


def iso_utc(value: datetime | None) -> str:
    """Return a stable UTC ISO string ending in Z for CSV/API/PostGIS use."""
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def configure_case(event_id: str, radar_site: str | None = None, date_folder: str | None = None, sweep: int | None = None) -> None:
    """
    Configure input/output paths for one isolated E-V-E case.

    The detection calculations below are preserved from the supplied correct
    detector. This only changes paths, case metadata, and the plot colormap.
    """
    global EVENT_ID, RADAR_SITE, DATE_FOLDER, SWEEP, SWEEP_INDEX
    global RADAR_DIR, CASE_OUT_DIR, FIGURE_OUT_DIR, CANDIDATE_OUT_DIR, LOG_OUT_DIR, EVENT_TIMES_CSV

    case = resolve_case(event_id, radar_site=radar_site, date_folder=date_folder, sweep=sweep)
    EVENT_ID = case.event_id
    RADAR_SITE = case.radar_site
    DATE_FOLDER = case.date_folder
    SWEEP = int(case.sweep)
    SWEEP_INDEX = SWEEP

    RADAR_DIR = PROJECT_ROOT / "data" / "raw" / "nexrad" / RADAR_SITE / DATE_FOLDER
    CASE_OUT_DIR = case_dir(EVENT_ID, PROJECT_ROOT)
    FIGURE_OUT_DIR = CASE_OUT_DIR / "figures"
    CANDIDATE_OUT_DIR = CASE_OUT_DIR / "candidates"
    LOG_OUT_DIR = CASE_OUT_DIR / "logs"
    EVENT_TIMES_CSV = LOG_OUT_DIR / f"{EVENT_ID}_event_times.csv"


def scan_metadata(path: Path, t: datetime | None) -> dict:
    return {
        "event_id": EVENT_ID,
        "radar_site": RADAR_SITE,
        "date_folder": DATE_FOLDER,
        "radar_file": path.name,
        "scan_time": iso_utc(t),
        "scan_time_utc": iso_utc(t),
        "sweep": SWEEP,
    }


configure_case(EVENT_ID, RADAR_SITE, DATE_FOLDER, SWEEP)

MPS_TO_KT = 1.943844
KM_TO_MI = 0.621371

# Range / intensity.
MIN_RANGE_MI = 30.0
MAX_RANGE_MI = 70.0
NEAR_RANGE_MI = 35.0
NEAR_DELTAV_KT = 90.0
MID_DELTAV_KT = 65.0

# Storm motion from simple scan-to-scan echo centroid displacement.
SRV_ENABLED = True
MOTION_REFL_DBZ = 30.0
MOTION_MIN_GATES = 500
MAX_STORM_SPEED_KT = 70.0
SMOOTH_MOTION = 0.55

# Candidate scanning.
CENTER_STEP = 3
PATCH_RADIUS_KM = 6.0
MIN_ECHO_DBZ = 15.0
MIN_PATCH_MAX_DBZ = 30.0
MIN_PATCH_ECHO_GATES = 20
ECHO_GATE_THRESHOLD_DBZ = 20.0

# Banded dipole thresholds.
STRONG_VEL_KT = 25.0
MID_VEL_KT = 12.0
NEUTRAL_VEL_KT = 8.0
CENTER_NEUTRAL_RADIUS_KM = 1.25

MIN_STRONG_GATES_PER_LOBE = 4
MIN_MID_GATES_PER_LOBE = 8

MIN_LOBE_SEPARATION_KM = 1.0
MAX_LOBE_SEPARATION_KM = 9.0
MAX_LOBE_RANGE_DIFF_KM = 4.0

MIN_OPPOSITION_SCORE = 0.45
MIN_BANDED_SCORE = 0.35
MIN_NEUTRAL_SCORE = 0.10
MIN_FINAL_SCORE = 72.0

DEDUP_DISTANCE_KM = 10.0
MAX_CANDIDATES = 3

# Continuity-aware ranking.
# This keeps the same physical vortex ranked as star 1 when it briefly weakens
# or becomes the second-best single-frame detection.
MAX_CONTINUITY_DISTANCE_KM = 28.0
MAX_CONTINUITY_BONUS = 22.0
CONTINUITY_KEEPALIVE_SCANS = 2


def eve_velocity_colormap():
    """Green/gray/red velocity display with 0 kt centered and no purple."""
    cmap = LinearSegmentedColormap.from_list(
        "eve_velocity_green_gray_red_centered",
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
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def radar_files() -> list[Path]:
    return sorted(p for p in RADAR_DIR.iterdir() if p.is_file() and not p.name.endswith("_MDM"))


def scan_time(path: Path) -> datetime | None:
    m = re.search(r"[A-Z]{4}(\d{8})_(\d{6})", path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def field_name(fields: set[str], names: list[str]) -> str | None:
    return next((n for n in names if n in fields), None)


def to_nan(data):
    return data.filled(np.nan) if np.ma.isMaskedArray(data) else np.asarray(data, dtype=float)


def sweep_arrays(radar, name: str):
    start = radar.sweep_start_ray_index["data"][SWEEP]
    end = radar.sweep_end_ray_index["data"][SWEEP]
    data = to_nan(radar.fields[name]["data"][start:end + 1, :])

    x_m, y_m, _ = radar.get_gate_x_y_z(SWEEP)
    x = x_m / 1000.0
    y = y_m / 1000.0
    rng = np.sqrt(x**2 + y**2)
    lats, lons, _ = radar.get_gate_lat_lon_alt(SWEEP)

    return data, x, y, rng, rng * KM_TO_MI, lats, lons


def azimuth_deg(x: float, y: float) -> float:
    return math.degrees(math.atan2(x, y)) % 360.0


def signed_angle_diff(angle, center):
    return (angle - center + 180.0) % 360.0 - 180.0


def delta_required(range_mi: float) -> float | None:
    if range_mi < MIN_RANGE_MI or range_mi > MAX_RANGE_MI:
        return None
    return NEAR_DELTAV_KT if range_mi <= NEAR_RANGE_MI else MID_DELTAV_KT


def echo_centroid(refl, x, y, rng_mi):
    if refl is None:
        return None

    mask = np.isfinite(refl) & (refl >= MOTION_REFL_DBZ) & (rng_mi <= 120.0)
    if np.sum(mask) < MOTION_MIN_GATES:
        return None

    weights = np.maximum(refl[mask] - MOTION_REFL_DBZ, 1.0)
    return {
        "x": float(np.sum(x[mask] * weights) / np.sum(weights)),
        "y": float(np.sum(y[mask] * weights) / np.sum(weights)),
    }


def storm_motion(prev_centroid, curr_centroid, prev_time, curr_time, prev_motion):
    zero = {"u_mps": 0.0, "v_mps": 0.0, "u_kt": 0.0, "v_kt": 0.0, "speed_kt": 0.0, "method": "base_first_scan"}

    if not SRV_ENABLED:
        return {**zero, "method": "disabled"}

    if prev_centroid is None or curr_centroid is None or prev_time is None or curr_time is None:
        return prev_motion or zero

    dt = (curr_time - prev_time).total_seconds()
    if dt <= 0:
        return prev_motion or zero

    u_mps = (curr_centroid["x"] - prev_centroid["x"]) * 1000.0 / dt
    v_mps = (curr_centroid["y"] - prev_centroid["y"]) * 1000.0 / dt

    speed_kt = math.sqrt((u_mps * MPS_TO_KT) ** 2 + (v_mps * MPS_TO_KT) ** 2)
    if speed_kt > MAX_STORM_SPEED_KT:
        return prev_motion or zero

    if prev_motion and prev_motion["method"] != "base_first_scan":
        u_mps = SMOOTH_MOTION * prev_motion["u_mps"] + (1.0 - SMOOTH_MOTION) * u_mps
        v_mps = SMOOTH_MOTION * prev_motion["v_mps"] + (1.0 - SMOOTH_MOTION) * v_mps

    u_kt = u_mps * MPS_TO_KT
    v_kt = v_mps * MPS_TO_KT

    return {
        "u_mps": u_mps,
        "v_mps": v_mps,
        "u_kt": u_kt,
        "v_kt": v_kt,
        "speed_kt": math.sqrt(u_kt**2 + v_kt**2),
        "method": "echo_centroid_displacement",
    }


def storm_relative(base_mps, x, y, rng, motion):
    if not SRV_ENABLED:
        return base_mps * MPS_TO_KT

    rx = np.divide(x, rng, out=np.zeros_like(x, dtype=float), where=rng > 0)
    ry = np.divide(y, rng, out=np.zeros_like(y, dtype=float), where=rng > 0)
    storm_radial = motion["u_mps"] * rx + motion["v_mps"] * ry
    return (base_mps - storm_radial) * MPS_TO_KT


def gate_table(vel, refl, x, y, rng, rng_mi, lats, lons):
    valid = np.isfinite(vel) & (rng_mi >= MIN_RANGE_MI) & (rng_mi <= MAX_RANGE_MI)

    if refl is not None:
        valid &= np.isfinite(refl) & (refl >= MIN_ECHO_DBZ)

    rows, cols = np.where(valid)
    xs = x[rows, cols]
    ys = y[rows, cols]

    table = {
        "x": xs,
        "y": ys,
        "range_km": rng[rows, cols],
        "range_mi": rng_mi[rows, cols],
        "lat": lats[rows, cols],
        "lon": lons[rows, cols],
        "vel": vel[rows, cols],
        "refl": refl[rows, cols] if refl is not None else np.full(rows.shape, np.nan),
        "az": np.array([azimuth_deg(float(a), float(b)) for a, b in zip(xs, ys)]),
    }

    return table, cKDTree(np.column_stack([xs, ys]))


def circular_mean_deg(angles, weights):
    rad = np.radians(angles)
    sx = np.sum(np.cos(rad) * weights)
    sy = np.sum(np.sin(rad) * weights)
    return math.degrees(math.atan2(sy, sx)) % 360.0


def angle_sep_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def candidate_from_center(path, t, center_idx, neighbors, table, motion):
    cx = float(table["x"][center_idx])
    cy = float(table["y"][center_idx])
    cr_mi = float(table["range_mi"][center_idx])
    required = delta_required(cr_mi)
    if required is None or len(neighbors) < 12:
        return None

    idx = np.array(neighbors, dtype=int)
    nx = table["x"][idx]
    ny = table["y"][idx]
    nv = table["vel"][idx]
    nr = table["range_km"][idx]
    nlat = table["lat"][idx]
    nlon = table["lon"][idx]
    nrefl = table["refl"][idx]

    d = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
    use = d <= PATCH_RADIUS_KM
    if not np.any(use):
        return None

    nx, ny, nv, nr, nlat, nlon, nrefl, d = [a[use] for a in (nx, ny, nv, nr, nlat, nlon, nrefl, d)]

    finite_refl = nrefl[np.isfinite(nrefl)]
    if finite_refl.size and (np.nanmax(finite_refl) < MIN_PATCH_MAX_DBZ or np.sum(finite_refl >= ECHO_GATE_THRESHOLD_DBZ) < MIN_PATCH_ECHO_GATES):
        return None

    pos_strong = nv >= STRONG_VEL_KT
    neg_strong = nv <= -STRONG_VEL_KT
    pos_mid = nv >= MID_VEL_KT
    neg_mid = nv <= -MID_VEL_KT

    if np.sum(pos_strong) < MIN_STRONG_GATES_PER_LOBE or np.sum(neg_strong) < MIN_STRONG_GATES_PER_LOBE:
        return None

    if np.sum(pos_mid) < MIN_MID_GATES_PER_LOBE or np.sum(neg_mid) < MIN_MID_GATES_PER_LOBE:
        return None

    pos_w = np.maximum(nv[pos_strong], 1.0)
    neg_w = np.maximum(np.abs(nv[neg_strong]), 1.0)

    px = float(np.sum(nx[pos_strong] * pos_w) / np.sum(pos_w))
    py = float(np.sum(ny[pos_strong] * pos_w) / np.sum(pos_w))
    nxg = float(np.sum(nx[neg_strong] * neg_w) / np.sum(neg_w))
    nyg = float(np.sum(ny[neg_strong] * neg_w) / np.sum(neg_w))

    sep = math.sqrt((px - nxg) ** 2 + (py - nyg) ** 2)
    if sep < MIN_LOBE_SEPARATION_KM or sep > MAX_LOBE_SEPARATION_KM:
        return None

    range_diff = abs(math.sqrt(px**2 + py**2) - math.sqrt(nxg**2 + nyg**2))
    if range_diff > MAX_LOBE_RANGE_DIFF_KM:
        return None

    cand_x = (px + nxg) / 2.0
    cand_y = (py + nyg) / 2.0
    cand_range_km = math.sqrt(cand_x**2 + cand_y**2)
    cand_range_mi = cand_range_km * KM_TO_MI

    required = delta_required(cand_range_mi)
    if required is None:
        return None

    pos_core = float(np.nanpercentile(nv[pos_strong], 90))
    neg_core = float(np.nanpercentile(nv[neg_strong], 10))
    delta_v = abs(pos_core) + abs(neg_core)
    if delta_v < required:
        return None

    local_theta = (np.degrees(np.arctan2(ny - cand_y, nx - cand_x)) + 360.0) % 360.0
    pos_angle = circular_mean_deg(local_theta[pos_strong], np.maximum(nv[pos_strong], 1.0))
    neg_angle = circular_mean_deg(local_theta[neg_strong], np.maximum(np.abs(nv[neg_strong]), 1.0))
    opposition = 1.0 - min(1.0, abs(angle_sep_deg(pos_angle, neg_angle) - 180.0) / 90.0)
    if opposition < MIN_OPPOSITION_SCORE:
        return None

    # Same-sign weaker bands around strong lobes: this rejects isolated pixels and straight-line noise.
    pos_band = np.sum(pos_mid) / max(1, np.sum(pos_strong))
    neg_band = np.sum(neg_mid) / max(1, np.sum(neg_strong))
    banded = min(1.0, min(pos_band, neg_band) / 2.5)
    if banded < MIN_BANDED_SCORE:
        return None

    # Gray / neutral transition near center.
    neutral = (np.abs(nv) <= NEUTRAL_VEL_KT) & (np.sqrt((nx - cand_x) ** 2 + (ny - cand_y) ** 2) <= CENTER_NEUTRAL_RADIUS_KM)
    neutral_score = min(1.0, np.sum(neutral) / 6.0)
    if neutral_score < MIN_NEUTRAL_SCORE:
        return None

    # Radial split: positive/negative lobes should be on opposite sides of radar radial through candidate.
    center_az = azimuth_deg(cand_x, cand_y)
    pos_side = signed_angle_diff(np.array([azimuth_deg(px, py)]), center_az)[0]
    neg_side = signed_angle_diff(np.array([azimuth_deg(nxg, nyg)]), center_az)[0]
    if pos_side * neg_side >= 0:
        return None

    side_balance = min(abs(pos_side), abs(neg_side)) / max(abs(pos_side), abs(neg_side), 0.001)

    reflectivity_score = 0.0
    if finite_refl.size:
        reflectivity_score = min(1.0, max(0.0, (float(np.nanmax(finite_refl)) - 30.0) / 30.0))

    score = (
        38.0 * min(1.4, delta_v / required)
        + 20.0 * opposition
        + 15.0 * banded
        + 12.0 * neutral_score
        + 10.0 * side_balance
        + 5.0 * reflectivity_score
    )
    score = round(min(100.0, score), 2)

    if score < MIN_FINAL_SCORE:
        return None

    return {
        **scan_metadata(path, t),
        "latitude": float(np.nanmean(nlat[(pos_strong | neg_strong)])),
        "longitude": float(np.nanmean(nlon[(pos_strong | neg_strong)])),
        "center_x_km": cand_x,
        "center_y_km": cand_y,
        "radar_range_km": cand_range_km,
        "radar_range_miles": cand_range_mi,
        "positive_lobe_x_km": px,
        "positive_lobe_y_km": py,
        "negative_lobe_x_km": nxg,
        "negative_lobe_y_km": nyg,
        "positive_core_velocity_kt": pos_core,
        "negative_core_velocity_kt": neg_core,
        "delta_v_kt": delta_v,
        "required_delta_v_kt": required,
        "lobe_separation_km": sep,
        "range_difference_km": range_diff,
        "opposition_score": opposition,
        "banded_score": banded,
        "neutral_center_score": neutral_score,
        "side_balance_score": side_balance,
        "window_max_reflectivity_dbz": float(np.nanmax(finite_refl)) if finite_refl.size else "",
        "storm_motion_u_kt": motion["u_kt"],
        "storm_motion_v_kt": motion["v_kt"],
        "storm_motion_speed_kt": motion["speed_kt"],
        "storm_motion_method": motion["method"],
        "confidence_score": score,
    }


def dedupe(candidates):
    candidates.sort(key=lambda c: c["confidence_score"], reverse=True)
    selected = []
    for c in candidates:
        if all(math.hypot(c["center_x_km"] - s["center_x_km"], c["center_y_km"] - s["center_y_km"]) >= DEDUP_DISTANCE_KM for s in selected):
            selected.append(c)
        if len(selected) >= MAX_CANDIDATES:
            break
    return selected


def detect(path, t, srv, refl, x, y, rng, rng_mi, lats, lons, motion):
    table, tree = gate_table(srv, refl, x, y, rng, rng_mi, lats, lons)
    raw = []

    for center_idx in range(0, len(table["x"]), CENTER_STEP):
        neighbors = tree.query_ball_point([float(table["x"][center_idx]), float(table["y"][center_idx])], r=PATCH_RADIUS_KM)
        c = candidate_from_center(path, t, center_idx, neighbors, table, motion)
        if c:
            raw.append(c)

    return dedupe(raw), len(raw)



def apply_continuity_ranking(candidates, previous_main, previous_main_age):
    """
    Re-rank detections using the previous main vortex position.

    This does not create full tracks. It only makes the star ranking less jumpy
    by preferring a candidate that is still close to the previous main vortex.
    """
    usable_previous = previous_main is not None and previous_main_age <= CONTINUITY_KEEPALIVE_SCANS

    for c in candidates:
        base_score = float(c["confidence_score"])
        c["base_confidence_score"] = base_score
        c["continuity_bonus"] = 0.0
        c["track_adjusted_score"] = base_score
        c["distance_from_previous_main_km"] = ""
        c["is_main_candidate"] = False

        if not usable_previous:
            continue

        distance = math.hypot(
            c["center_x_km"] - previous_main["center_x_km"],
            c["center_y_km"] - previous_main["center_y_km"],
        )

        c["distance_from_previous_main_km"] = distance

        if distance <= MAX_CONTINUITY_DISTANCE_KM:
            bonus = MAX_CONTINUITY_BONUS * (1.0 - distance / MAX_CONTINUITY_DISTANCE_KM)
            c["continuity_bonus"] = round(bonus, 2)
            c["track_adjusted_score"] = round(base_score + bonus, 2)

    candidates.sort(key=lambda row: row["track_adjusted_score"], reverse=True)

    for i, c in enumerate(candidates):
        c["is_main_candidate"] = i == 0

    return candidates

def save_candidates(path, candidates):
    CANDIDATE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = CANDIDATE_OUT_DIR / f"{path.stem}_sweep{SWEEP}_banded_srv_tvs_candidates.csv"

    fields = [
        "event_id", "radar_site", "date_folder", "radar_file", "scan_time", "scan_time_utc", "sweep",
        "latitude", "longitude",
        "center_x_km", "center_y_km", "radar_range_km", "radar_range_miles",
        "positive_lobe_x_km", "positive_lobe_y_km", "negative_lobe_x_km", "negative_lobe_y_km",
        "positive_core_velocity_kt", "negative_core_velocity_kt", "delta_v_kt", "required_delta_v_kt",
        "lobe_separation_km", "range_difference_km", "opposition_score", "banded_score",
        "neutral_center_score", "side_balance_score", "window_max_reflectivity_dbz",
        "storm_motion_u_kt", "storm_motion_v_kt", "storm_motion_speed_kt",
        "storm_motion_method", "base_confidence_score", "continuity_bonus",
        "track_adjusted_score", "distance_from_previous_main_km", "is_main_candidate", "confidence_score",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    return out


def plot(path, srv, x, y, candidates, motion):
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_OUT_DIR / f"{path.stem}_sweep{SWEEP}_banded_srv_tvs_candidates.png"

    plt.figure(figsize=(14, 12))
    norm = TwoSlopeNorm(vmin=-45.0, vcenter=0.0, vmax=45.0)
    mesh = plt.pcolormesh(x, y, srv, shading="auto", cmap=eve_velocity_colormap(), norm=norm)
    plt.colorbar(mesh, label="Storm-relative velocity (kt)")
    plt.title(
        f"{RADAR_SITE} Banded Storm-Relative TVS Candidates | {path.stem} | Sweep {SWEEP}\n"
        f"storm motion: u={motion['u_kt']:.1f} kt, v={motion['v_kt']:.1f} kt, speed={motion['speed_kt']:.1f} kt"
    )
    plt.xlabel("East West distance from radar (km)")
    plt.ylabel("North South distance from radar (km)")
    plt.xlim(-250, 250)
    plt.ylim(-250, 250)
    plt.gca().set_aspect("equal", adjustable="box")

    for i, c in enumerate(candidates, start=1):
        plt.scatter(c["center_x_km"], c["center_y_km"], marker="*", s=280)
        plt.text(c["center_x_km"], c["center_y_km"], str(i), fontsize=10)
        plt.plot(
            [c["positive_lobe_x_km"], c["negative_lobe_x_km"]],
            [c["positive_lobe_y_km"], c["negative_lobe_y_km"]],
            linewidth=1.0,
        )

    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def process(path, prev_centroid, prev_time, prev_motion, prev_main, prev_main_age):
    print(f"\nProcessing {path.name}")

    t = scan_time(path)
    radar = pyart.io.read(str(path))
    fields = set(radar.fields.keys())

    vname = field_name(fields, ["velocity", "corrected_velocity", "VEL"])
    rname = field_name(fields, ["reflectivity", "corrected_reflectivity", "DBZ"])

    if not vname:
        raise ValueError("No velocity field found")

    base_mps, x, y, rng, rng_mi, lats, lons = sweep_arrays(radar, vname)
    refl = sweep_arrays(radar, rname)[0] if rname else None

    curr_centroid = echo_centroid(refl, x, y, rng_mi)
    motion = storm_motion(prev_centroid, curr_centroid, prev_time, t, prev_motion)
    srv = storm_relative(base_mps, x, y, rng, motion)

    candidates, raw_count = detect(path, t, srv, refl, x, y, rng, rng_mi, lats, lons, motion)
    candidates = apply_continuity_ranking(candidates, prev_main, prev_main_age)

    csv_path = save_candidates(path, candidates)
    fig_path = plot(path, srv, x, y, candidates, motion)

    top = candidates[0] if candidates else None

    print(
        f"  storm motion={motion['speed_kt']:.1f} kt ({motion['method']}), "
        f"raw={raw_count}, final={len(candidates)}"
    )

    if top:
        print(
            f"  main score={top['track_adjusted_score']:.1f} "
            f"(base={top['base_confidence_score']:.1f}, continuity={top['continuity_bonus']:.1f}) | "
            f"center=({top['center_x_km']:.1f}, {top['center_y_km']:.1f})"
        )

    row = {
        **scan_metadata(path, t),
        "candidate_count": len(candidates),
        "raw_candidate_count": raw_count,
        "storm_motion_u_kt": motion["u_kt"],
        "storm_motion_v_kt": motion["v_kt"],
        "storm_motion_speed_kt": motion["speed_kt"],
        "storm_motion_method": motion["method"],
        "top_confidence_score": top["confidence_score"] if top else 0.0,
        "top_base_confidence_score": top["base_confidence_score"] if top else 0.0,
        "top_continuity_bonus": top["continuity_bonus"] if top else 0.0,
        "top_track_adjusted_score": top["track_adjusted_score"] if top else 0.0,
        "top_distance_from_previous_main_km": top["distance_from_previous_main_km"] if top else "",
        "top_delta_v_kt": top["delta_v_kt"] if top else 0.0,
        "top_center_x_km": top["center_x_km"] if top else "",
        "top_center_y_km": top["center_y_km"] if top else "",
        "csv_path": str(csv_path),
        "figure_path": str(fig_path),
        "status": "ok",
        "message": "",
    }

    return row, curr_centroid, t, motion, top



def save_event_times(rows):
    """Write one time-step row per radar scan for the API/frontend time slider."""
    LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id", "radar_site", "date_folder", "radar_file",
        "scan_time", "scan_time_utc", "sweep", "sort_order",
    ]

    cleaned = []
    seen = set()
    for row in rows:
        key = (row.get("event_id"), row.get("scan_time"), row.get("sweep"))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)

    cleaned.sort(key=lambda row: (row.get("event_id", ""), row.get("scan_time", ""), row.get("sweep", 0)))

    with EVENT_TIMES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)

    return EVENT_TIMES_CSV


def save_summary(rows):
    LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_OUT_DIR / f"{RADAR_SITE}_{DATE_FOLDER}_banded_srv_tvs_batch_summary.csv"

    fields = [
        "event_id", "radar_site", "date_folder", "radar_file", "scan_time", "scan_time_utc", "sweep",
        "candidate_count", "raw_candidate_count",
        "storm_motion_u_kt", "storm_motion_v_kt", "storm_motion_speed_kt", "storm_motion_method",
        "top_confidence_score", "top_base_confidence_score", "top_continuity_bonus",
        "top_track_adjusted_score", "top_distance_from_previous_main_km",
        "top_delta_v_kt", "top_center_x_km", "top_center_y_km",
        "csv_path", "figure_path", "status", "message",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return out




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect E-V-E storm-relative banded TVS candidates for one case.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID", EVENT_ID))
    parser.add_argument("--radar-site", default=os.getenv("RADAR_SITE"))
    parser.add_argument("--date-folder", default=os.getenv("DATE_FOLDER"))
    parser.add_argument("--sweep", type=int, default=int(os.getenv("SWEEP", str(SWEEP))))
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing candidate/log/figure outputs.")
    return parser.parse_args()


def reset_detection_outputs() -> None:
    for folder in [CANDIDATE_OUT_DIR, LOG_OUT_DIR, FIGURE_OUT_DIR]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    case = resolve_case(args.event_id, radar_site=args.radar_site, date_folder=args.date_folder, sweep=args.sweep)
    configure_case(case.event_id, case.radar_site, case.date_folder, case.sweep)

    if not args.keep_existing:
        reset_detection_outputs()

    for d in [FIGURE_OUT_DIR, CANDIDATE_OUT_DIR, LOG_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    event_time_rows = []
    prev_centroid = None
    prev_time = None
    prev_motion = None
    prev_main = None
    prev_main_age = 999

    print("=" * 120)
    print("E-V-E — Banded Storm-Relative TVS Candidate Detection + Continuity Ranking")
    print("=" * 120)
    print(f"Case: {EVENT_ID} | {RADAR_SITE} | {DATE_FOLDER} | sweep={SWEEP}")
    print(f"Radar input: {RADAR_DIR}")
    print(f"Candidate output: {CANDIDATE_OUT_DIR}")
    print("Velocity display: centered green/gray/red, no purple")

    for scan_order, path in enumerate(radar_files(), start=1):
        parsed_time = scan_time(path)
        event_time_rows.append({
            **scan_metadata(path, parsed_time),
            "sort_order": scan_order,
        })

        try:
            row, prev_centroid, prev_time, prev_motion, current_main = process(
                path,
                prev_centroid,
                prev_time,
                prev_motion,
                prev_main,
                prev_main_age,
            )

            if current_main is not None:
                prev_main = current_main
                prev_main_age = 0
            else:
                prev_main_age += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            prev_main_age += 1
            row = {
                **scan_metadata(path, scan_time(path)),
                "candidate_count": 0,
                "raw_candidate_count": 0,
                "storm_motion_u_kt": 0.0,
                "storm_motion_v_kt": 0.0,
                "storm_motion_speed_kt": 0.0,
                "storm_motion_method": "error",
                "top_confidence_score": 0.0,
                "top_base_confidence_score": 0.0,
                "top_continuity_bonus": 0.0,
                "top_track_adjusted_score": 0.0,
                "top_distance_from_previous_main_km": "",
                "top_delta_v_kt": 0.0,
                "top_center_x_km": "",
                "top_center_y_km": "",
                "csv_path": "",
                "figure_path": "",
                "status": "error",
                "message": str(e),
            }

        rows.append(row)

    event_times = save_event_times(event_time_rows)
    summary = save_summary(rows)

    print("\nSummary:")
    for r in sorted(rows, key=lambda row: row["top_track_adjusted_score"], reverse=True):
        print(
            f"{r['radar_file']} | final={r['candidate_count']} | raw={r['raw_candidate_count']} | "
            f"track_score={float(r['top_track_adjusted_score']):.1f} | "
            f"base={float(r['top_base_confidence_score']):.1f} | "
            f"cont={float(r['top_continuity_bonus']):.1f} | "
            f"center=({r['top_center_x_km']}, {r['top_center_y_km']})"
        )

    print(f"\nSaved event times: {event_times}")
    print(f"Saved summary: {summary}")


if __name__ == "__main__":
    main()
