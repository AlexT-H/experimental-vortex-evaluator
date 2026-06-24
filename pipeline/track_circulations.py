from pathlib import Path
import argparse
import os
import shutil
import csv
import json
import math
import re
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from case_config import case_dir, resolve_case


# =============================================================================
# E-V-E / Vortex Lab
# Phase 5 — Track Building + Quality Ranking + 15/30 Minute Nowcasting
#
# Purpose:
#   Read banded storm-relative TVS candidate CSVs, link detections into tracks,
#   score/rank the tracks by vortex-track quality, and export 15/30 minute
#   nowcast paths.
#
# Important:
#   This version does NOT reject tracks just because they do not move with the
#   parent storm motion. Vortices can deviate from broader storm motion.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = os.getenv("EVENT_ID", "test_case_1")
INPUT_PATTERN = "*_banded_srv_tvs_candidates.csv"


def configure_case(event_id: str, radar_site: str | None = None, date_folder: str | None = None) -> None:
    """
    Point tracker outputs at this event's isolated case folder.

    Tracking, quality scoring, ranking, and nowcast calculations are preserved
    from the supplied correct quality-ranked tracker.
    """
    global EVENT_ID, CASE_META, CASE_OUT_DIR, CANDIDATE_DIR, TRACK_OUT_DIR, NOWCAST_OUT_DIR, GEOJSON_OUT_DIR, FIGURE_OUT_DIR
    global TRACK_POINTS_CSV, TRACKS_CSV, HIGH_QUALITY_TRACKS_CSV, BEST_TRACK_CSV, BEST_TRACK_POINTS_CSV
    global NOWCASTS_CSV, HIGH_QUALITY_NOWCASTS_CSV, BEST_NOWCASTS_CSV
    global TRACK_POINTS_GEOJSON, TRACKS_GEOJSON, NOWCASTS_GEOJSON
    global HIGH_QUALITY_TRACKS_GEOJSON, HIGH_QUALITY_NOWCASTS_GEOJSON
    global BEST_TRACK_GEOJSON, BEST_TRACK_POINTS_GEOJSON, BEST_NOWCASTS_GEOJSON, TRACK_FIGURE

    CASE_META = resolve_case(event_id, radar_site=radar_site, date_folder=date_folder)
    EVENT_ID = CASE_META.event_id
    CASE_OUT_DIR = case_dir(EVENT_ID, PROJECT_ROOT)

    CANDIDATE_DIR = CASE_OUT_DIR / "candidates"
    TRACK_OUT_DIR = CASE_OUT_DIR / "tracks"
    NOWCAST_OUT_DIR = CASE_OUT_DIR / "nowcasts"
    GEOJSON_OUT_DIR = CASE_OUT_DIR / "geojson"
    FIGURE_OUT_DIR = CASE_OUT_DIR / "figures"

    TRACK_POINTS_CSV = TRACK_OUT_DIR / "banded_tvs_track_points.csv"
    TRACKS_CSV = TRACK_OUT_DIR / "banded_tvs_tracks.csv"
    HIGH_QUALITY_TRACKS_CSV = TRACK_OUT_DIR / "high_quality_banded_tvs_tracks.csv"
    BEST_TRACK_CSV = TRACK_OUT_DIR / "best_banded_tvs_track.csv"
    BEST_TRACK_POINTS_CSV = TRACK_OUT_DIR / "best_banded_tvs_track_points.csv"

    NOWCASTS_CSV = NOWCAST_OUT_DIR / "banded_tvs_nowcasts_15_30.csv"
    HIGH_QUALITY_NOWCASTS_CSV = NOWCAST_OUT_DIR / "high_quality_banded_tvs_nowcasts_15_30.csv"
    BEST_NOWCASTS_CSV = NOWCAST_OUT_DIR / "best_banded_tvs_nowcasts_15_30.csv"

    TRACK_POINTS_GEOJSON = GEOJSON_OUT_DIR / "banded_tvs_track_points.geojson"
    TRACKS_GEOJSON = GEOJSON_OUT_DIR / "banded_tvs_tracks.geojson"
    NOWCASTS_GEOJSON = GEOJSON_OUT_DIR / "banded_tvs_nowcasts_15_30.geojson"

    HIGH_QUALITY_TRACKS_GEOJSON = GEOJSON_OUT_DIR / "high_quality_banded_tvs_tracks.geojson"
    HIGH_QUALITY_NOWCASTS_GEOJSON = GEOJSON_OUT_DIR / "high_quality_banded_tvs_nowcasts_15_30.geojson"

    BEST_TRACK_GEOJSON = GEOJSON_OUT_DIR / "best_banded_tvs_track.geojson"
    BEST_TRACK_POINTS_GEOJSON = GEOJSON_OUT_DIR / "best_banded_tvs_track_points.geojson"
    BEST_NOWCASTS_GEOJSON = GEOJSON_OUT_DIR / "best_banded_tvs_nowcasts_15_30.geojson"

    # Keep the original correct filename inside each isolated case folder.
    TRACK_FIGURE = FIGURE_OUT_DIR / "banded_tvs_tracks_quality_ranked_nowcasts_15_30.png"


configure_case(EVENT_ID)

# Link-building thresholds.
MAX_TIME_GAP_MIN = 16.0
MAX_TRACK_SPEED_KT = 90.0
MIN_LINK_DISTANCE_KM = 10.0
MAX_LINK_DISTANCE_KM = 30.0
VELOCITY_SMOOTHING = 0.55

# Nowcast policy.
NOWCAST_MINUTES = [15, 30]
NOWCAST_MIN_TRACK_DETECTIONS = 2
NOWCAST_MIN_SPEED_KT = 5.0

# Quality ranking policy.
HIGH_QUALITY_MIN_SCORE = 65.0
MIN_TRACK_DETECTIONS_FOR_LINE = 2


def parse_scan_time(value, radar_file=None):
    if value and not pd.isna(value):
        try:
            return pd.to_datetime(value, utc=True).to_pydatetime()
        except Exception:
            pass

    if radar_file:
        match = re.search(r"[A-Z]{4}(\d{8})_(\d{6})", str(radar_file))
        if match:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

    return None


def bearing_from_vector(vx_km_min, vy_km_min):
    if vx_km_min == 0 and vy_km_min == 0:
        return 0.0
    return math.degrees(math.atan2(vx_km_min, vy_km_min)) % 360.0


def speed_kt_from_vector(vx_km_min, vy_km_min):
    km_per_hour = math.hypot(vx_km_min, vy_km_min) * 60.0
    return km_per_hour / 1.852


def point_distance_km(a, b):
    return math.hypot(float(a["center_x_km"]) - float(b["center_x_km"]),
                      float(a["center_y_km"]) - float(b["center_y_km"]))


def add_latlon_offset(lat, lon, dx_km, dy_km):
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None, None

    lat = float(lat)
    lon = float(lon)

    lat2 = lat + (dy_km / 111.32)
    lon2 = lon + (dx_km / (111.32 * math.cos(math.radians(lat))))
    return lat2, lon2


def angle_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def load_candidates():
    files = sorted(CANDIDATE_DIR.glob(INPUT_PATTERN))

    if not files:
        raise FileNotFoundError(f"No candidate CSVs found at {CANDIDATE_DIR / INPUT_PATTERN}")

    frames = []

    for path in files:
        df = pd.read_csv(path)

        if df.empty:
            continue

        df["source_csv"] = str(path)
        df["candidate_row"] = range(len(df))

        if "radar_file" not in df.columns:
            df["radar_file"] = path.name

        if "scan_time_utc" not in df.columns:
            df["scan_time_utc"] = df["scan_time"] if "scan_time" in df.columns else ""

        if "event_id" not in df.columns:
            df["event_id"] = EVENT_ID

        if "radar_site" not in df.columns:
            df["radar_site"] = [infer_radar_site(value) for value in df["radar_file"]]

        if "date_folder" not in df.columns:
            df["date_folder"] = ""

        if "sweep" not in df.columns:
            df["sweep"] = ""

        for col in ["storm_motion_u_kt", "storm_motion_v_kt", "storm_motion_speed_kt"]:
            if col not in df.columns:
                df[col] = 0.0

        df["scan_time"] = [
            parse_scan_time(value, radar_file)
            for value, radar_file in zip(df["scan_time_utc"], df["radar_file"])
        ]

        frames.append(df)

    if not frames:
        raise ValueError("Candidate CSV files were found, but all were empty.")

    candidates = pd.concat(frames, ignore_index=True)

    # Case-safety filter. This prevents stale/copied candidates from another
    # event, radar, or date from leaking into this event. It does not change
    # link-building, scoring, ranking, or nowcast calculations.
    for col in ["event_id", "radar_site", "date_folder"]:
        if col in candidates.columns:
            candidates[col] = candidates[col].astype(str).str.strip()

    before_filter = len(candidates)
    candidates = candidates[
        (candidates["event_id"] == EVENT_ID)
        & (candidates["radar_site"] == CASE_META.radar_site)
        & (candidates["date_folder"] == CASE_META.date_folder)
    ].copy()
    dropped_rows = before_filter - len(candidates)

    if candidates.empty:
        raise ValueError(
            "No candidate rows matched the selected case after safety filtering. "
            f"Expected event_id={EVENT_ID}, radar_site={CASE_META.radar_site}, "
            f"date_folder={CASE_META.date_folder}. Check {CANDIDATE_DIR}."
        )

    if dropped_rows:
        print(f"Case-safety filter dropped {dropped_rows} wrong-case candidate rows.")

    required = [
        "center_x_km",
        "center_y_km",
        "latitude",
        "longitude",
        "confidence_score",
        "delta_v_kt",
        "scan_time",
    ]

    missing = [col for col in required if col not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate CSVs are missing required columns: {missing}")

    candidates = candidates.dropna(subset=["scan_time", "center_x_km", "center_y_km"])
    candidates = candidates.sort_values(["scan_time", "confidence_score"], ascending=[True, False]).reset_index(drop=True)
    candidates["detection_id"] = [f"DET_{i:04d}" for i in range(1, len(candidates) + 1)]

    return candidates


class Track:
    def __init__(self, track_id, first_detection):
        self.track_id = track_id
        self.points = []
        self.vx_km_min = 0.0
        self.vy_km_min = 0.0
        self.add(first_detection)

    @property
    def last(self):
        return self.points[-1]

    @property
    def last_time(self):
        return self.last["scan_time"]

    def predict_xy(self, scan_time):
        dt_min = (scan_time - self.last_time).total_seconds() / 60.0
        return (
            float(self.last["center_x_km"]) + self.vx_km_min * dt_min,
            float(self.last["center_y_km"]) + self.vy_km_min * dt_min,
        )

    def add(self, detection):
        if self.points:
            prev = self.points[-1]
            dt_min = (detection["scan_time"] - prev["scan_time"]).total_seconds() / 60.0

            if dt_min > 0:
                new_vx = (float(detection["center_x_km"]) - float(prev["center_x_km"])) / dt_min
                new_vy = (float(detection["center_y_km"]) - float(prev["center_y_km"])) / dt_min

                if len(self.points) == 1:
                    self.vx_km_min = new_vx
                    self.vy_km_min = new_vy
                else:
                    self.vx_km_min = VELOCITY_SMOOTHING * self.vx_km_min + (1.0 - VELOCITY_SMOOTHING) * new_vx
                    self.vy_km_min = VELOCITY_SMOOTHING * self.vy_km_min + (1.0 - VELOCITY_SMOOTHING) * new_vy

        row = dict(detection)
        row["track_id"] = self.track_id
        self.points.append(row)


def allowed_link_distance_km(dt_min):
    speed_based = MAX_TRACK_SPEED_KT * 1.852 * dt_min / 60.0
    return min(MAX_LINK_DISTANCE_KM, max(MIN_LINK_DISTANCE_KM, speed_based))


def build_tracks(candidates):
    tracks = []
    next_track_num = 1

    for scan_time, group in candidates.groupby("scan_time", sort=True):
        detections = group.sort_values("confidence_score", ascending=False).to_dict("records")

        active_tracks = [
            track for track in tracks
            if 0 < (scan_time - track.last_time).total_seconds() / 60.0 <= MAX_TIME_GAP_MIN
        ]

        possible_links = []

        for track in active_tracks:
            dt_min = (scan_time - track.last_time).total_seconds() / 60.0
            max_dist = allowed_link_distance_km(dt_min)
            pred_x, pred_y = track.predict_xy(scan_time)

            for det_idx, det in enumerate(detections):
                dist = math.hypot(float(det["center_x_km"]) - pred_x, float(det["center_y_km"]) - pred_y)

                if dist <= max_dist:
                    quality_bonus = float(det.get("confidence_score", 0.0)) / 100.0
                    possible_links.append((dist - quality_bonus, dist, track, det_idx))

        possible_links.sort(key=lambda item: item[0])

        used_tracks = set()
        used_detections = set()

        for _, _, track, det_idx in possible_links:
            if track.track_id in used_tracks or det_idx in used_detections:
                continue

            track.add(detections[det_idx])
            used_tracks.add(track.track_id)
            used_detections.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in used_detections:
                continue

            track_id = f"TRK_{next_track_num:03d}"
            next_track_num += 1
            tracks.append(Track(track_id, det))

    return tracks


def step_metrics(points):
    step_distances = []
    step_speeds = []
    bearings = []

    for a, b in zip(points[:-1], points[1:]):
        dt_min = (b["scan_time"] - a["scan_time"]).total_seconds() / 60.0
        if dt_min <= 0:
            continue

        dx = float(b["center_x_km"]) - float(a["center_x_km"])
        dy = float(b["center_y_km"]) - float(a["center_y_km"])
        dist = math.hypot(dx, dy)

        step_distances.append(dist)
        step_speeds.append((dist / dt_min) * 60.0 / 1.852)
        bearings.append(bearing_from_vector(dx / dt_min, dy / dt_min))

    turn_angles = []
    for a, b in zip(bearings[:-1], bearings[1:]):
        turn_angles.append(angle_diff_deg(a, b))

    return step_distances, step_speeds, bearings, turn_angles


def score_track_quality(summary, step_distances, step_speeds, turn_angles):
    detection_count = int(summary["detection_count"])
    duration_min = float(summary["duration_min"])
    mean_conf = float(summary["mean_confidence_score"])
    max_conf = float(summary["max_confidence_score"])
    mean_delta = float(summary["mean_delta_v_kt"])
    max_delta = float(summary["max_delta_v_kt"])
    speed = float(summary["speed_kt"])

    persistence_score = 18.0 * clamp((detection_count - 1) / 5.0)
    duration_score = 14.0 * clamp(duration_min / 32.0)

    confidence_score = 18.0 * clamp((mean_conf - 60.0) / 35.0)
    peak_confidence_score = 6.0 * clamp((max_conf - 70.0) / 25.0)

    delta_score = 15.0 * clamp((mean_delta - 55.0) / 45.0)
    peak_delta_score = 6.0 * clamp((max_delta - 70.0) / 45.0)

    # Real vortex tracks usually have meaningful motion but not absurd displacement.
    # The ideal band is broad, because storm/circulation motion can vary.
    if speed <= 5.0:
        speed_score = 2.0
    elif speed <= 20.0:
        speed_score = 8.0 + 4.0 * clamp((speed - 5.0) / 15.0)
    elif speed <= 55.0:
        speed_score = 12.0
    elif speed <= 80.0:
        speed_score = 12.0 * clamp((80.0 - speed) / 25.0)
    else:
        speed_score = 0.0

    if len(step_distances) >= 2:
        mean_step = sum(step_distances) / len(step_distances)
        max_step = max(step_distances)
        step_variance = sum((d - mean_step) ** 2 for d in step_distances) / len(step_distances)
        step_std = math.sqrt(step_variance)
        step_cv = step_std / mean_step if mean_step > 0 else 1.0
        step_consistency_score = 8.0 * clamp(1.0 - step_cv)
        max_step_penalty = 14.0 * clamp((max_step - 28.0) / 25.0)
    else:
        step_consistency_score = 3.0 if detection_count >= 2 else 0.0
        max_step_penalty = 0.0

    if turn_angles:
        mean_turn = sum(turn_angles) / len(turn_angles)
        smoothness_score = 12.0 * clamp(1.0 - mean_turn / 130.0)
        erratic_penalty = 10.0 * clamp((mean_turn - 95.0) / 85.0)
    else:
        smoothness_score = 5.0 if detection_count >= 2 else 0.0
        erratic_penalty = 0.0

    single_detection_penalty = 25.0 if detection_count < 2 else 0.0
    weak_track_penalty = 10.0 if detection_count == 2 and duration_min < 6.0 else 0.0

    raw_score = (
        persistence_score
        + duration_score
        + confidence_score
        + peak_confidence_score
        + delta_score
        + peak_delta_score
        + speed_score
        + step_consistency_score
        + smoothness_score
        - max_step_penalty
        - erratic_penalty
        - single_detection_penalty
        - weak_track_penalty
    )

    score = round(clamp(raw_score, 0.0, 100.0), 2)

    return {
        "track_quality_score": score,
        "quality_persistence_score": round(persistence_score, 2),
        "quality_duration_score": round(duration_score, 2),
        "quality_confidence_score": round(confidence_score + peak_confidence_score, 2),
        "quality_delta_v_score": round(delta_score + peak_delta_score, 2),
        "quality_speed_score": round(speed_score, 2),
        "quality_smoothness_score": round(step_consistency_score + smoothness_score, 2),
        "quality_penalty": round(max_step_penalty + erratic_penalty + single_detection_penalty + weak_track_penalty, 2),
    }


def summarize_track(track):
    points = sorted(track.points, key=lambda p: p["scan_time"])
    start = points[0]
    end = points[-1]

    duration_min = (end["scan_time"] - start["scan_time"]).total_seconds() / 60.0

    path_length_km = 0.0
    for a, b in zip(points[:-1], points[1:]):
        path_length_km += point_distance_km(a, b)

    if len(points) >= 2 and duration_min > 0:
        overall_vx = (float(end["center_x_km"]) - float(start["center_x_km"])) / duration_min
        overall_vy = (float(end["center_y_km"]) - float(start["center_y_km"])) / duration_min
    else:
        overall_vx = 0.0
        overall_vy = 0.0

    speed_kt = speed_kt_from_vector(track.vx_km_min, track.vy_km_min)
    bearing_deg = bearing_from_vector(track.vx_km_min, track.vy_km_min)

    if speed_kt < NOWCAST_MIN_SPEED_KT and len(points) >= 2:
        speed_kt = speed_kt_from_vector(overall_vx, overall_vy)
        bearing_deg = bearing_from_vector(overall_vx, overall_vy)

    step_distances, step_speeds, bearings, turn_angles = step_metrics(points)

    confidence_values = [float(p.get("confidence_score", 0.0)) for p in points]
    delta_values = [float(p.get("delta_v_kt", 0.0)) for p in points]

    summary = {
        "event_id": start.get("event_id", EVENT_ID),
        "radar_site": start.get("radar_site", CASE_META.radar_site),
        "date_folder": start.get("date_folder", CASE_META.date_folder),
        "track_id": track.track_id,
        "detection_count": len(points),
        "start_time_utc": start["scan_time"].isoformat(),
        "end_time_utc": end["scan_time"].isoformat(),
        "duration_min": round(duration_min, 2),
        "persistence_min": round(duration_min, 2),
        "path_length_km": round(path_length_km, 3),
        "speed_kt": round(speed_kt, 2),
        "bearing_deg": round(bearing_deg, 2),
        "motion_vx_km_min": round(track.vx_km_min, 5),
        "motion_vy_km_min": round(track.vy_km_min, 5),
        "mean_step_distance_km": round(sum(step_distances) / len(step_distances), 3) if step_distances else 0.0,
        "max_step_distance_km": round(max(step_distances), 3) if step_distances else 0.0,
        "mean_step_speed_kt": round(sum(step_speeds) / len(step_speeds), 2) if step_speeds else 0.0,
        "max_step_speed_kt": round(max(step_speeds), 2) if step_speeds else 0.0,
        "mean_turn_angle_deg": round(sum(turn_angles) / len(turn_angles), 2) if turn_angles else 0.0,
        "max_turn_angle_deg": round(max(turn_angles), 2) if turn_angles else 0.0,
        "start_x_km": start["center_x_km"],
        "start_y_km": start["center_y_km"],
        "end_x_km": end["center_x_km"],
        "end_y_km": end["center_y_km"],
        "start_latitude": start["latitude"],
        "start_longitude": start["longitude"],
        "end_latitude": end["latitude"],
        "end_longitude": end["longitude"],
        "mean_confidence_score": round(sum(confidence_values) / len(confidence_values), 2),
        "max_confidence_score": round(max(confidence_values), 2),
        "mean_delta_v_kt": round(sum(delta_values) / len(delta_values), 2),
        "max_delta_v_kt": round(max(delta_values), 2),
    }

    summary.update(score_track_quality(summary, step_distances, step_speeds, turn_angles))

    summary["track_quality_label"] = (
        "high" if summary["track_quality_score"] >= HIGH_QUALITY_MIN_SCORE else
        "medium" if summary["track_quality_score"] >= 45.0 else
        "low"
    )

    summary["projectable"] = (
        len(points) >= NOWCAST_MIN_TRACK_DETECTIONS
        and speed_kt >= NOWCAST_MIN_SPEED_KT
        and summary["track_quality_label"] in {"high", "medium"}
    )

    return summary


def rank_tracks(track_summaries):
    ranked = sorted(
        track_summaries,
        key=lambda row: (
            row["track_quality_score"],
            row["detection_count"],
            row["duration_min"],
            row["max_confidence_score"],
            row["max_delta_v_kt"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(ranked, start=1):
        row["track_quality_rank"] = rank
        row["ranked_track_name"] = f"RANK_{rank:02d}_{row['track_id']}"

    return ranked


def flatten_track_points(tracks, summary_lookup):
    rows = []

    for track in tracks:
        summary = summary_lookup[track.track_id]

        for order, point in enumerate(sorted(track.points, key=lambda p: p["scan_time"]), start=1):
            row = dict(point)
            row["event_id"] = row.get("event_id", EVENT_ID)
            row["radar_site"] = row.get("radar_site", CASE_META.radar_site)
            row["date_folder"] = row.get("date_folder", CASE_META.date_folder)
            row["track_order"] = order
            row["scan_time_utc"] = point["scan_time"].isoformat()
            row["track_quality_rank"] = summary["track_quality_rank"]
            row["ranked_track_name"] = summary["ranked_track_name"]
            row["track_quality_score"] = summary["track_quality_score"]
            row["track_quality_label"] = summary["track_quality_label"]
            row.pop("scan_time", None)
            rows.append(row)

    return rows


def make_nowcasts(tracks, summary_lookup, allowed_labels=None):
    rows = []

    for track in tracks:
        summary = summary_lookup[track.track_id]

        if allowed_labels is not None and summary["track_quality_label"] not in allowed_labels:
            continue

        if not summary["projectable"]:
            continue

        points = sorted(track.points, key=lambda p: p["scan_time"])
        last = points[-1]

        vx = track.vx_km_min
        vy = track.vy_km_min

        if speed_kt_from_vector(vx, vy) < NOWCAST_MIN_SPEED_KT and len(points) >= 2:
            first = points[0]
            duration_min = (last["scan_time"] - first["scan_time"]).total_seconds() / 60.0
            if duration_min > 0:
                vx = (float(last["center_x_km"]) - float(first["center_x_km"])) / duration_min
                vy = (float(last["center_y_km"]) - float(first["center_y_km"])) / duration_min

        speed_kt = speed_kt_from_vector(vx, vy)
        bearing_deg = bearing_from_vector(vx, vy)

        for minutes in NOWCAST_MINUTES:
            dx = vx * minutes
            dy = vy * minutes

            projected_x = float(last["center_x_km"]) + dx
            projected_y = float(last["center_y_km"]) + dy
            projected_lat, projected_lon = add_latlon_offset(last["latitude"], last["longitude"], dx, dy)

            rows.append(
                {
                    "event_id": last.get("event_id", summary.get("event_id", EVENT_ID)),
                    "radar_site": last.get("radar_site", summary.get("radar_site", CASE_META.radar_site)),
                    "date_folder": last.get("date_folder", summary.get("date_folder", CASE_META.date_folder)),
                    "track_id": track.track_id,
                    "track_quality_rank": summary["track_quality_rank"],
                    "ranked_track_name": summary["ranked_track_name"],
                    "track_quality_score": summary["track_quality_score"],
                    "track_quality_label": summary["track_quality_label"],
                    "issued_time_utc": last["scan_time"].isoformat(),
                    "projection_min": minutes,
                    "valid_time_utc": (last["scan_time"] + timedelta(minutes=minutes)).isoformat(),
                    "start_x_km": last["center_x_km"],
                    "start_y_km": last["center_y_km"],
                    "end_x_km": projected_x,
                    "end_y_km": projected_y,
                    "start_latitude": last["latitude"],
                    "start_longitude": last["longitude"],
                    "end_latitude": projected_lat,
                    "end_longitude": projected_lon,
                    "speed_kt": round(speed_kt, 2),
                    "bearing_deg": round(bearing_deg, 2),
                    "source_detection_id": last.get("detection_id", ""),
                    "source_radar_file": last.get("radar_file", ""),
                }
            )

    rows.sort(key=lambda row: (row["track_quality_rank"], row["projection_min"]))
    return rows


def clean_json_value(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def track_points_geojson(rows):
    features = []

    for row in rows:
        lat = row.get("latitude")
        lon = row.get("longitude")

        if pd.isna(lat) or pd.isna(lon):
            continue

        props = {k: clean_json_value(v) for k, v in row.items() if k not in {"latitude", "longitude"}}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            }
        )

    return feature_collection(features)


def tracks_geojson(track_summaries, tracks):
    summary_lookup = {row["track_id"]: row for row in track_summaries}
    features = []

    for track in tracks:
        summary = summary_lookup.get(track.track_id)
        if summary is None:
            continue
        if len(track.points) < MIN_TRACK_DETECTIONS_FOR_LINE:
            continue

        coords = []
        for point in sorted(track.points, key=lambda p: p["scan_time"]):
            if not pd.isna(point.get("longitude")) and not pd.isna(point.get("latitude")):
                coords.append([float(point["longitude"]), float(point["latitude"])])

        if len(coords) < 2:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {k: clean_json_value(v) for k, v in summary.items()},
            }
        )

    return feature_collection(features)


def nowcasts_geojson(rows):
    features = []

    for row in rows:
        if pd.isna(row.get("start_latitude")) or pd.isna(row.get("start_longitude")):
            continue
        if row.get("end_latitude") is None or row.get("end_longitude") is None:
            continue

        coords = [
            [float(row["start_longitude"]), float(row["start_latitude"])],
            [float(row["end_longitude"]), float(row["end_latitude"])],
        ]

        props = {
            k: clean_json_value(v)
            for k, v in row.items()
            if k not in {"start_latitude", "start_longitude", "end_latitude", "end_longitude"}
        }

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )

    return feature_collection(features)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def track_red_color(rank, max_rank):
    """
    Return a red shade where rank 1 is darkest and lower-ranked tracks are lighter.

    This keeps the original red-shade formula, then clamps the RGB value so
    Matplotlib cannot crash when there are many ranked tracks.
    """
    if max_rank <= 1:
        shade = 0.50
    else:
        rank_position = (rank - 1) / (max_rank - 1)
        shade = 0.50 + 0.90 * rank_position

    shade = clamp(shade, 0.0, 0.95)
    return (shade, 0.0, 0.0)


def plot_tracks(track_summaries, tracks, nowcasts):
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lookup = {row["track_id"]: row for row in track_summaries}
    max_rank = max([row["track_quality_rank"] for row in track_summaries], default=1)

    projection_pink = "#DA70D6"

    plt.figure(figsize=(12, 10))

    for track in tracks:
        summary = summary_lookup[track.track_id]
        if len(track.points) < 2:
            continue

        points = sorted(track.points, key=lambda p: p["scan_time"])
        xs = [float(p["center_x_km"]) for p in points]
        ys = [float(p["center_y_km"]) for p in points]

        rank = int(summary["track_quality_rank"])
        line_color = track_red_color(rank, max_rank)

        width = 2.4 if rank == 1 else 1.3
        marker_size = 6 if rank == 1 else 4

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=width,
            markersize=marker_size,
            color=line_color,
            markerfacecolor=line_color,
            markeredgecolor=line_color,
        )
        plt.text(xs[-1], ys[-1], summary["ranked_track_name"], fontsize=8, color=line_color)

    for row in nowcasts:
        width = 2.0 if row["track_quality_rank"] == 1 else 1.0
        plt.plot(
            [float(row["start_x_km"]), float(row["end_x_km"])],
            [float(row["start_y_km"]), float(row["end_y_km"])],
            linestyle="--",
            linewidth=width,
            color=projection_pink,
        )
        plt.text(
            float(row["end_x_km"]),
            float(row["end_y_km"]),
            f"{row['projection_min']}m",
            fontsize=8,
            color=projection_pink,
        )

    plt.title("Banded TVS Tracks Ranked by Vortex-Track Quality + 15/30 Minute Nowcasts")
    plt.xlabel("East/West distance from radar (km)")
    plt.ylabel("North/South distance from radar (km)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.savefig(TRACK_FIGURE, dpi=200, bbox_inches="tight")
    plt.close()




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build E-V-E quality-ranked circulation tracks and 15/30 minute nowcasts for one case."
    )
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID", EVENT_ID))
    parser.add_argument("--radar-site", default=os.getenv("RADAR_SITE"))
    parser.add_argument("--date-folder", default=os.getenv("DATE_FOLDER"))
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing tracker outputs.")
    return parser.parse_args()


def reset_tracking_outputs() -> None:
    for folder in [TRACK_OUT_DIR, NOWCAST_OUT_DIR, GEOJSON_OUT_DIR]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "banded_tvs_tracks_quality_ranked_nowcasts_15_30.png",
        f"{EVENT_ID}_banded_tvs_tracks_quality_ranked_nowcasts_15_30.png",
        "banded_tvs_tracks_nowcasts_15_30.png",
    ]:
        path = FIGURE_OUT_DIR / name
        if path.exists():
            path.unlink()


def main():
    args = parse_args()
    case = resolve_case(args.event_id, radar_site=args.radar_site, date_folder=args.date_folder)
    configure_case(case.event_id, radar_site=case.radar_site, date_folder=case.date_folder)

    if not args.keep_existing:
        reset_tracking_outputs()

    for folder in [TRACK_OUT_DIR, NOWCAST_OUT_DIR, GEOJSON_OUT_DIR, FIGURE_OUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates()

    print("=" * 120)
    print("E-V-E Phase 5 — Track Building + Quality Ranking + 15/30 Minute Nowcasting")
    print("=" * 120)
    print(f"Case: {EVENT_ID} | {CASE_META.radar_site} | {CASE_META.date_folder}")
    print(f"Candidate detections loaded: {len(candidates)}")
    print(f"Scans represented: {candidates['scan_time'].nunique()}")
    print(f"Input pattern: {CANDIDATE_DIR / INPUT_PATTERN}")

    tracks = build_tracks(candidates)

    summaries = [summarize_track(track) for track in tracks]
    ranked_summaries = rank_tracks(summaries)
    summary_lookup = {row["track_id"]: row for row in ranked_summaries}

    sorted_tracks = sorted(tracks, key=lambda t: summary_lookup[t.track_id]["track_quality_rank"])

    track_points = flatten_track_points(sorted_tracks, summary_lookup)
    all_nowcasts = make_nowcasts(sorted_tracks, summary_lookup, allowed_labels=None)
    high_quality_nowcasts = make_nowcasts(sorted_tracks, summary_lookup, allowed_labels={"high"})

    high_quality_tracks = [row for row in ranked_summaries if row["track_quality_label"] == "high"]
    best_track = ranked_summaries[:1]
    best_track_id = best_track[0]["track_id"] if best_track else None
    best_track_points = [row for row in track_points if row["track_id"] == best_track_id]
    best_nowcasts = [row for row in all_nowcasts if row["track_id"] == best_track_id]

    save_csv(TRACK_POINTS_CSV, track_points)
    save_csv(TRACKS_CSV, ranked_summaries)
    save_csv(HIGH_QUALITY_TRACKS_CSV, high_quality_tracks)
    save_csv(BEST_TRACK_CSV, best_track)
    save_csv(BEST_TRACK_POINTS_CSV, best_track_points)

    save_csv(NOWCASTS_CSV, all_nowcasts)
    save_csv(HIGH_QUALITY_NOWCASTS_CSV, high_quality_nowcasts)
    save_csv(BEST_NOWCASTS_CSV, best_nowcasts)

    write_geojson(TRACK_POINTS_GEOJSON, track_points_geojson(track_points))
    write_geojson(TRACKS_GEOJSON, tracks_geojson(ranked_summaries, sorted_tracks))
    write_geojson(NOWCASTS_GEOJSON, nowcasts_geojson(all_nowcasts))

    write_geojson(HIGH_QUALITY_TRACKS_GEOJSON, tracks_geojson(high_quality_tracks, sorted_tracks))
    write_geojson(HIGH_QUALITY_NOWCASTS_GEOJSON, nowcasts_geojson(high_quality_nowcasts))

    write_geojson(BEST_TRACK_POINTS_GEOJSON, track_points_geojson(best_track_points))
    write_geojson(BEST_TRACK_GEOJSON, tracks_geojson(best_track, sorted_tracks))
    write_geojson(BEST_NOWCASTS_GEOJSON, nowcasts_geojson(best_nowcasts))

    plot_tracks(ranked_summaries, sorted_tracks, all_nowcasts)

    print("\nRanked tracks:")
    for row in ranked_summaries:
        print(
            f"#{row['track_quality_rank']:02d} {row['track_id']} | "
            f"name={row['ranked_track_name']} | "
            f"quality={row['track_quality_score']} ({row['track_quality_label']}) | "
            f"detections={row['detection_count']} | duration={row['duration_min']} min | "
            f"speed={row['speed_kt']} kt | mean_conf={row['mean_confidence_score']} | "
            f"mean_delta={row['mean_delta_v_kt']}"
        )

    print("\nSaved outputs:")
    print(f"  Ranked tracks CSV:          {TRACKS_CSV}")
    print(f"  High-quality tracks CSV:    {HIGH_QUALITY_TRACKS_CSV}")
    print(f"  Best track CSV:             {BEST_TRACK_CSV}")
    print(f"  Best track points CSV:      {BEST_TRACK_POINTS_CSV}")
    print(f"  All nowcasts CSV:           {NOWCASTS_CSV}")
    print(f"  High-quality nowcasts CSV:  {HIGH_QUALITY_NOWCASTS_CSV}")
    print(f"  Best nowcasts CSV:          {BEST_NOWCASTS_CSV}")
    print(f"  Figure:                     {TRACK_FIGURE}")
    print(f"\nNowcast windows: {NOWCAST_MINUTES} minutes only")


if __name__ == "__main__":
    main()
