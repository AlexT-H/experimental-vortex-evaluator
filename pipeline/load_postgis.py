#!/usr/bin/env python3
"""
E-V-E — Load time-aware E-V-E outputs into PostGIS.

This version extends the previous PostGIS loader so the API/frontend can drive a
time slider.

It loads:
    outputs/logs/{event_id}_event_times.csv
    outputs/geojson/banded_tvs_track_points.geojson
    outputs/geojson/banded_tvs_tracks.geojson
    outputs/geojson/banded_tvs_nowcasts_15_30.geojson
    outputs/ml/track_model_predictions.csv
    outputs/web/radar_frames/{event_id}/radar_frames.csv

Main temporal fields:
    event_times.scan_time
    circulation_detections.scan_time
    signature_tracks.start_time
    signature_tracks.end_time
    signature_tracks.latest_scan_time
    nowcast_paths.source_scan_time
    radar_frames.scan_time
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
except ImportError as exc:
    raise ImportError(
        "Missing dependency: psycopg2.\n\n"
        "Add this to your pipeline requirements file:\n"
        "  psycopg2-binary\n\n"
        "Then rebuild:\n"
        "  docker compose build pipeline\n"
    ) from exc


ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

from case_config import case_dir, resolve_case


def default_case_dir(event_id: str) -> Path:
    return case_dir(event_id, ROOT)


def default_event_times_csv(event_id: str) -> Path:
    return default_case_dir(event_id) / "logs" / f"{event_id}_event_times.csv"


def default_track_points_geojson(event_id: str) -> Path:
    return default_case_dir(event_id) / "geojson" / "banded_tvs_track_points.geojson"


def default_tracks_geojson(event_id: str) -> Path:
    return default_case_dir(event_id) / "geojson" / "banded_tvs_tracks.geojson"


def default_nowcasts_geojson(event_id: str) -> Path:
    return default_case_dir(event_id) / "geojson" / "banded_tvs_nowcasts_15_30.geojson"


def default_ml_predictions_csv(event_id: str) -> Path:
    return default_case_dir(event_id) / "ml" / "track_model_predictions.csv"


def default_radar_frames_csv(event_id: str) -> Path:
    return ROOT / "outputs" / "web" / "radar_frames" / event_id / "radar_frames.csv"


def db_connect():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "eve"),
        user=os.getenv("POSTGRES_USER", "eve_user"),
        password=os.getenv("POSTGRES_PASSWORD", "eve_password"),
        host=os.getenv("POSTGRES_HOST", "postgis"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def load_geojson(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing GeoJSON, skipping: {path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected FeatureCollection in {path}")

    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Invalid features list in {path}")

    return features


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing CSV, skipping: {path}")
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"nan", "none", "null", "na"}:
            return None
        return v

    try:
        if value != value:
            return None
    except Exception:
        pass

    return value


def as_float(value: Any) -> Optional[float]:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def as_int(value: Any) -> Optional[int]:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def as_text(value: Any) -> Optional[str]:
    value = clean_value(value)
    if value is None:
        return None
    return str(value)


def prop(props: Dict[str, Any], *names: str, default: Any = None) -> Any:
    if not props:
        return default

    exact = {str(k): v for k, v in props.items()}
    lowered = {str(k).lower(): v for k, v in props.items()}

    for name in names:
        if name in exact:
            return clean_value(exact[name])

        low = name.lower()
        if low in lowered:
            return clean_value(lowered[low])

    return default


def geometry_json(feature: Dict[str, Any]) -> Optional[str]:
    geom = feature.get("geometry")
    if not geom:
        return None
    return json.dumps(geom)


def point_lon_lat(feature: Dict[str, Any], props: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lon = as_float(prop(props, "longitude", "lon", "center_lon", "center_longitude", "x_lon"))
    lat = as_float(prop(props, "latitude", "lat", "center_lat", "center_latitude", "y_lat"))

    if lon is not None and lat is not None:
        return lon, lat

    geom = feature.get("geometry") or {}
    if geom.get("type") == "Point":
        coords = geom.get("coordinates") or []
        if len(coords) >= 2:
            return as_float(coords[0]), as_float(coords[1])

    return lon, lat


def infer_event_id_from_track_id(track_id: Optional[str]) -> Optional[str]:
    if not track_id:
        return None
    if "__" in track_id:
        return track_id.split("__", 1)[0]
    return None


def normalize_track_id(raw_track_id: Any, event_id: str, prefix_track_ids: bool = True) -> Optional[str]:
    track_id = as_text(raw_track_id)
    if not track_id:
        return None

    if "__" in track_id:
        return track_id

    if prefix_track_ids:
        return f"{event_id}__{track_id}"

    return track_id


def short_hash(*parts: Any) -> str:
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def feature_props(feature: Dict[str, Any]) -> Dict[str, Any]:
    props = feature.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    return props


def ensure_schema(conn) -> None:
    """
    Create/extend the schema needed for the temporal API.

    This is intentionally additive. It creates missing tables and adds missing
    columns without dropping existing schema.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_name TEXT,
                radar_site TEXT,
                date_folder TEXT,
                description TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_times (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                scan_time TIMESTAMPTZ NOT NULL,
                radar_site TEXT,
                radar_file TEXT,
                sweep INTEGER,
                sort_order INTEGER
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS radar_scans (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                radar_file TEXT,
                radar_site TEXT,
                scan_time TIMESTAMPTZ,
                scan_time_utc TIMESTAMPTZ,
                sweep INTEGER,
                storm_motion_u_kt DOUBLE PRECISION,
                storm_motion_v_kt DOUBLE PRECISION,
                storm_motion_speed_kt DOUBLE PRECISION
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS circulation_detections (
                detection_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                track_id TEXT,
                radar_file TEXT,
                radar_site TEXT,
                scan_time TIMESTAMPTZ,
                scan_time_utc TIMESTAMPTZ,
                sweep INTEGER,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                center_x_km DOUBLE PRECISION,
                center_y_km DOUBLE PRECISION,
                confidence_score DOUBLE PRECISION,
                delta_v_kt DOUBLE PRECISION,
                required_delta_v_kt DOUBLE PRECISION,
                radar_range_miles DOUBLE PRECISION,
                geom geometry(Geometry, 4326)
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signature_tracks (
                track_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                ranked_track_name TEXT,
                track_quality_rank INTEGER,
                track_quality_score DOUBLE PRECISION,
                track_quality_label TEXT,
                detection_count INTEGER,
                start_time TIMESTAMPTZ,
                end_time TIMESTAMPTZ,
                latest_scan_time TIMESTAMPTZ,
                start_time_utc TIMESTAMPTZ,
                end_time_utc TIMESTAMPTZ,
                duration_min DOUBLE PRECISION,
                speed_kt DOUBLE PRECISION,
                bearing_deg DOUBLE PRECISION,
                mean_confidence_score DOUBLE PRECISION,
                max_confidence_score DOUBLE PRECISION,
                mean_delta_v_kt DOUBLE PRECISION,
                max_delta_v_kt DOUBLE PRECISION,
                geom geometry(Geometry, 4326)
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nowcast_paths (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                track_id TEXT,
                track_quality_rank INTEGER,
                ranked_track_name TEXT,
                track_quality_score DOUBLE PRECISION,
                track_quality_label TEXT,
                source_scan_time TIMESTAMPTZ,
                source_scan_time_utc TIMESTAMPTZ,
                issued_time_utc TIMESTAMPTZ,
                projection_min INTEGER,
                valid_time_utc TIMESTAMPTZ,
                speed_kt DOUBLE PRECISION,
                bearing_deg DOUBLE PRECISION,
                geom geometry(Geometry, 4326)
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS radar_frames (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                scan_time TIMESTAMPTZ NOT NULL,
                radar_site TEXT,
                date_folder TEXT,
                radar_file TEXT,
                sweep INTEGER,
                product TEXT,
                field_name TEXT,
                image_path TEXT,
                image_url_path TEXT,
                south DOUBLE PRECISION,
                west DOUBLE PRECISION,
                north DOUBLE PRECISION,
                east DOUBLE PRECISION,
                vmin_kt DOUBLE PRECISION,
                vmax_kt DOUBLE PRECISION,
                display_range_mi DOUBLE PRECISION,
                status TEXT,
                message TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_track_predictions (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                ranked_track_name TEXT,
                ml_rank INTEGER,
                ml_score DOUBLE PRECISION,
                ml_label TEXT,
                track_quality_rank INTEGER,
                track_quality_score DOUBLE PRECISION,
                track_quality_label TEXT,
                tornado_associated INTEGER,
                model_name TEXT
            );
            """
        )

        # Add columns for users who already have older versions of these tables.
        alters = [
            "ALTER TABLE event_times ADD COLUMN IF NOT EXISTS radar_site TEXT;",
            "ALTER TABLE event_times ADD COLUMN IF NOT EXISTS radar_file TEXT;",
            "ALTER TABLE event_times ADD COLUMN IF NOT EXISTS sweep INTEGER;",
            "ALTER TABLE event_times ADD COLUMN IF NOT EXISTS sort_order INTEGER;",

            "ALTER TABLE radar_scans ADD COLUMN IF NOT EXISTS radar_site TEXT;",
            "ALTER TABLE radar_scans ADD COLUMN IF NOT EXISTS scan_time TIMESTAMPTZ;",
            "ALTER TABLE radar_scans ADD COLUMN IF NOT EXISTS scan_time_utc TIMESTAMPTZ;",

            "ALTER TABLE circulation_detections ADD COLUMN IF NOT EXISTS radar_site TEXT;",
            "ALTER TABLE circulation_detections ADD COLUMN IF NOT EXISTS scan_time TIMESTAMPTZ;",
            "ALTER TABLE circulation_detections ADD COLUMN IF NOT EXISTS scan_time_utc TIMESTAMPTZ;",
            "ALTER TABLE circulation_detections ADD COLUMN IF NOT EXISTS sweep INTEGER;",

            "ALTER TABLE signature_tracks ADD COLUMN IF NOT EXISTS start_time TIMESTAMPTZ;",
            "ALTER TABLE signature_tracks ADD COLUMN IF NOT EXISTS end_time TIMESTAMPTZ;",
            "ALTER TABLE signature_tracks ADD COLUMN IF NOT EXISTS latest_scan_time TIMESTAMPTZ;",
            "ALTER TABLE signature_tracks ADD COLUMN IF NOT EXISTS start_time_utc TIMESTAMPTZ;",
            "ALTER TABLE signature_tracks ADD COLUMN IF NOT EXISTS end_time_utc TIMESTAMPTZ;",

            "ALTER TABLE nowcast_paths ADD COLUMN IF NOT EXISTS source_scan_time TIMESTAMPTZ;",
            "ALTER TABLE nowcast_paths ADD COLUMN IF NOT EXISTS source_scan_time_utc TIMESTAMPTZ;",

            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS scan_time TIMESTAMPTZ;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS radar_site TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS date_folder TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS radar_file TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS sweep INTEGER;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS product TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS field_name TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS image_path TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS image_url_path TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS south DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS west DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS north DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS east DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS vmin_kt DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS vmax_kt DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS display_range_mi DOUBLE PRECISION;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS status TEXT;",
            "ALTER TABLE radar_frames ADD COLUMN IF NOT EXISTS message TEXT;",
        ]

        for sql in alters:
            cur.execute(sql)

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_event_times_event_time ON event_times (event_id, scan_time);",
            "CREATE INDEX IF NOT EXISTS idx_radar_scans_event_time ON radar_scans (event_id, scan_time);",
            "CREATE INDEX IF NOT EXISTS idx_detections_event_scan_time ON circulation_detections (event_id, scan_time);",
            "CREATE INDEX IF NOT EXISTS idx_nowcasts_event_source_scan_time ON nowcast_paths (event_id, source_scan_time);",
            "CREATE INDEX IF NOT EXISTS idx_tracks_event_start_end ON signature_tracks (event_id, start_time, end_time);",
            "CREATE INDEX IF NOT EXISTS idx_radar_frames_event_time ON radar_frames (event_id, scan_time);",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_event_track ON ml_track_predictions (event_id, track_id);",
        ]

        for sql in indexes:
            cur.execute(sql)


def upsert_event(conn, event_id: str, event_name: str, radar_site: str, date_folder: str, description: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events (event_id, event_name, radar_site, date_folder, description)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (event_id)
            DO UPDATE SET
                event_name = EXCLUDED.event_name,
                radar_site = EXCLUDED.radar_site,
                date_folder = EXCLUDED.date_folder,
                description = EXCLUDED.description;
            """,
            (event_id, event_name, radar_site, date_folder, description),
        )


def clear_event(conn, event_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ml_track_predictions WHERE event_id = %s;", (event_id,))
        cur.execute("DELETE FROM nowcast_paths WHERE event_id = %s;", (event_id,))
        cur.execute("DELETE FROM signature_tracks WHERE event_id = %s;", (event_id,))
        cur.execute("DELETE FROM circulation_detections WHERE event_id = %s;", (event_id,))
        cur.execute("DELETE FROM radar_scans WHERE event_id = %s;", (event_id,))
        cur.execute("DELETE FROM radar_frames WHERE event_id = %s;", (event_id,))
        cur.execute("DELETE FROM event_times WHERE event_id = %s;", (event_id,))


def event_time_rows_from_features(features: List[Dict[str, Any]], event_id: str, radar_site: str) -> List[Dict[str, Any]]:
    unique: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}

    for feature in features:
        props = feature_props(feature)
        row_event_id = as_text(prop(props, "event_id")) or event_id
        if row_event_id != event_id:
            continue

        scan_time = as_text(prop(props, "scan_time", "scan_time_utc", "time_utc", "valid_time_utc"))
        radar_file = as_text(prop(props, "radar_file", "file", "filename", "source_file"))
        sweep = as_int(prop(props, "sweep", "sweep_index", "sweep_number"))
        row_radar_site = as_text(prop(props, "radar_site")) or radar_site

        if not scan_time:
            continue

        key = (row_event_id, scan_time, sweep)
        unique[key] = {
            "event_id": row_event_id,
            "scan_time": scan_time,
            "radar_site": row_radar_site,
            "radar_file": radar_file,
            "sweep": sweep,
            "sort_order": None,
        }

    rows = list(unique.values())
    rows.sort(key=lambda row: (row.get("event_id", ""), row.get("scan_time", ""), row.get("sweep") or 0))
    for i, row in enumerate(rows, start=1):
        if row.get("sort_order") is None:
            row["sort_order"] = i
    return rows


def insert_event_times(conn, rows: List[Dict[str, Any]], event_id: str, radar_site: str) -> int:
    inserted = 0

    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for idx, row in enumerate(rows, start=1):
        row_event_id = as_text(prop(row, "event_id")) or event_id
        if row_event_id != event_id:
            continue

        scan_time = as_text(prop(row, "scan_time", "scan_time_utc"))
        if not scan_time:
            continue

        sweep = as_int(prop(row, "sweep", "sweep_index", "sweep_number"))
        key = (row_event_id, scan_time, sweep)

        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            {
                "event_id": row_event_id,
                "scan_time": scan_time,
                "radar_site": as_text(prop(row, "radar_site")) or radar_site,
                "radar_file": as_text(prop(row, "radar_file", "file", "filename", "source_file")),
                "sweep": sweep,
                "sort_order": as_int(prop(row, "sort_order")) or idx,
            }
        )

    with conn.cursor() as cur:
        for row in cleaned:
            cur.execute(
                """
                INSERT INTO event_times (
                    event_id,
                    scan_time,
                    radar_site,
                    radar_file,
                    sweep,
                    sort_order
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (
                    row["event_id"],
                    row["scan_time"],
                    row["radar_site"],
                    row["radar_file"],
                    row["sweep"],
                    row["sort_order"],
                ),
            )
            inserted += 1

    return inserted


def insert_radar_scans(
    conn,
    event_time_rows: List[Dict[str, Any]],
    detection_features: List[Dict[str, Any]],
    event_id: str,
    radar_site: str,
) -> int:
    unique: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}

    for row in event_time_rows:
        row_event_id = as_text(prop(row, "event_id")) or event_id
        if row_event_id != event_id:
            continue

        radar_file = as_text(prop(row, "radar_file", "file", "filename", "source_file"))
        scan_time = as_text(prop(row, "scan_time", "scan_time_utc"))
        sweep = as_int(prop(row, "sweep", "sweep_index", "sweep_number"))
        row_radar_site = as_text(prop(row, "radar_site")) or radar_site

        if not scan_time:
            continue

        unique[(radar_file, scan_time, sweep)] = {
            "event_id": event_id,
            "radar_file": radar_file,
            "radar_site": row_radar_site,
            "scan_time": scan_time,
            "scan_time_utc": scan_time,
            "sweep": sweep,
            "storm_motion_u_kt": None,
            "storm_motion_v_kt": None,
            "storm_motion_speed_kt": None,
        }

    for feature in detection_features:
        props = feature_props(feature)

        row_event_id = as_text(prop(props, "event_id")) or event_id
        if row_event_id != event_id:
            continue

        radar_file = as_text(prop(props, "radar_file", "file", "filename", "source_file"))
        scan_time = as_text(prop(props, "scan_time", "scan_time_utc", "time_utc", "valid_time_utc"))
        sweep = as_int(prop(props, "sweep", "sweep_index", "sweep_number"))
        row_radar_site = as_text(prop(props, "radar_site")) or radar_site

        if not radar_file and not scan_time:
            continue

        key = (radar_file, scan_time, sweep)
        unique[key] = {
            "event_id": event_id,
            "radar_file": radar_file,
            "radar_site": row_radar_site,
            "scan_time": scan_time,
            "scan_time_utc": scan_time,
            "sweep": sweep,
            "storm_motion_u_kt": as_float(prop(props, "storm_motion_u_kt", "mean_storm_motion_u_kt")),
            "storm_motion_v_kt": as_float(prop(props, "storm_motion_v_kt", "mean_storm_motion_v_kt")),
            "storm_motion_speed_kt": as_float(prop(props, "storm_motion_speed_kt", "mean_storm_speed_kt")),
        }

    if not unique:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for row in unique.values():
            cur.execute(
                """
                INSERT INTO radar_scans (
                    event_id,
                    radar_file,
                    radar_site,
                    scan_time,
                    scan_time_utc,
                    sweep,
                    storm_motion_u_kt,
                    storm_motion_v_kt,
                    storm_motion_speed_kt
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    row["event_id"],
                    row["radar_file"],
                    row["radar_site"],
                    row["scan_time"],
                    row["scan_time_utc"],
                    row["sweep"],
                    row["storm_motion_u_kt"],
                    row["storm_motion_v_kt"],
                    row["storm_motion_speed_kt"],
                ),
            )
            inserted += 1

    return inserted


def insert_detections(conn, features: List[Dict[str, Any]], event_id: str, radar_site: str, prefix_track_ids: bool = True) -> int:
    inserted = 0

    with conn.cursor() as cur:
        for i, feature in enumerate(features):
            props = feature_props(feature)
            geom_json = geometry_json(feature)

            row_event_id = as_text(prop(props, "event_id")) or event_id
            if row_event_id != event_id:
                continue

            raw_track_id = prop(props, "track_id", "id")
            track_id = normalize_track_id(raw_track_id, row_event_id, prefix_track_ids=prefix_track_ids)

            radar_file = as_text(prop(props, "radar_file", "file", "filename", "source_file"))
            row_radar_site = as_text(prop(props, "radar_site")) or radar_site
            scan_time = as_text(prop(props, "scan_time", "scan_time_utc", "time_utc", "valid_time_utc"))
            sweep = as_int(prop(props, "sweep", "sweep_index", "sweep_number"))

            lon, lat = point_lon_lat(feature, props)

            detection_id = as_text(prop(props, "detection_id", "candidate_id"))
            if not detection_id:
                detection_id = f"{row_event_id}__DET_{short_hash(track_id, radar_file, scan_time, lon, lat, i)}"

            cur.execute(
                """
                INSERT INTO circulation_detections (
                    detection_id,
                    event_id,
                    track_id,
                    radar_file,
                    radar_site,
                    scan_time,
                    scan_time_utc,
                    sweep,
                    latitude,
                    longitude,
                    center_x_km,
                    center_y_km,
                    confidence_score,
                    delta_v_kt,
                    required_delta_v_kt,
                    radar_range_miles,
                    geom
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    CASE
                        WHEN %s IS NULL THEN NULL
                        ELSE ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                    END
                )
                ON CONFLICT (detection_id)
                DO UPDATE SET
                    event_id = EXCLUDED.event_id,
                    track_id = EXCLUDED.track_id,
                    radar_file = EXCLUDED.radar_file,
                    radar_site = EXCLUDED.radar_site,
                    scan_time = EXCLUDED.scan_time,
                    scan_time_utc = EXCLUDED.scan_time_utc,
                    sweep = EXCLUDED.sweep,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    center_x_km = EXCLUDED.center_x_km,
                    center_y_km = EXCLUDED.center_y_km,
                    confidence_score = EXCLUDED.confidence_score,
                    delta_v_kt = EXCLUDED.delta_v_kt,
                    required_delta_v_kt = EXCLUDED.required_delta_v_kt,
                    radar_range_miles = EXCLUDED.radar_range_miles,
                    geom = EXCLUDED.geom;
                """,
                (
                    detection_id,
                    row_event_id,
                    track_id,
                    radar_file,
                    row_radar_site,
                    scan_time,
                    scan_time,
                    sweep,
                    lat,
                    lon,
                    as_float(prop(props, "center_x_km", "x_km", "center_x")),
                    as_float(prop(props, "center_y_km", "y_km", "center_y")),
                    as_float(prop(props, "confidence_score", "confidence", "score")),
                    as_float(prop(props, "delta_v_kt", "max_delta_v_kt", "dv_kt")),
                    as_float(prop(props, "required_delta_v_kt", "required_dv_kt")),
                    as_float(prop(props, "radar_range_miles", "range_miles", "range_mi")),
                    geom_json,
                    geom_json,
                ),
            )

            inserted += 1

    return inserted


def insert_tracks(conn, features: List[Dict[str, Any]], event_id: str, prefix_track_ids: bool = True) -> int:
    inserted = 0

    with conn.cursor() as cur:
        for i, feature in enumerate(features):
            props = feature_props(feature)
            geom_json = geometry_json(feature)

            row_event_id = as_text(prop(props, "event_id")) or event_id
            if row_event_id != event_id:
                continue

            raw_track_id = prop(props, "track_id", "id")
            track_id = normalize_track_id(raw_track_id, row_event_id, prefix_track_ids=prefix_track_ids)
            if not track_id:
                track_id = f"{row_event_id}__TRK_{short_hash(i, geom_json)}"

            start_time = as_text(prop(props, "start_time", "start_time_utc"))
            end_time = as_text(prop(props, "end_time", "end_time_utc"))
            latest_scan_time = as_text(prop(props, "latest_scan_time", "end_time", "end_time_utc")) or end_time

            cur.execute(
                """
                INSERT INTO signature_tracks (
                    track_id,
                    event_id,
                    ranked_track_name,
                    track_quality_rank,
                    track_quality_score,
                    track_quality_label,
                    detection_count,
                    start_time,
                    end_time,
                    latest_scan_time,
                    start_time_utc,
                    end_time_utc,
                    duration_min,
                    speed_kt,
                    bearing_deg,
                    mean_confidence_score,
                    max_confidence_score,
                    mean_delta_v_kt,
                    max_delta_v_kt,
                    geom
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    CASE
                        WHEN %s IS NULL THEN NULL
                        ELSE ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                    END
                )
                ON CONFLICT (track_id)
                DO UPDATE SET
                    event_id = EXCLUDED.event_id,
                    ranked_track_name = EXCLUDED.ranked_track_name,
                    track_quality_rank = EXCLUDED.track_quality_rank,
                    track_quality_score = EXCLUDED.track_quality_score,
                    track_quality_label = EXCLUDED.track_quality_label,
                    detection_count = EXCLUDED.detection_count,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    latest_scan_time = EXCLUDED.latest_scan_time,
                    start_time_utc = EXCLUDED.start_time_utc,
                    end_time_utc = EXCLUDED.end_time_utc,
                    duration_min = EXCLUDED.duration_min,
                    speed_kt = EXCLUDED.speed_kt,
                    bearing_deg = EXCLUDED.bearing_deg,
                    mean_confidence_score = EXCLUDED.mean_confidence_score,
                    max_confidence_score = EXCLUDED.max_confidence_score,
                    mean_delta_v_kt = EXCLUDED.mean_delta_v_kt,
                    max_delta_v_kt = EXCLUDED.max_delta_v_kt,
                    geom = EXCLUDED.geom;
                """,
                (
                    track_id,
                    row_event_id,
                    as_text(prop(props, "ranked_track_name", "track_name", "name")),
                    as_int(prop(props, "track_quality_rank", "quality_rank")),
                    as_float(prop(props, "track_quality_score", "quality_score")),
                    as_text(prop(props, "track_quality_label", "quality_label")),
                    as_int(prop(props, "detection_count", "n_detections", "num_detections")),
                    start_time,
                    end_time,
                    latest_scan_time,
                    start_time,
                    end_time,
                    as_float(prop(props, "duration_min", "persistence_min")),
                    as_float(prop(props, "speed_kt", "mean_speed_kt")),
                    as_float(prop(props, "bearing_deg", "motion_bearing_deg")),
                    as_float(prop(props, "mean_confidence_score", "avg_confidence_score")),
                    as_float(prop(props, "max_confidence_score")),
                    as_float(prop(props, "mean_delta_v_kt", "avg_delta_v_kt")),
                    as_float(prop(props, "max_delta_v_kt")),
                    geom_json,
                    geom_json,
                ),
            )

            inserted += 1

    return inserted


def insert_nowcasts(conn, features: List[Dict[str, Any]], event_id: str, prefix_track_ids: bool = True) -> int:
    inserted = 0

    with conn.cursor() as cur:
        for feature in features:
            props = feature_props(feature)
            geom_json = geometry_json(feature)

            row_event_id = as_text(prop(props, "event_id")) or event_id
            if row_event_id != event_id:
                continue

            raw_track_id = prop(props, "track_id", "id")
            track_id = normalize_track_id(raw_track_id, row_event_id, prefix_track_ids=prefix_track_ids)

            source_scan_time = as_text(
                prop(
                    props,
                    "source_scan_time",
                    "source_scan_time_utc",
                    "issued_time_utc",
                    "issue_time_utc",
                    "end_time_utc",
                    "track_end_time_utc",
                )
            )

            issued_time = as_text(prop(props, "issued_time_utc", "issue_time_utc")) or source_scan_time

            cur.execute(
                """
                INSERT INTO nowcast_paths (
                    event_id,
                    track_id,
                    track_quality_rank,
                    ranked_track_name,
                    track_quality_score,
                    track_quality_label,
                    source_scan_time,
                    source_scan_time_utc,
                    issued_time_utc,
                    projection_min,
                    valid_time_utc,
                    speed_kt,
                    bearing_deg,
                    geom
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    CASE
                        WHEN %s IS NULL THEN NULL
                        ELSE ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                    END
                );
                """,
                (
                    row_event_id,
                    track_id,
                    as_int(prop(props, "track_quality_rank", "quality_rank")),
                    as_text(prop(props, "ranked_track_name", "track_name", "name")),
                    as_float(prop(props, "track_quality_score", "quality_score")),
                    as_text(prop(props, "track_quality_label", "quality_label")),
                    source_scan_time,
                    source_scan_time,
                    issued_time,
                    as_int(prop(props, "projection_min", "projection_minutes", "nowcast_min")),
                    as_text(prop(props, "valid_time_utc", "projected_time_utc", "projection_time_utc")),
                    as_float(prop(props, "speed_kt", "nowcast_speed_kt")),
                    as_float(prop(props, "bearing_deg", "nowcast_bearing_deg")),
                    geom_json,
                    geom_json,
                ),
            )

            inserted += 1

    return inserted


def insert_radar_frames(conn, rows: List[Dict[str, Any]], event_id: str, radar_site: str, date_folder: str) -> int:
    inserted = 0

    with conn.cursor() as cur:
        for row in rows:
            row_event_id = as_text(prop(row, "event_id")) or event_id
            if row_event_id != event_id:
                continue

            status = as_text(prop(row, "status")) or "ok"
            scan_time = as_text(prop(row, "scan_time", "scan_time_utc"))
            image_url_path = as_text(prop(row, "image_url_path", "image_url"))
            image_path = as_text(prop(row, "image_path"))

            if not scan_time:
                continue

            # Keep metadata for failed frames out of the frontend lookup table.
            # The CSV still preserves those failures for QA.
            if status != "ok":
                continue

            cur.execute(
                """
                INSERT INTO radar_frames (
                    event_id,
                    scan_time,
                    radar_site,
                    date_folder,
                    radar_file,
                    sweep,
                    product,
                    field_name,
                    image_path,
                    image_url_path,
                    south,
                    west,
                    north,
                    east,
                    vmin_kt,
                    vmax_kt,
                    display_range_mi,
                    status,
                    message
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                );
                """,
                (
                    row_event_id,
                    scan_time,
                    as_text(prop(row, "radar_site")) or radar_site,
                    as_text(prop(row, "date_folder")) or date_folder,
                    as_text(prop(row, "radar_file")),
                    as_int(prop(row, "sweep")),
                    as_text(prop(row, "product")) or "velocity",
                    as_text(prop(row, "field_name")),
                    image_path,
                    image_url_path,
                    as_float(prop(row, "south")),
                    as_float(prop(row, "west")),
                    as_float(prop(row, "north")),
                    as_float(prop(row, "east")),
                    as_float(prop(row, "vmin_kt")),
                    as_float(prop(row, "vmax_kt")),
                    as_float(prop(row, "display_range_mi")),
                    status,
                    as_text(prop(row, "message")),
                ),
            )

            inserted += 1

    return inserted


def insert_ml_predictions(conn, rows: List[Dict[str, Any]], event_id: str, prefix_track_ids: bool = True) -> int:
    inserted = 0

    with conn.cursor() as cur:
        for row in rows:
            raw_track_id = prop(row, "track_id")
            inferred_event_id = as_text(prop(row, "event_id")) or infer_event_id_from_track_id(as_text(raw_track_id)) or event_id

            if inferred_event_id != event_id:
                continue

            track_id = normalize_track_id(raw_track_id, inferred_event_id, prefix_track_ids=prefix_track_ids)
            if not track_id:
                continue

            cur.execute(
                """
                INSERT INTO ml_track_predictions (
                    event_id,
                    track_id,
                    ranked_track_name,
                    ml_rank,
                    ml_score,
                    ml_label,
                    track_quality_rank,
                    track_quality_score,
                    track_quality_label,
                    tornado_associated,
                    model_name
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    inferred_event_id,
                    track_id,
                    as_text(prop(row, "ranked_track_name", "track_name", "name")),
                    as_int(prop(row, "ml_rank")),
                    as_float(prop(row, "ml_score")),
                    as_text(prop(row, "ml_label")),
                    as_int(prop(row, "track_quality_rank")),
                    as_float(prop(row, "track_quality_score")),
                    as_text(prop(row, "track_quality_label")),
                    as_int(prop(row, "tornado_associated")),
                    as_text(prop(row, "model_name")) or "eve_track_scoring_model",
                ),
            )

            inserted += 1

    return inserted


def count_table(conn, table: str, event_id: Optional[str] = None) -> int:
    with conn.cursor() as cur:
        if event_id is None:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
        else:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE event_id = %s;", (event_id,))
        return int(cur.fetchone()[0])


def run_temporal_smoke_queries(conn, event_id: str) -> None:
    with conn.cursor() as cur:
        print("\nEvent time steps:")
        cur.execute(
            """
            SELECT event_id, scan_time, radar_file, sweep
            FROM event_times
            WHERE event_id = %s
            ORDER BY scan_time
            LIMIT 20;
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  [none]")
        else:
            for row in rows:
                print(f"  {row[0]} | {row[1]} | {row[2]} | sweep={row[3]}")

        print("\nDetection counts by scan:")
        cur.execute(
            """
            SELECT event_id, scan_time, COUNT(*)
            FROM circulation_detections
            WHERE event_id = %s
            GROUP BY event_id, scan_time
            ORDER BY scan_time
            LIMIT 20;
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  [none]")
        else:
            for row in rows:
                print(f"  {row[0]} | {row[1]} | detections={row[2]}")

        print("\nRadar frames by scan:")
        cur.execute(
            """
            SELECT event_id, scan_time, radar_file, image_url_path
            FROM radar_frames
            WHERE event_id = %s
            ORDER BY scan_time
            LIMIT 20;
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  [none]")
        else:
            for row in rows:
                print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")

        print("\nNowcast counts by source scan:")
        cur.execute(
            """
            SELECT event_id, source_scan_time, projection_min, COUNT(*)
            FROM nowcast_paths
            WHERE event_id = %s
            GROUP BY event_id, source_scan_time, projection_min
            ORDER BY source_scan_time, projection_min
            LIMIT 20;
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  [none]")
        else:
            for row in rows:
                print(f"  {row[0]} | {row[1]} | {row[2]} min | nowcasts={row[3]}")


def run_smoke_queries(conn, event_id: str) -> None:
    with conn.cursor() as cur:
        print("\nTop tracks by ML score:")
        cur.execute(
            """
            SELECT
                track_id,
                ranked_track_name,
                ml_score,
                ml_rank,
                ml_label,
                track_quality_score
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
            LIMIT 10;
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  [none]")
        else:
            for r in rows:
                print(f"  {r[0]} | {r[1]} | ml_score={r[2]} | ml_rank={r[3]} | {r[4]} | quality={r[5]}")

        print("\nTop tracks by track quality:")
        cur.execute(
            """
            SELECT
                track_id,
                ranked_track_name,
                track_quality_score,
                track_quality_rank,
                detection_count,
                start_time,
                end_time
            FROM signature_tracks
            WHERE event_id = %s
            ORDER BY track_quality_score DESC NULLS LAST, track_quality_rank ASC NULLS LAST
            LIMIT 10;
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  [none]")
        else:
            for r in rows:
                print(f"  {r[0]} | {r[1]} | quality={r[2]} | rank={r[3]} | detections={r[4]} | {r[5]} → {r[6]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load time-aware E-V-E track/detection/nowcast/ML outputs into PostGIS."
    )

    parser.add_argument("--event-id", default=os.getenv("EVENT_ID", "test_case_1"))
    parser.add_argument("--event-name", default=os.getenv("EVENT_NAME"))
    parser.add_argument("--radar-site", default=os.getenv("RADAR_SITE"))
    parser.add_argument("--date-folder", default=os.getenv("DATE_FOLDER"))
    parser.add_argument(
        "--description",
        default=os.getenv("EVENT_DESCRIPTION"),
    )

    parser.add_argument("--event-times-csv", type=Path, default=None)
    parser.add_argument("--radar-frames-csv", type=Path, default=None)
    parser.add_argument("--track-points-geojson", type=Path, default=None)
    parser.add_argument("--tracks-geojson", type=Path, default=None)
    parser.add_argument("--nowcasts-geojson", type=Path, default=None)
    parser.add_argument("--ml-predictions-csv", type=Path, default=None)

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing event rows instead of clearing the selected event first.",
    )
    parser.add_argument(
        "--no-prefix-track-ids",
        action="store_true",
        help="Do not convert TRK_### to event_id__TRK_###.",
    )
    parser.add_argument(
        "--no-smoke-queries",
        action="store_true",
        help="Skip final top-track query printout.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = resolve_case(args.event_id, event_name=args.event_name, radar_site=args.radar_site, date_folder=args.date_folder, description=args.description)
    args.event_id = case.event_id
    args.event_name = case.event_name
    args.radar_site = case.radar_site
    args.date_folder = case.date_folder
    args.description = case.description
    prefix_track_ids = not args.no_prefix_track_ids
    event_times_csv = args.event_times_csv or default_event_times_csv(args.event_id)
    radar_frames_csv = args.radar_frames_csv or default_radar_frames_csv(args.event_id)
    track_points_geojson = args.track_points_geojson or default_track_points_geojson(args.event_id)
    tracks_geojson = args.tracks_geojson or default_tracks_geojson(args.event_id)
    nowcasts_geojson = args.nowcasts_geojson or default_nowcasts_geojson(args.event_id)
    ml_predictions_csv = args.ml_predictions_csv or default_ml_predictions_csv(args.event_id)

    print("E-V-E — Time-Aware PostGIS Loader")
    print("----------------------------------------------")
    print(f"Project root:     {ROOT}")
    print(f"Event ID:         {args.event_id}")
    print(f"Radar site:       {args.radar_site}")
    print(f"Date folder:      {args.date_folder}")
    print(f"Event times CSV:  {event_times_csv}")
    print(f"Radar frames CSV: {radar_frames_csv}")
    print(f"Track points:      {track_points_geojson}")
    print(f"Tracks:            {tracks_geojson}")
    print(f"Nowcasts:          {nowcasts_geojson}")
    print(f"ML predictions:    {ml_predictions_csv}")
    print()

    event_time_rows = load_csv_rows(event_times_csv)
    track_point_features = load_geojson(track_points_geojson)
    track_features = load_geojson(tracks_geojson)
    nowcast_features = load_geojson(nowcasts_geojson)
    radar_frame_rows = load_csv_rows(radar_frames_csv)
    ml_rows = load_csv_rows(ml_predictions_csv)

    if not event_time_rows:
        print("[WARN] No event_times CSV rows found. Falling back to scan times from track-point GeoJSON.")
        event_time_rows = event_time_rows_from_features(track_point_features, args.event_id, args.radar_site)

    print("Input counts:")
    print(f"  event time rows:       {len(event_time_rows)}")
    print(f"  track point features:  {len(track_point_features)}")
    print(f"  track features:        {len(track_features)}")
    print(f"  nowcast features:      {len(nowcast_features)}")
    print(f"  radar frame rows:      {len(radar_frame_rows)}")
    print(f"  ML prediction rows:    {len(ml_rows)}")
    print()

    conn = db_connect()

    try:
        with conn:
            ensure_schema(conn)

            upsert_event(
                conn=conn,
                event_id=args.event_id,
                event_name=args.event_name,
                radar_site=args.radar_site,
                date_folder=args.date_folder,
                description=args.description,
            )

            if not args.append:
                clear_event(conn, args.event_id)

            event_time_count = insert_event_times(conn, event_time_rows, args.event_id, args.radar_site)
            scan_count = insert_radar_scans(conn, event_time_rows, track_point_features, args.event_id, args.radar_site)
            detection_count = insert_detections(conn, track_point_features, args.event_id, args.radar_site, prefix_track_ids)
            track_count = insert_tracks(conn, track_features, args.event_id, prefix_track_ids)
            nowcast_count = insert_nowcasts(conn, nowcast_features, args.event_id, prefix_track_ids)
            radar_frame_count = insert_radar_frames(
                conn,
                radar_frame_rows,
                args.event_id,
                args.radar_site,
                args.date_folder,
            )
            ml_count = insert_ml_predictions(conn, ml_rows, args.event_id, prefix_track_ids)

        print("Load complete.")
        print(f"  event_times inserted:           {event_time_count}")
        print(f"  radar_scans inserted:           {scan_count}")
        print(f"  circulation_detections inserted:{detection_count}")
        print(f"  signature_tracks inserted:      {track_count}")
        print(f"  nowcast_paths inserted:         {nowcast_count}")
        print(f"  radar_frames inserted:          {radar_frame_count}")
        print(f"  ml_track_predictions inserted:  {ml_count}")

        print("\nDatabase counts for event:")
        for table in [
            "event_times",
            "radar_scans",
            "circulation_detections",
            "signature_tracks",
            "nowcast_paths",
            "radar_frames",
            "ml_track_predictions",
        ]:
            print(f"  {table}: {count_table(conn, table, args.event_id)}")

        if not args.no_smoke_queries:
            run_temporal_smoke_queries(conn, args.event_id)
            run_smoke_queries(conn, args.event_id)

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
