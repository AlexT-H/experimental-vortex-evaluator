from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


app = FastAPI(
    title="E-V-E API",
    description="PostGIS-backed API for E-V-E radar-derived circulation tracks, nowcasts, and ML scores.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def project_root_candidates() -> List[Path]:
    """
    Return likely project roots for Docker and local dev.

    This avoids the common issue where the API process runs from /app/backend,
    /app, or another working directory while the radar PNGs are mounted under a
    different project root.
    """
    candidates: List[Path] = []

    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()

    candidates.extend(
        [
            cwd,
            cwd.parent,
            here.parent,
            here.parent.parent,
            Path("/app"),
            Path("/workspace"),
        ]
    )

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)

    return unique


def radar_frames_dir_candidates() -> List[Path]:
    candidates: List[Path] = []

    for root in project_root_candidates():
        # Local/Docker pipeline output path.
        candidates.append(root / "outputs" / "web" / "radar_frames")

        # Future renamed output folder support.
        candidates.append(root / "outputs" / "frontend" / "radar_frames")

        # Hosted/Vercel-friendly committed static frame paths.
        candidates.append(root / "backend" / "static" / "radar_frames")
        candidates.append(root / "static" / "radar_frames")

    return candidates

def resolve_radar_frame_file(event_id: str, filename: str) -> Optional[Path]:
    """
    Resolve a radar PNG by checking multiple possible project roots.

    Only returns files inside outputs/web/radar_frames/{event_id}; this prevents
    path traversal while still being robust across Docker/local path layouts.
    """
    safe_event_id = Path(event_id).name
    safe_filename = Path(filename).name

    for base_dir in radar_frames_dir_candidates():
        candidate = (base_dir / safe_event_id / safe_filename).resolve()

        try:
            allowed_root = (base_dir / safe_event_id).resolve()
            candidate.relative_to(allowed_root)
        except Exception:
            continue

        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def db_connect():
    """
    Connect to PostGIS.

    Local/Docker usage can use individual POSTGRES_* variables.
    Hosted deployments such as Vercel + Neon/Supabase can use DATABASE_URL
    or POSTGRES_URL without breaking local behavior.
    """
    database_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    if database_url:
        return psycopg2.connect(database_url)

    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "eve"),
        user=os.getenv("POSTGRES_USER", "eve_user"),
        password=os.getenv("POSTGRES_PASSWORD", "eve_password"),
        host=os.getenv("POSTGRES_HOST", "postgis"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )

def json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def fetch_all(sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    conn = db_connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [{k: json_safe(v) for k, v in dict(row).items()} for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_one(sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def make_feature_collection(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    features = []

    for row in rows:
        geom = row.pop("geometry", None)

        if isinstance(geom, str):
            geom = json.loads(geom)

        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": row,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def time_filter_sql(column_expr: str, param_position: int = 1) -> str:
    """
    Compare timestamp-ish columns safely even if an older DB column was created
    as text. The loader now creates TIMESTAMPTZ columns, but this avoids fragile
    type mismatches during transition.
    """
    return f"AND ({column_expr})::timestamptz = %s::timestamptz"


@app.get("/")
def root():
    return {
        "name": "E-V-E API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    try:
        result = fetch_one("SELECT PostGIS_Version() AS postgis_version;")
        return {
            "status": "ok",
            "postgis_version": result["postgis_version"] if result else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))




@app.get("/debug/radar-frame-paths")
def debug_radar_frame_paths():
    """
    Debug helper: shows where the API is looking for radar frame PNGs.
    """
    dirs = []
    for directory in radar_frames_dir_candidates():
        exists = directory.exists()
        sample_files: List[str] = []

        if exists:
            try:
                sample_files = [
                    str(path.relative_to(directory))
                    for path in sorted(directory.rglob("*.png"))[:10]
                ]
            except Exception:
                sample_files = []

        dirs.append(
            {
                "directory": str(directory),
                "exists": exists,
                "sample_pngs": sample_files,
            }
        )

    return {
        "cwd": str(Path.cwd().resolve()),
        "file": str(Path(__file__).resolve()),
        "project_root_env": os.getenv("PROJECT_ROOT"),
        "checked_directories": dirs,
    }


@app.get("/radar_frames/{event_id}/{filename:path}")
def serve_radar_frame(event_id: str, filename: str):
    """
    Serve rendered radar PNGs.

    This replaces StaticFiles because Docker path layouts often make the static
    mount point resolve to the wrong folder.
    """
    path = resolve_radar_frame_file(event_id, filename)

    if not path:
        checked = [str(directory / Path(event_id).name / Path(filename).name) for directory in radar_frames_dir_candidates()]
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Radar frame PNG not found by API process.",
                "event_id": event_id,
                "filename": filename,
                "checked_paths": checked,
                "fix": "Make sure ./outputs is mounted into the API container, set PROJECT_ROOT, or place hosted frames under backend/static/radar_frames.",
            },
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=path.name,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/events")
def get_events():
    return fetch_all(
        """
        SELECT
            event_id,
            event_name,
            radar_site,
            date_folder,
            description
        FROM events
        ORDER BY event_id;
        """
    )


@app.get("/events/{event_id}/times")
def get_event_times(event_id: str):
    """
    Return one row per radar scan/volume time for the selected event.

    This endpoint powers the frontend time slider.
    """
    rows = fetch_all(
        """
        WITH raw_times AS (
            SELECT
                event_id,
                scan_time,
                radar_site,
                radar_file,
                sweep,
                sort_order
            FROM event_times
            WHERE event_id = %s

            UNION ALL

            SELECT
                event_id,
                COALESCE(scan_time::text, scan_time_utc::text)::timestamptz AS scan_time,
                radar_site,
                radar_file,
                sweep,
                NULL::integer AS sort_order
            FROM radar_scans
            WHERE event_id = %s
              AND COALESCE(scan_time::text, scan_time_utc::text) IS NOT NULL
        ),
        deduped AS (
            SELECT DISTINCT ON (scan_time, sweep)
                event_id,
                scan_time,
                to_char(scan_time AT TIME ZONE 'UTC', 'HH24:MI UTC') AS label,
                radar_site,
                radar_file,
                sweep,
                sort_order
            FROM raw_times
            WHERE scan_time IS NOT NULL
            ORDER BY scan_time, sweep, sort_order NULLS LAST
        )
        SELECT *
        FROM deduped
        ORDER BY scan_time;
        """,
        (event_id, event_id),
    )

    return rows


@app.get("/events/{event_id}/summary")
def get_event_summary(event_id: str):
    event = fetch_one(
        """
        SELECT
            event_id,
            event_name,
            radar_site,
            date_folder,
            description
        FROM events
        WHERE event_id = %s;
        """,
        (event_id,),
    )

    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    counts = {}
    for table in [
        "event_times",
        "radar_scans",
        "circulation_detections",
        "signature_tracks",
        "nowcast_paths",
        "radar_frames",
        "ml_track_predictions",
    ]:
        row = fetch_one(
            f"SELECT COUNT(*) AS count FROM {table} WHERE event_id = %s;",
            (event_id,),
        )
        counts[table] = row["count"] if row else 0

    best_track = fetch_one(
        """
        SELECT
            p.track_id,
            p.ranked_track_name,
            p.ml_rank,
            p.ml_score,
            p.ml_label,
            p.track_quality_rank,
            p.track_quality_score,
            p.track_quality_label
        FROM ml_track_predictions p
        WHERE p.event_id = %s
        ORDER BY p.ml_score DESC NULLS LAST, p.ml_rank ASC NULLS LAST
        LIMIT 1;
        """,
        (event_id,),
    )

    event["counts"] = counts
    event["best_track"] = best_track

    return event


@app.get("/events/{event_id}/tracks")
def get_tracks(event_id: str, time: Optional[str] = None):
    time_clause = ""
    params: List[Any] = [event_id, event_id]

    if time:
        time_clause = """
          AND COALESCE(t.start_time::text, t.start_time_utc::text)::timestamptz <= %s::timestamptz
          AND COALESCE(t.end_time::text, t.end_time_utc::text)::timestamptz >= %s::timestamptz
        """
        params.extend([time, time])

    rows = fetch_all(
        f"""
        WITH ml_best AS (
            SELECT DISTINCT ON (track_id)
                track_id,
                ranked_track_name AS ml_ranked_track_name,
                ml_rank,
                ml_score,
                ml_label,
                tornado_associated,
                model_name
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY track_id, ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
        )
        SELECT
            t.event_id,
            t.track_id,
            COALESCE(t.ranked_track_name, ml.ml_ranked_track_name) AS ranked_track_name,
            t.track_quality_rank,
            t.track_quality_score,
            t.track_quality_label,
            t.detection_count,
            COALESCE(t.start_time::text, t.start_time_utc::text) AS start_time,
            COALESCE(t.end_time::text, t.end_time_utc::text) AS end_time,
            COALESCE(t.latest_scan_time::text, t.end_time::text, t.end_time_utc::text) AS latest_scan_time,
            t.start_time_utc,
            t.end_time_utc,
            t.duration_min,
            t.speed_kt,
            t.bearing_deg,
            t.mean_confidence_score,
            t.max_confidence_score,
            t.mean_delta_v_kt,
            t.max_delta_v_kt,
            ml.ml_rank,
            ml.ml_score,
            ml.ml_label,
            ml.tornado_associated,
            ml.model_name,
            ST_AsGeoJSON(t.geom)::json AS geometry
        FROM signature_tracks t
        LEFT JOIN ml_best ml
          ON t.track_id = ml.track_id
        WHERE t.event_id = %s
        {time_clause}
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            t.track_quality_score DESC NULLS LAST,
            t.track_quality_rank ASC NULLS LAST;
        """,
        tuple(params),
    )

    return make_feature_collection(rows)


@app.get("/events/{event_id}/nowcasts")
def get_nowcasts(event_id: str, time: Optional[str] = None):
    time_clause = ""
    params: List[Any] = [event_id, event_id]

    if time:
        time_clause = """
          AND COALESCE(n.source_scan_time::text, n.source_scan_time_utc::text, n.issued_time_utc::text)::timestamptz = %s::timestamptz
        """
        params.append(time)

    rows = fetch_all(
        f"""
        WITH ml_best AS (
            SELECT DISTINCT ON (track_id)
                track_id,
                ml_rank,
                ml_score,
                ml_label,
                tornado_associated,
                model_name
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY track_id, ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
        )
        SELECT
            n.event_id,
            n.track_id,
            n.ranked_track_name,
            n.track_quality_rank,
            n.track_quality_score,
            n.track_quality_label,
            COALESCE(n.source_scan_time::text, n.source_scan_time_utc::text, n.issued_time_utc::text) AS source_scan_time,
            n.source_scan_time_utc,
            n.issued_time_utc,
            n.projection_min,
            n.valid_time_utc,
            n.speed_kt,
            n.bearing_deg,
            ml.ml_rank,
            ml.ml_score,
            ml.ml_label,
            ml.tornado_associated,
            ml.model_name,
            ST_AsGeoJSON(n.geom)::json AS geometry
        FROM nowcast_paths n
        LEFT JOIN ml_best ml
          ON n.track_id = ml.track_id
        WHERE n.event_id = %s
        {time_clause}
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            n.track_quality_score DESC NULLS LAST,
            n.track_id,
            n.projection_min;
        """,
        tuple(params),
    )

    return make_feature_collection(rows)


@app.get("/events/{event_id}/detections")
def get_detections(event_id: str, time: Optional[str] = None, limit: int = 1000):
    time_clause = ""
    params: List[Any] = [event_id, event_id]

    if time:
        time_clause = """
          AND COALESCE(d.scan_time::text, d.scan_time_utc::text)::timestamptz = %s::timestamptz
        """
        params.append(time)

    params.append(limit)

    rows = fetch_all(
        f"""
        WITH ml_best AS (
            SELECT DISTINCT ON (track_id)
                track_id,
                ml_rank,
                ml_score,
                ml_label
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY track_id, ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
        )
        SELECT
            d.event_id,
            d.detection_id,
            d.track_id,
            d.radar_file,
            d.radar_site,
            COALESCE(d.scan_time::text, d.scan_time_utc::text) AS scan_time,
            d.scan_time_utc,
            d.sweep,
            d.latitude,
            d.longitude,
            d.center_x_km,
            d.center_y_km,
            d.confidence_score,
            d.delta_v_kt,
            d.required_delta_v_kt,
            d.radar_range_miles,
            ml.ml_rank,
            ml.ml_score,
            ml.ml_label,
            ST_AsGeoJSON(d.geom)::json AS geometry
        FROM circulation_detections d
        LEFT JOIN ml_best ml
          ON d.track_id = ml.track_id
        WHERE d.event_id = %s
        {time_clause}
        ORDER BY
            COALESCE(d.scan_time::text, d.scan_time_utc::text)::timestamptz ASC NULLS LAST,
            ml.ml_score DESC NULLS LAST,
            d.confidence_score DESC NULLS LAST
        LIMIT %s;
        """,
        tuple(params),
    )

    return make_feature_collection(rows)


@app.get("/events/{event_id}/radar-frame")
def get_radar_frame(event_id: str, time: str):
    """
    Return the Doppler radar PNG overlay metadata for one event scan time.

    The frontend uses this response with Leaflet L.imageOverlay().
    """
    row = fetch_one(
        """
        SELECT
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
            display_range_mi
        FROM radar_frames
        WHERE event_id = %s
          AND status = 'ok'
          AND scan_time = %s::timestamptz
        ORDER BY scan_time
        LIMIT 1;
        """,
        (event_id, time),
    )

    if not row:
        # Fallback to nearest frame within 180 seconds. This protects against
        # tiny formatting differences between event_times and radar_frames.
        row = fetch_one(
            """
            SELECT
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
                display_range_mi
            FROM radar_frames
            WHERE event_id = %s
              AND status = 'ok'
              AND ABS(EXTRACT(EPOCH FROM (scan_time - %s::timestamptz))) <= 180
            ORDER BY ABS(EXTRACT(EPOCH FROM (scan_time - %s::timestamptz)))
            LIMIT 1;
            """,
            (event_id, time, time),
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No radar frame found for {event_id} at {time}",
        )

    image_url_path = row.get("image_url_path") or ""
    if not image_url_path.startswith("/"):
        image_url_path = f"/{image_url_path}"

    return {
        "event_id": row["event_id"],
        "scan_time": row["scan_time"],
        "radar_site": row.get("radar_site"),
        "date_folder": row.get("date_folder"),
        "radar_file": row.get("radar_file"),
        "sweep": row.get("sweep"),
        "product": row.get("product"),
        "field_name": row.get("field_name"),
        "image_url": image_url_path,
        "image_path": row.get("image_path"),
        "bounds": {
            "south": row["south"],
            "west": row["west"],
            "north": row["north"],
            "east": row["east"],
        },
        "display": {
            "vmin_kt": row.get("vmin_kt"),
            "vmax_kt": row.get("vmax_kt"),
            "display_range_mi": row.get("display_range_mi"),
        },
    }


@app.get("/events/{event_id}/best-track")
def get_best_track(event_id: str, time: Optional[str] = None):
    time_clause = ""
    params: List[Any] = [event_id, event_id]

    if time:
        time_clause = """
          AND COALESCE(t.start_time::text, t.start_time_utc::text)::timestamptz <= %s::timestamptz
          AND COALESCE(t.end_time::text, t.end_time_utc::text)::timestamptz >= %s::timestamptz
        """
        params.extend([time, time])

    rows = fetch_all(
        f"""
        WITH ml_best AS (
            SELECT DISTINCT ON (track_id)
                track_id,
                ranked_track_name AS ml_ranked_track_name,
                ml_rank,
                ml_score,
                ml_label,
                tornado_associated,
                model_name
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY track_id, ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
        ),
        ranked_tracks AS (
            SELECT
                t.event_id,
                t.track_id,
                COALESCE(t.ranked_track_name, ml.ml_ranked_track_name) AS ranked_track_name,
                t.track_quality_rank,
                t.track_quality_score,
                t.track_quality_label,
                t.detection_count,
                COALESCE(t.start_time::text, t.start_time_utc::text) AS start_time,
                COALESCE(t.end_time::text, t.end_time_utc::text) AS end_time,
                COALESCE(t.latest_scan_time::text, t.end_time::text, t.end_time_utc::text) AS latest_scan_time,
                t.start_time_utc,
                t.end_time_utc,
                t.duration_min,
                t.speed_kt,
                t.bearing_deg,
                t.mean_confidence_score,
                t.max_confidence_score,
                t.mean_delta_v_kt,
                t.max_delta_v_kt,
                ml.ml_rank,
                ml.ml_score,
                ml.ml_label,
                ml.tornado_associated,
                ml.model_name,
                ST_AsGeoJSON(t.geom)::json AS geometry
            FROM signature_tracks t
            LEFT JOIN ml_best ml
              ON t.track_id = ml.track_id
            WHERE t.event_id = %s
            {time_clause}
        )
        SELECT *
        FROM ranked_tracks
        ORDER BY
            ml_score DESC NULLS LAST,
            track_quality_score DESC NULLS LAST,
            track_quality_rank ASC NULLS LAST
        LIMIT 1;
        """,
        tuple(params),
    )

    return make_feature_collection(rows)


@app.get("/events/{event_id}/best-track/nowcasts")
def get_best_track_nowcasts(event_id: str, time: Optional[str] = None):
    best = fetch_one(
        """
        WITH ml_best AS (
            SELECT DISTINCT ON (track_id)
                track_id,
                ml_score,
                ml_rank
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY track_id, ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
        )
        SELECT t.track_id
        FROM signature_tracks t
        LEFT JOIN ml_best ml
          ON t.track_id = ml.track_id
        WHERE t.event_id = %s
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            t.track_quality_score DESC NULLS LAST,
            t.track_quality_rank ASC NULLS LAST
        LIMIT 1;
        """,
        (event_id, event_id),
    )

    if not best:
        return make_feature_collection([])

    time_clause = ""
    params: List[Any] = [event_id, event_id, best["track_id"]]

    if time:
        time_clause = """
          AND COALESCE(n.source_scan_time::text, n.source_scan_time_utc::text, n.issued_time_utc::text)::timestamptz = %s::timestamptz
        """
        params.append(time)

    rows = fetch_all(
        f"""
        WITH ml_best AS (
            SELECT DISTINCT ON (track_id)
                track_id,
                ml_rank,
                ml_score,
                ml_label,
                tornado_associated,
                model_name
            FROM ml_track_predictions
            WHERE event_id = %s
            ORDER BY track_id, ml_score DESC NULLS LAST, ml_rank ASC NULLS LAST
        )
        SELECT
            n.event_id,
            n.track_id,
            n.ranked_track_name,
            n.track_quality_rank,
            n.track_quality_score,
            n.track_quality_label,
            COALESCE(n.source_scan_time::text, n.source_scan_time_utc::text, n.issued_time_utc::text) AS source_scan_time,
            n.source_scan_time_utc,
            n.issued_time_utc,
            n.projection_min,
            n.valid_time_utc,
            n.speed_kt,
            n.bearing_deg,
            ml.ml_rank,
            ml.ml_score,
            ml.ml_label,
            ml.tornado_associated,
            ml.model_name,
            ST_AsGeoJSON(n.geom)::json AS geometry
        FROM nowcast_paths n
        LEFT JOIN ml_best ml
          ON n.track_id = ml.track_id
        WHERE n.event_id = %s
          AND n.track_id = %s
          {time_clause}
        ORDER BY n.projection_min;
        """,
        tuple(params),
    )

    return make_feature_collection(rows)
