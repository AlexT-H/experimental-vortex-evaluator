#!/usr/bin/env python3
"""
Query PostGIS and export API-preview files.

This script reads the E-V-E PostGIS tables loaded by:

    pipeline/11_load_postgis.py

and creates clean, frontend/API-ready outputs.

It proves the database can now power the web preview instead of the frontend
reading raw pipeline files directly.

Default output folder:

    outputs/api_preview/

Main outputs:

    outputs/api_preview/event_summary.json
    outputs/api_preview/tracks_with_scores.geojson
    outputs/api_preview/nowcasts_with_scores.geojson
    outputs/api_preview/best_track.geojson
    outputs/api_preview/best_track_nowcasts.geojson
    outputs/api_preview/high_priority_tracks.geojson
    outputs/api_preview/high_priority_nowcasts.geojson
    outputs/api_preview/detections.geojson

Example:

    docker compose run --rm pipeline python pipeline/12_query_postgis_preview.py \
        --event-id test_case_1

For KBMX/test case:

    docker compose run --rm pipeline python pipeline/12_query_postgis_preview.py \
        --event-id test_case_1

Optional stricter high-priority export:

    docker compose run --rm pipeline python pipeline/12_query_postgis_preview.py \
        --event-id test_case_1 \
        --min-ml-score 0.70 \
        --min-quality-score 65
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:
    raise ImportError(
        "Missing dependency: psycopg2.\n\n"
        "Add this to your pipeline requirements file:\n"
        "  psycopg2-binary\n\n"
        "Then rebuild:\n"
        "  docker compose build pipeline\n"
    ) from exc


ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def default_output_dir(event_id: str) -> Path:
    return ROOT / "outputs" / "cases" / event_id / "api_preview"


def json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def clean_props(row: Dict[str, Any], skip: Sequence[str] = ("geometry",)) -> Dict[str, Any]:
    return {key: json_safe(value) for key, value in row.items() if key not in skip}


def parse_geometry(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if value.strip() == "":
            return None
        return json.loads(value)
    return None


def feature_from_row(row: Dict[str, Any], geometry_key: str = "geometry") -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": parse_geometry(row.get(geometry_key)),
        "properties": clean_props(row, skip=(geometry_key,)),
    }


def feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def write_geojson(path: Path, features: List[Dict[str, Any]]) -> None:
    write_json(path, feature_collection(features))


def db_connect():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "eve"),
        user=os.getenv("POSTGRES_USER", "eve_user"),
        password=os.getenv("POSTGRES_PASSWORD", "eve_password"),
        host=os.getenv("POSTGRES_HOST", "postgis"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def fetch_all(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def fetch_one(conn, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
    rows = fetch_all(conn, sql, params)
    return rows[0] if rows else None


def scalar(conn, sql: str, params: Sequence[Any] = ()) -> Any:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None


def table_exists(conn, table_name: str) -> bool:
    result = scalar(
        conn,
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        );
        """,
        (table_name,),
    )
    return bool(result)


def require_tables(conn) -> None:
    required = [
        "events",
        "circulation_detections",
        "signature_tracks",
        "nowcast_paths",
        "ml_track_predictions",
    ]

    missing = [table for table in required if not table_exists(conn, table)]

    if missing:
        raise RuntimeError(
            "Missing required PostGIS tables: "
            + ", ".join(missing)
            + "\nRun database/schema.sql and pipeline/11_load_postgis.py first."
        )


def get_event_summary(conn, event_id: str) -> Dict[str, Any]:
    event = fetch_one(
        conn,
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

    if event is None:
        event = {
            "event_id": event_id,
            "event_name": None,
            "radar_site": None,
            "date_folder": None,
            "description": "Event row not found, but related tables may still contain records.",
        }

    counts = {}
    for table in [
        "radar_scans",
        "circulation_detections",
        "signature_tracks",
        "nowcast_paths",
        "ml_track_predictions",
    ]:
        counts[table] = int(
            scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE event_id = %s;", (event_id,)) or 0
        )

    best_track = fetch_one(
        conn,
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

    if best_track is None:
        best_track = fetch_one(
            conn,
            """
            SELECT
                t.track_id,
                t.ranked_track_name,
                NULL::integer AS ml_rank,
                NULL::double precision AS ml_score,
                NULL::text AS ml_label,
                t.track_quality_rank,
                t.track_quality_score,
                t.track_quality_label
            FROM signature_tracks t
            WHERE t.event_id = %s
            ORDER BY t.track_quality_score DESC NULLS LAST, t.track_quality_rank ASC NULLS LAST
            LIMIT 1;
            """,
            (event_id,),
        )

    return {
        **{k: json_safe(v) for k, v in event.items()},
        "counts": counts,
        "best_track": {k: json_safe(v) for k, v in best_track.items()} if best_track else None,
        "api_preview_files": {
            "tracks_with_scores": "tracks_with_scores.geojson",
            "nowcasts_with_scores": "nowcasts_with_scores.geojson",
            "best_track": "best_track.geojson",
            "best_track_nowcasts": "best_track_nowcasts.geojson",
            "high_priority_tracks": "high_priority_tracks.geojson",
            "high_priority_nowcasts": "high_priority_nowcasts.geojson",
            "detections": "detections.geojson",
        },
    }


def query_tracks_with_scores(conn, event_id: str) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
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
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            t.track_quality_score DESC NULLS LAST,
            t.track_quality_rank ASC NULLS LAST;
        """,
        (event_id, event_id),
    )


def query_high_priority_tracks(
    conn,
    event_id: str,
    min_ml_score: float,
    min_quality_score: float,
) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
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
          AND (
                COALESCE(ml.ml_score, 0) >= %s
             OR COALESCE(t.track_quality_score, 0) >= %s
          )
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            t.track_quality_score DESC NULLS LAST,
            t.track_quality_rank ASC NULLS LAST;
        """,
        (event_id, event_id, min_ml_score, min_quality_score),
    )


def query_best_track(conn, event_id: str) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
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
        )
        SELECT *
        FROM ranked_tracks
        ORDER BY
            ml_score DESC NULLS LAST,
            track_quality_score DESC NULLS LAST,
            track_quality_rank ASC NULLS LAST
        LIMIT 1;
        """,
        (event_id, event_id),
    )


def query_nowcasts_with_scores(conn, event_id: str) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
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
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            n.track_quality_score DESC NULLS LAST,
            n.track_id,
            n.projection_min;
        """,
        (event_id, event_id),
    )


def query_high_priority_nowcasts(
    conn,
    event_id: str,
    min_ml_score: float,
    min_quality_score: float,
) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
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
          AND (
                COALESCE(ml.ml_score, 0) >= %s
             OR COALESCE(n.track_quality_score, 0) >= %s
          )
        ORDER BY
            ml.ml_score DESC NULLS LAST,
            n.track_quality_score DESC NULLS LAST,
            n.track_id,
            n.projection_min;
        """,
        (event_id, event_id, min_ml_score, min_quality_score),
    )


def query_best_track_nowcasts(conn, event_id: str) -> List[Dict[str, Any]]:
    best = query_best_track(conn, event_id)
    if not best:
        return []

    best_track_id = best[0]["track_id"]

    return fetch_all(
        conn,
        """
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
        ORDER BY n.projection_min;
        """,
        (event_id, event_id, best_track_id),
    )


def query_detections(conn, event_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    params: List[Any] = [event_id, event_id]

    sql = """
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
            d.scan_time_utc,
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
        ORDER BY
            d.scan_time_utc ASC NULLS LAST,
            ml.ml_score DESC NULLS LAST,
            d.confidence_score DESC NULLS LAST
    """

    if limit is not None and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)

    sql += ";"

    return fetch_all(conn, sql, params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PostGIS-backed E-V-E outputs into API-preview GeoJSON/JSON files."
    )
    parser.add_argument(
        "--event-id",
        default=os.getenv("EVENT_ID", "test_case_1"),
        help="Event ID to export, e.g. test_case_1 or test_case_1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where API-preview files will be written. Default: outputs/cases/{event_id}/api_preview.",
    )
    parser.add_argument(
        "--min-ml-score",
        type=float,
        default=0.70,
        help="Minimum ML score for high-priority exports.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=65.0,
        help="Minimum track quality score for high-priority exports.",
    )
    parser.add_argument(
        "--detections-limit",
        type=int,
        default=0,
        help="Optional max number of detections to export. 0 means export all detections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.event_id)

    print("E-V-E — PostGIS API Preview Export")
    print("-----------------------------------------------")
    print(f"Project root:       {ROOT}")
    print(f"Event ID:           {args.event_id}")
    print(f"Output directory:   {output_dir}")
    print(f"High priority ML:   >= {args.min_ml_score}")
    print(f"High priority qual: >= {args.min_quality_score}")
    print()

    conn = db_connect()

    try:
        require_tables(conn)

        postgis_version = scalar(conn, "SELECT PostGIS_Version();")
        print(f"Connected to PostGIS: {postgis_version}")
        print()

        summary = get_event_summary(conn, args.event_id)

        tracks = query_tracks_with_scores(conn, args.event_id)
        high_priority_tracks = query_high_priority_tracks(
            conn,
            args.event_id,
            min_ml_score=args.min_ml_score,
            min_quality_score=args.min_quality_score,
        )
        best_track = query_best_track(conn, args.event_id)

        nowcasts = query_nowcasts_with_scores(conn, args.event_id)
        high_priority_nowcasts = query_high_priority_nowcasts(
            conn,
            args.event_id,
            min_ml_score=args.min_ml_score,
            min_quality_score=args.min_quality_score,
        )
        best_track_nowcasts = query_best_track_nowcasts(conn, args.event_id)

        detection_limit = None if args.detections_limit <= 0 else args.detections_limit
        detections = query_detections(conn, args.event_id, limit=detection_limit)

        write_json(output_dir / "event_summary.json", summary)
        write_geojson(output_dir / "tracks_with_scores.geojson", [feature_from_row(r) for r in tracks])
        write_geojson(output_dir / "high_priority_tracks.geojson", [feature_from_row(r) for r in high_priority_tracks])
        write_geojson(output_dir / "best_track.geojson", [feature_from_row(r) for r in best_track])
        write_geojson(output_dir / "nowcasts_with_scores.geojson", [feature_from_row(r) for r in nowcasts])
        write_geojson(output_dir / "high_priority_nowcasts.geojson", [feature_from_row(r) for r in high_priority_nowcasts])
        write_geojson(output_dir / "best_track_nowcasts.geojson", [feature_from_row(r) for r in best_track_nowcasts])
        write_geojson(output_dir / "detections.geojson", [feature_from_row(r) for r in detections])

        print("\nExport counts:")
        print(f"  tracks_with_scores:       {len(tracks)}")
        print(f"  high_priority_tracks:     {len(high_priority_tracks)}")
        print(f"  best_track:               {len(best_track)}")
        print(f"  nowcasts_with_scores:     {len(nowcasts)}")
        print(f"  high_priority_nowcasts:   {len(high_priority_nowcasts)}")
        print(f"  best_track_nowcasts:      {len(best_track_nowcasts)}")
        print(f"  detections:               {len(detections)}")

        best = summary.get("best_track")
        if best:
            print("\nBest track:")
            print(f"  track_id:            {best.get('track_id')}")
            print(f"  ranked_track_name:   {best.get('ranked_track_name')}")
            print(f"  ml_score:            {best.get('ml_score')}")
            print(f"  ml_rank:             {best.get('ml_rank')}")
            print(f"  ml_label:            {best.get('ml_label')}")
            print(f"  quality_score:       {best.get('track_quality_score')}")
        else:
            print("\nBest track: none found")

        print("\nDone.")

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
