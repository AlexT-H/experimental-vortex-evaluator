CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_name TEXT,
    radar_site TEXT,
    date_folder TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS event_times (
    id SERIAL PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    scan_time TIMESTAMPTZ NOT NULL,
    radar_site TEXT,
    radar_file TEXT,
    sweep INTEGER,
    sort_order INTEGER
);

CREATE TABLE IF NOT EXISTS radar_scans (
    id SERIAL PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    radar_file TEXT,
    radar_site TEXT,
    scan_time TIMESTAMPTZ,
    scan_time_utc TIMESTAMPTZ,
    sweep INTEGER,
    storm_motion_u_kt DOUBLE PRECISION,
    storm_motion_v_kt DOUBLE PRECISION,
    storm_motion_speed_kt DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS circulation_detections (
    detection_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
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
    geom GEOMETRY(Geometry, 4326)
);

CREATE TABLE IF NOT EXISTS signature_tracks (
    track_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
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
    geom GEOMETRY(Geometry, 4326)
);

CREATE TABLE IF NOT EXISTS nowcast_paths (
    id SERIAL PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
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
    geom GEOMETRY(Geometry, 4326)
);

CREATE TABLE IF NOT EXISTS radar_frames (
    id SERIAL PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
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

CREATE TABLE IF NOT EXISTS ml_track_predictions (
    id SERIAL PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
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
