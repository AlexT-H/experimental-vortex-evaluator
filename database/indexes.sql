CREATE INDEX IF NOT EXISTS idx_event_times_event_time
ON event_times (event_id, scan_time);

CREATE INDEX IF NOT EXISTS idx_radar_scans_event_time
ON radar_scans (event_id, scan_time);

CREATE INDEX IF NOT EXISTS idx_radar_scans_event_time_utc
ON radar_scans (event_id, scan_time_utc);

CREATE INDEX IF NOT EXISTS idx_circulation_detections_event_time
ON circulation_detections (event_id, scan_time);

CREATE INDEX IF NOT EXISTS idx_circulation_detections_event_time_utc
ON circulation_detections (event_id, scan_time_utc);

CREATE INDEX IF NOT EXISTS idx_circulation_detections_track
ON circulation_detections (track_id);

CREATE INDEX IF NOT EXISTS idx_circulation_detections_geom
ON circulation_detections USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_signature_tracks_event
ON signature_tracks (event_id);

CREATE INDEX IF NOT EXISTS idx_signature_tracks_quality
ON signature_tracks (event_id, track_quality_score DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_signature_tracks_geom
ON signature_tracks USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_nowcast_paths_event_source_time
ON nowcast_paths (event_id, source_scan_time);

CREATE INDEX IF NOT EXISTS idx_nowcast_paths_track
ON nowcast_paths (track_id);

CREATE INDEX IF NOT EXISTS idx_nowcast_paths_geom
ON nowcast_paths USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_radar_frames_event_time
ON radar_frames (event_id, scan_time);

CREATE INDEX IF NOT EXISTS idx_ml_track_predictions_event_track
ON ml_track_predictions (event_id, track_id);

CREATE INDEX IF NOT EXISTS idx_ml_track_predictions_score
ON ml_track_predictions (event_id, ml_score DESC NULLS LAST);
