from pathlib import Path
import argparse
import os
import csv
import math
import sys

import pandas as pd


# =============================================================================
# E-V-E
# E-V-E — Build Track-Level ML Feature Table
#
# Purpose:
#   Create a simple ML-ready table where each row is one tracked circulation.
#   This does NOT train the model. It prepares features and merges manual labels.
#
# Inputs:
#   outputs/tracks/banded_tvs_tracks.csv
#   outputs/tracks/banded_tvs_track_points.csv              optional but useful
#   outputs/nowcasts/banded_tvs_nowcasts_15_30.csv          optional
#   data/manual/track_labels.csv                            manual label file
#
# Outputs:
#   outputs/ml/track_feature_table.csv
#   outputs/ml/unlabeled_track_feature_table.csv
#   data/manual/track_labels.csv                            created if missing
# =============================================================================


from case_config import case_dir, resolve_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = os.getenv("EVENT_ID", "test_case_1")


def configure_case(event_id: str) -> None:
    """Point all ML feature outputs at this event's isolated case folder."""
    global EVENT_ID, CASE_OUT_DIR, TRACKS_CSV, TRACK_POINTS_CSV, NOWCASTS_CSV
    global MANUAL_LABEL_DIR, MANUAL_LABELS_CSV, ML_OUT_DIR, FEATURE_TABLE_CSV, UNLABELED_FEATURE_TABLE_CSV

    EVENT_ID = event_id
    CASE_OUT_DIR = case_dir(EVENT_ID, PROJECT_ROOT)
    TRACKS_CSV = CASE_OUT_DIR / "tracks" / "banded_tvs_tracks.csv"
    TRACK_POINTS_CSV = CASE_OUT_DIR / "tracks" / "banded_tvs_track_points.csv"
    NOWCASTS_CSV = CASE_OUT_DIR / "nowcasts" / "banded_tvs_nowcasts_15_30.csv"

    MANUAL_LABEL_DIR = CASE_OUT_DIR / "manual"
    MANUAL_LABELS_CSV = MANUAL_LABEL_DIR / "track_labels.csv"

    ML_OUT_DIR = CASE_OUT_DIR / "ml"
    FEATURE_TABLE_CSV = ML_OUT_DIR / "track_feature_table.csv"
    UNLABELED_FEATURE_TABLE_CSV = ML_OUT_DIR / "unlabeled_track_feature_table.csv"


configure_case(EVENT_ID)

LABEL_COLUMN = "tornado_associated"


def ensure_dirs():
    MANUAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    ML_OUT_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            f"Run the previous pipeline phase first, or check that the path is correct."
        )


def to_number(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return numerator / denominator


def create_label_template_if_missing(track_ids):
    if MANUAL_LABELS_CSV.exists():
        return False

    rows = []
    for track_id in track_ids:
        rows.append(
            {
                "track_id": track_id,
                LABEL_COLUMN: "",
                "label_notes": "Set 1 for tornado-associated / validated main circulation track, 0 for non-tornado-associated or false/secondary track.",
            }
        )

    with MANUAL_LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", LABEL_COLUMN, "label_notes"])
        writer.writeheader()
        writer.writerows(rows)

    return True


def load_labels():
    if not MANUAL_LABELS_CSV.exists():
        return pd.DataFrame(columns=["track_id", LABEL_COLUMN, "label_notes"])

    labels = pd.read_csv(MANUAL_LABELS_CSV)

    if "track_id" not in labels.columns:
        raise ValueError(f"{MANUAL_LABELS_CSV} must contain a track_id column.")

    if LABEL_COLUMN not in labels.columns:
        labels[LABEL_COLUMN] = ""

    if "label_notes" not in labels.columns:
        labels["label_notes"] = ""

    labels[LABEL_COLUMN] = pd.to_numeric(labels[LABEL_COLUMN], errors="coerce")
    return labels[["track_id", LABEL_COLUMN, "label_notes"]]


def aggregate_track_points():
    if not TRACK_POINTS_CSV.exists():
        return pd.DataFrame(columns=["track_id"])

    points = pd.read_csv(TRACK_POINTS_CSV)

    if points.empty or "track_id" not in points.columns:
        return pd.DataFrame(columns=["track_id"])

    numeric_candidates = [
        "confidence_score",
        "delta_v_kt",
        "required_delta_v_kt",
        "radar_range_km",
        "radar_range_miles",
        "lobe_separation_km",
        "range_difference_km",
        "opposition_score",
        "banded_score",
        "neutral_center_score",
        "side_balance_score",
        "window_max_reflectivity_dbz",
        "storm_motion_speed_kt",
    ]

    for col in numeric_candidates:
        if col in points.columns:
            points[col] = to_number(points[col])

    aggs = {}

    if "confidence_score" in points.columns:
        aggs["point_mean_confidence_score"] = ("confidence_score", "mean")
        aggs["point_min_confidence_score"] = ("confidence_score", "min")

    if "delta_v_kt" in points.columns:
        aggs["point_mean_delta_v_kt"] = ("delta_v_kt", "mean")
        aggs["point_min_delta_v_kt"] = ("delta_v_kt", "min")

    if "required_delta_v_kt" in points.columns:
        aggs["point_mean_required_delta_v_kt"] = ("required_delta_v_kt", "mean")

    if "radar_range_miles" in points.columns:
        aggs["mean_radar_range_miles"] = ("radar_range_miles", "mean")
        aggs["max_radar_range_miles"] = ("radar_range_miles", "max")

    if "lobe_separation_km" in points.columns:
        aggs["mean_lobe_separation_km"] = ("lobe_separation_km", "mean")
        aggs["max_lobe_separation_km"] = ("lobe_separation_km", "max")

    if "opposition_score" in points.columns:
        aggs["mean_opposition_score"] = ("opposition_score", "mean")

    if "banded_score" in points.columns:
        aggs["mean_banded_score"] = ("banded_score", "mean")

    if "neutral_center_score" in points.columns:
        aggs["mean_neutral_center_score"] = ("neutral_center_score", "mean")

    if "window_max_reflectivity_dbz" in points.columns:
        aggs["mean_reflectivity_dbz"] = ("window_max_reflectivity_dbz", "mean")
        aggs["max_reflectivity_dbz"] = ("window_max_reflectivity_dbz", "max")

    if "storm_motion_speed_kt" in points.columns:
        aggs["mean_storm_motion_speed_kt"] = ("storm_motion_speed_kt", "mean")

    if not aggs:
        return pd.DataFrame({"track_id": sorted(points["track_id"].unique())})

    return points.groupby("track_id").agg(**aggs).reset_index()


def aggregate_nowcasts():
    if not NOWCASTS_CSV.exists():
        return pd.DataFrame(columns=["track_id"])

    nowcasts = pd.read_csv(NOWCASTS_CSV)

    if nowcasts.empty or "track_id" not in nowcasts.columns:
        return pd.DataFrame(columns=["track_id"])

    if "projection_min" in nowcasts.columns:
        nowcasts["projection_min"] = to_number(nowcasts["projection_min"])
        proj = nowcasts.groupby("track_id").agg(
            nowcast_projection_count=("projection_min", "count"),
            max_projection_min=("projection_min", "max"),
        ).reset_index()
        return proj

    return pd.DataFrame({"track_id": sorted(nowcasts["track_id"].unique())})


def build_features():
    require_file(TRACKS_CSV, "Phase 5 track summary CSV")

    tracks = pd.read_csv(TRACKS_CSV)

    if tracks.empty:
        raise ValueError(f"{TRACKS_CSV} is empty. Run Phase 5 again and check its outputs.")

    if "track_id" not in tracks.columns:
        raise ValueError(f"{TRACKS_CSV} must contain a track_id column.")

    created_template = create_label_template_if_missing(tracks["track_id"].tolist())

    points_agg = aggregate_track_points()
    nowcast_agg = aggregate_nowcasts()
    labels = load_labels()

    features = tracks.copy()

    if not points_agg.empty:
        features = features.merge(points_agg, on="track_id", how="left")

    if not nowcast_agg.empty:
        features = features.merge(nowcast_agg, on="track_id", how="left")

    features = features.merge(labels, on="track_id", how="left")

    numeric_defaults = {
        "detection_count": 0,
        "duration_min": 0,
        "persistence_min": 0,
        "path_length_km": 0,
        "speed_kt": 0,
        "bearing_deg": 0,
        "mean_step_distance_km": 0,
        "max_step_distance_km": 0,
        "mean_step_speed_kt": 0,
        "max_step_speed_kt": 0,
        "mean_turn_angle_deg": 0,
        "max_turn_angle_deg": 0,
        "mean_confidence_score": 0,
        "max_confidence_score": 0,
        "mean_delta_v_kt": 0,
        "max_delta_v_kt": 0,
        "track_quality_score": 0,
        "track_quality_rank": 999,
        "quality_persistence_score": 0,
        "quality_duration_score": 0,
        "quality_confidence_score": 0,
        "quality_delta_v_score": 0,
        "quality_speed_score": 0,
        "quality_smoothness_score": 0,
        "quality_penalty": 0,
        "point_mean_confidence_score": 0,
        "point_min_confidence_score": 0,
        "point_mean_delta_v_kt": 0,
        "point_min_delta_v_kt": 0,
        "point_mean_required_delta_v_kt": 0,
        "mean_radar_range_miles": 0,
        "max_radar_range_miles": 0,
        "mean_lobe_separation_km": 0,
        "max_lobe_separation_km": 0,
        "mean_opposition_score": 0,
        "mean_banded_score": 0,
        "mean_neutral_center_score": 0,
        "mean_reflectivity_dbz": 0,
        "max_reflectivity_dbz": 0,
        "mean_storm_motion_speed_kt": 0,
        "nowcast_projection_count": 0,
        "max_projection_min": 0,
    }

    for col, default in numeric_defaults.items():
        if col not in features.columns:
            features[col] = default
        features[col] = to_number(features[col], default=default)

    # Derived ML features.
    features["delta_v_growth_kt"] = features["max_delta_v_kt"] - features["mean_delta_v_kt"]
    features["confidence_growth"] = features["max_confidence_score"] - features["mean_confidence_score"]
    features["confidence_strength"] = features["mean_confidence_score"] * features["detection_count"]
    features["persistence_strength"] = features["duration_min"] * features["mean_delta_v_kt"]
    features["quality_conf_delta_combo"] = features["track_quality_score"] * features["mean_delta_v_kt"]
    features["smoothness_penalty_feature"] = features["mean_turn_angle_deg"] + features["max_step_distance_km"]
    features["delta_v_over_required"] = features.apply(
        lambda row: safe_divide(row["point_mean_delta_v_kt"], row["point_mean_required_delta_v_kt"]),
        axis=1,
    )
    features["path_efficiency"] = features.apply(
        lambda row: safe_divide(
            math.hypot(row.get("end_x_km", 0) - row.get("start_x_km", 0), row.get("end_y_km", 0) - row.get("start_y_km", 0)),
            row["path_length_km"],
        ),
        axis=1,
    )

    if "track_quality_label" not in features.columns:
        features["track_quality_label"] = "unknown"

    if "ranked_track_name" not in features.columns:
        features["ranked_track_name"] = features["track_id"]

    # Put important identifiers and label columns first.
    first_cols = [
        "track_id",
        "ranked_track_name",
        "track_quality_rank",
        "track_quality_score",
        "track_quality_label",
        LABEL_COLUMN,
        "label_notes",
    ]

    remaining_cols = [col for col in features.columns if col not in first_cols]
    features = features[first_cols + remaining_cols]

    features.to_csv(FEATURE_TABLE_CSV, index=False)

    unlabeled = features[features[LABEL_COLUMN].isna()].copy()
    unlabeled.to_csv(UNLABELED_FEATURE_TABLE_CSV, index=False)

    return features, created_template, unlabeled



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E-V-E ML features for one event case.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID", EVENT_ID))
    return parser.parse_args()


def main():
    args = parse_args()
    case = resolve_case(args.event_id)
    configure_case(case.event_id)
    ensure_dirs()

    features, created_template, unlabeled = build_features()

    labeled_count = int(features[LABEL_COLUMN].notna().sum())
    positive_count = int((features[LABEL_COLUMN] == 1).sum())
    negative_count = int((features[LABEL_COLUMN] == 0).sum())

    print("=" * 110)
    print("E-V-E — Build Track-Level ML Feature Table")
    print("=" * 110)
    print(f"Input tracks:       {TRACKS_CSV}")
    print(f"Output features:    {FEATURE_TABLE_CSV}")
    print(f"Unlabeled features: {UNLABELED_FEATURE_TABLE_CSV}")
    print(f"Manual labels:      {MANUAL_LABELS_CSV}")
    print()
    print(f"Rows:               {len(features)}")
    print(f"Labeled rows:       {labeled_count}")
    print(f"Positive labels:    {positive_count}")
    print(f"Negative labels:    {negative_count}")
    print(f"Unlabeled rows:     {len(unlabeled)}")

    if created_template:
        print("\nA manual label template was created.")
        print("Edit it before training the ML model:")
        print(f"  {MANUAL_LABELS_CSV}")
        print("\nExample:")
        print("  TRK_006,1,visually confirmed main circulation track")
        print("  TRK_001,0,false or secondary track")

    if labeled_count == 0:
        print("\nNo labels are filled in yet. Fill data/manual/track_labels.csv, rerun this script, then run Phase 7.")
    else:
        print("\nFeature table is ready for Phase 7 training.")


if __name__ == "__main__":
    main()
