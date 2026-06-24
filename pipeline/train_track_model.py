from pathlib import Path
import argparse
import os
import json
import math
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# E-V-E
# E-V-E — Train Simple Track-Scoring ML Model
#
# Purpose:
#   Train a simple ML model that scores radar-derived circulation tracks based
#   on whether they resemble manually/weakly labeled tornado-associated tracks.
#
# Responsible framing:
#   This is NOT a direct tornado prediction model. It is an experimental
#   tornado-associated circulation scoring model built from derived track
#   features.
#
# Inputs:
#   outputs/ml/track_feature_table.csv
#
# Outputs:
#   outputs/ml/track_model_predictions.csv
#   outputs/ml/model_metrics.json
#   models/eve_track_scoring_model.joblib
#   outputs/figures/ml_feature_importance.png
#   outputs/figures/ml_track_scores.png
# =============================================================================


from case_config import case_dir, resolve_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = os.getenv("EVENT_ID", "test_case_1")
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "eve_track_scoring_model.joblib"


def configure_case(event_id: str) -> None:
    """Point ML predictions/figures at this event's isolated case folder."""
    global EVENT_ID, CASE_OUT_DIR, ML_OUT_DIR, FIGURE_OUT_DIR, GEOJSON_DIR
    global FEATURE_TABLE_CSV, PREDICTIONS_CSV, METRICS_JSON
    global FEATURE_IMPORTANCE_FIG, TRACK_SCORES_FIG, TRACKS_GEOJSON, TRACKS_WITH_ML_GEOJSON

    EVENT_ID = event_id
    CASE_OUT_DIR = case_dir(EVENT_ID, PROJECT_ROOT)
    ML_OUT_DIR = CASE_OUT_DIR / "ml"
    FIGURE_OUT_DIR = CASE_OUT_DIR / "figures"
    GEOJSON_DIR = CASE_OUT_DIR / "geojson"

    FEATURE_TABLE_CSV = ML_OUT_DIR / "track_feature_table.csv"
    PREDICTIONS_CSV = ML_OUT_DIR / "track_model_predictions.csv"
    METRICS_JSON = ML_OUT_DIR / "model_metrics.json"

    FEATURE_IMPORTANCE_FIG = FIGURE_OUT_DIR / "ml_feature_importance.png"
    TRACK_SCORES_FIG = FIGURE_OUT_DIR / "ml_track_scores.png"

    TRACKS_GEOJSON = GEOJSON_DIR / "banded_tvs_tracks.geojson"
    TRACKS_WITH_ML_GEOJSON = GEOJSON_DIR / "banded_tvs_tracks_with_ml_scores.geojson"


configure_case(EVENT_ID)

LABEL_COLUMN = "tornado_associated"

FEATURE_COLUMNS = [
    "detection_count",
    "duration_min",
    "persistence_min",
    "path_length_km",
    "speed_kt",
    "mean_step_distance_km",
    "max_step_distance_km",
    "mean_step_speed_kt",
    "max_step_speed_kt",
    "mean_turn_angle_deg",
    "max_turn_angle_deg",
    "mean_confidence_score",
    "max_confidence_score",
    "mean_delta_v_kt",
    "max_delta_v_kt",
    "track_quality_score",
    "quality_persistence_score",
    "quality_duration_score",
    "quality_confidence_score",
    "quality_delta_v_score",
    "quality_speed_score",
    "quality_smoothness_score",
    "quality_penalty",
    "point_mean_confidence_score",
    "point_min_confidence_score",
    "point_mean_delta_v_kt",
    "point_min_delta_v_kt",
    "point_mean_required_delta_v_kt",
    "mean_radar_range_miles",
    "max_radar_range_miles",
    "mean_lobe_separation_km",
    "max_lobe_separation_km",
    "mean_opposition_score",
    "mean_banded_score",
    "mean_neutral_center_score",
    "mean_reflectivity_dbz",
    "max_reflectivity_dbz",
    "mean_storm_motion_speed_kt",
    "nowcast_projection_count",
    "max_projection_min",
    "delta_v_growth_kt",
    "confidence_growth",
    "confidence_strength",
    "persistence_strength",
    "quality_conf_delta_combo",
    "smoothness_penalty_feature",
    "delta_v_over_required",
    "path_efficiency",
]


def ensure_dirs():
    ML_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def import_sklearn():
    try:
        from joblib import dump
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "This script requires scikit-learn and joblib.\n"
            "Install them in your pipeline environment, for example:\n"
            "  pip install scikit-learn joblib\n"
            "or add them to your Docker requirements file."
        ) from exc

    return {
        "dump": dump,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "StratifiedKFold": StratifiedKFold,
        "cross_val_predict": cross_val_predict,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def load_feature_table():
    if not FEATURE_TABLE_CSV.exists():
        raise FileNotFoundError(
            f"Missing feature table: {FEATURE_TABLE_CSV}\n"
            "Run pipeline/09_build_ml_features.py first."
        )

    df = pd.read_csv(FEATURE_TABLE_CSV)

    if df.empty:
        raise ValueError(f"{FEATURE_TABLE_CSV} is empty.")

    if "track_id" not in df.columns:
        raise ValueError("Feature table must contain track_id.")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Feature table must contain {LABEL_COLUMN}. "
            "Fill data/manual/track_labels.csv and rerun pipeline/09_build_ml_features.py."
        )

    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    return df


def prepare_training_data(df):
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]

    if not available_features:
        raise ValueError("No expected numeric feature columns were found in the feature table.")

    labeled = df[df[LABEL_COLUMN].notna()].copy()

    if labeled.empty:
        raise ValueError(
            "No labeled rows found.\n"
            "Edit data/manual/track_labels.csv, set tornado_associated to 1 or 0, "
            "then rerun pipeline/09_build_ml_features.py."
        )

    labeled[LABEL_COLUMN] = labeled[LABEL_COLUMN].astype(int)

    class_counts = labeled[LABEL_COLUMN].value_counts().to_dict()

    if len(class_counts) < 2:
        raise ValueError(
            "Training requires at least one positive label and one negative label.\n"
            "Example: set the validated circulation track to 1 and several false/secondary tracks to 0."
        )

    X = labeled[available_features].apply(pd.to_numeric, errors="coerce")
    y = labeled[LABEL_COLUMN].astype(int)

    X_all = df[available_features].apply(pd.to_numeric, errors="coerce")

    return labeled, X, y, X_all, available_features, class_counts


def evaluate_model(model, X, y, skl):
    metrics = {
        "evaluation_mode": "train_on_all_only",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "confusion_matrix": None,
        "notes": [],
    }

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    n_samples = len(y)

    if n_samples < 6 or min_class_count < 2:
        metrics["notes"].append(
            "Dataset is too small or class balance is too limited for reliable cross-validation. "
            "Metrics are not reported as validation performance."
        )
        return metrics

    n_splits = min(5, min_class_count)

    try:
        cv = skl["StratifiedKFold"](n_splits=n_splits, shuffle=True, random_state=42)
        pred = skl["cross_val_predict"](model, X, y, cv=cv, method="predict")
        proba = skl["cross_val_predict"](model, X, y, cv=cv, method="predict_proba")[:, 1]

        metrics["evaluation_mode"] = f"{n_splits}-fold stratified cross-validation"
        metrics["accuracy"] = float(skl["accuracy_score"](y, pred))
        metrics["precision"] = float(skl["precision_score"](y, pred, zero_division=0))
        metrics["recall"] = float(skl["recall_score"](y, pred, zero_division=0))
        metrics["f1"] = float(skl["f1_score"](y, pred, zero_division=0))
        metrics["confusion_matrix"] = skl["confusion_matrix"](y, pred).tolist()

        try:
            metrics["roc_auc"] = float(skl["roc_auc_score"](y, proba))
        except Exception:
            metrics["roc_auc"] = None

    except Exception as exc:
        metrics["notes"].append(f"Cross-validation failed: {exc}")

    return metrics


def make_models(skl):
    random_forest = skl["Pipeline"](
        steps=[
            ("imputer", skl["SimpleImputer"](strategy="median")),
            (
                "model",
                skl["RandomForestClassifier"](
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    logistic = skl["Pipeline"](
        steps=[
            ("imputer", skl["SimpleImputer"](strategy="median")),
            ("scaler", skl["StandardScaler"]()),
            (
                "model",
                skl["LogisticRegression"](
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return random_forest, logistic


def get_feature_importance(fitted_model, feature_columns):
    try:
        rf = fitted_model.named_steps["model"]
        importances = rf.feature_importances_
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame({"feature": feature_columns, "importance": [0.0] * len(feature_columns)})


def save_feature_importance_plot(importances):
    if importances.empty:
        return

    top = importances.head(15).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Circulation Track ML Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FIG, dpi=200, bbox_inches="tight")
    plt.close()


def save_track_scores_plot(predictions):
    plot_df = predictions.sort_values("ml_score", ascending=True)

    plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
    plt.barh(plot_df["track_id"], plot_df["ml_score"])
    plt.title("ML Tornado-Associated Circulation Score by Track")
    plt.xlabel("ML score")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(TRACK_SCORES_FIG, dpi=200, bbox_inches="tight")
    plt.close()


def update_track_geojson(predictions):
    if not TRACKS_GEOJSON.exists():
        return False

    with TRACKS_GEOJSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pred_lookup = predictions.set_index("track_id").to_dict(orient="index")

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        track_id = props.get("track_id")

        if track_id in pred_lookup:
            row = pred_lookup[track_id]
            props["ml_score"] = float(row["ml_score"])
            props["ml_rank"] = int(row["ml_rank"])
            props["ml_label"] = row["ml_label"]
            props["model_name"] = row["model_name"]

    with TRACKS_WITH_ML_GEOJSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return True



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/score E-V-E tracks for one event case.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID", EVENT_ID))
    return parser.parse_args()


def main():
    args = parse_args()
    case = resolve_case(args.event_id)
    configure_case(case.event_id)
    ensure_dirs()
    skl = import_sklearn()

    df = load_feature_table()
    labeled, X, y, X_all, feature_columns, class_counts = prepare_training_data(df)

    random_forest, logistic = make_models(skl)

    # Random forest is the main version because it handles small nonlinear tabular data well.
    final_model = random_forest
    model_name = "RandomForestClassifier"

    metrics = evaluate_model(final_model, X, y, skl)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X, y)

    all_scores = final_model.predict_proba(X_all)[:, 1]

    predictions = df[[
        col for col in [
            "track_id",
            "ranked_track_name",
            "track_quality_rank",
            "track_quality_score",
            "track_quality_label",
            LABEL_COLUMN,
            "label_notes",
        ]
        if col in df.columns
    ]].copy()

    predictions["ml_score"] = all_scores
    predictions["ml_score"] = predictions["ml_score"].round(4)
    predictions["model_name"] = model_name
    predictions["ml_label"] = predictions["ml_score"].apply(
        lambda score: "high" if score >= 0.70 else "medium" if score >= 0.40 else "low"
    )

    predictions = predictions.sort_values("ml_score", ascending=False).reset_index(drop=True)
    predictions["ml_rank"] = range(1, len(predictions) + 1)

    ordered_cols = [
        "track_id",
        "ranked_track_name",
        "ml_rank",
        "ml_score",
        "ml_label",
        "track_quality_rank",
        "track_quality_score",
        "track_quality_label",
        LABEL_COLUMN,
        "label_notes",
        "model_name",
    ]
    ordered_cols = [col for col in ordered_cols if col in predictions.columns]
    predictions = predictions[ordered_cols]

    predictions.to_csv(PREDICTIONS_CSV, index=False)

    importances = get_feature_importance(final_model, feature_columns)
    importances.to_csv(ML_OUT_DIR / "feature_importance.csv", index=False)

    save_feature_importance_plot(importances)
    save_track_scores_plot(predictions)

    skl["dump"](
        {
            "model": final_model,
            "model_name": model_name,
            "feature_columns": feature_columns,
            "label_column": LABEL_COLUMN,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    geojson_updated = update_track_geojson(predictions)

    metrics_out = {
        "model_name": model_name,
        "label_column": LABEL_COLUMN,
        "feature_columns": feature_columns,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "labeled_rows": int(len(labeled)),
        "total_rows_scored": int(len(df)),
        "metrics": metrics,
        "responsible_framing": (
            "This is an experimental tornado-associated circulation scoring model. "
            "It scores radar-derived tracks; it is not an official forecast or warning model."
        ),
    }

    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print("=" * 110)
    print("E-V-E — Train Simple Track-Scoring ML Model")
    print("=" * 110)
    print(f"Feature table:       {FEATURE_TABLE_CSV}")
    print(f"Predictions:         {PREDICTIONS_CSV}")
    print(f"Model:               {MODEL_PATH}")
    print(f"Metrics:             {METRICS_JSON}")
    print(f"Feature importance:  {FEATURE_IMPORTANCE_FIG}")
    print(f"Track score plot:    {TRACK_SCORES_FIG}")
    if geojson_updated:
        print(f"Tracks + ML GeoJSON: {TRACKS_WITH_ML_GEOJSON}")
    print()
    print("Top ML-ranked tracks:")
    for _, row in predictions.head(10).iterrows():
        print(
            f"#{int(row['ml_rank']):02d} {row['track_id']} | "
            f"ml_score={row['ml_score']} | "
            f"quality={row.get('track_quality_score', 'NA')}"
        )
    print()
    print("Note:")
    print("  With one event or only a few manual labels, this proves the ML pipeline works,")
    print("  but the score should be presented as experimental, not operationally validated.")


if __name__ == "__main__":
    main()
