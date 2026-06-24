# E.V.E. Experimental Vortex Evaluator

**E.V.E.** is a PostGIS-backed radar analysis application that ingests historical NEXRAD Level II radar data, detects storm-relative velocity couplets, links detections into ranked circulation tracks, generates short-range projected paths, applies machine-learning track scoring, and visualizes the results in an interactive React and Leaflet web map with Doppler radar scan overlays.

The demo version is focused on one validated historical case:

```text
test_case_2: KBMX 2011-04-27 Central Alabama Storm Outbreak
```

The application is intentionally configured around one polished case study so the public demo presents the strongest and most defensible output from the project.

**To run your own version of E-V-E with different storm data, follow the directions included in:**
```
docs/run_new_storm.md
```

## Responsible use

E.V.E. is an experimental portfolio project for radar analysis, geospatial processing, machine-learning scoring, and web GIS visualization. It is not an official forecast, warning system, or public-safety product. Users should rely on National Weather Service warnings and official emergency guidance.

## Project overview

E.V.E. connects several technical layers into one reproducible geospatial workflow:

```text
Raw NEXRAD Level II radar files
        ↓
Python radar-processing pipeline
        ↓
Storm-relative TVS and velocity-couplet candidate detection
        ↓
Cross-scan circulation tracking
        ↓
15 and 30 minute nowcast projection
        ↓
ML-based track scoring
        ↓
PostGIS spatial database
        ↓
FastAPI backend
        ↓
React and Leaflet frontend
```

The project demonstrates:

- radar-data ingestion from historical NEXRAD Level II files,
- velocity and reflectivity processing with Python,
- storm-relative TVS-style candidate detection,
- circulation tracking across multiple radar scans,
- short-range 15 and 30 minute nowcast generation,
- tabular ML scoring for already-detected circulation tracks,
- PostGIS storage for detections, tracks, nowcasts, radar frames, and ML outputs,
- FastAPI endpoints for serving geospatial layers,
- a React and Leaflet dashboard with radar-frame overlays, layer controls, scan-time navigation, feature inspection, and mobile sidebar support.

## Demo behavior

The web application is designed as a focused case-study demo rather than a general storm browser.

Current behavior:

```text
- The app loads test_case_2 automatically.
- There is no public test-case selector.
- Full storm tracks remain visible for the selected case.
- Full nowcast paths remain visible for the selected case.
- Detection points are cumulative through the selected radar scan.
- The radar velocity image changes with the time slider.
- Priority tracks require ml_score >= 0.90.
- The map centers on the associated Doppler radar site.
- The sidebar is toggleable on mobile.
- The browser tab uses the E.V.E. favicon.
```

Expected layer behavior:

```text
Tracks       -> full event track layer
Nowcasts     -> full event 15 and 30 minute projection layer
Detections   -> cumulative through selected scan time
Radar image  -> selected scan only
```

## Tech stack

```text
Python
Py-ART
NumPy
Pandas
Matplotlib
scikit-learn and joblib
PostgreSQL and PostGIS
FastAPI
Docker Compose
React
Vite
Leaflet
```

## Repository structure

```text
EVE/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── pipeline/
│   ├── run_pipeline.py
│   ├── case_config.py
│   ├── download_nexrad_event.py
│   ├── detect_tvs_candidates.py
│   ├── track_circulations.py
│   ├── build_ml_features.py
│   ├── train_track_model.py
│   ├── render_radar_frames.py
│   ├── load_postgis.py
│   └── export_api_preview.py
│
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── database/
│   ├── schema.sql
│   └── indexes.sql
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── public/
│   │   └── favicon.svg
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── main.jsx
│       └── index.css
│
└── docs/
    ├── run_new_storm.md
    └── methodology.md
```

## Local run guide

These commands assume Docker Compose is configured and the project contains the retained `test_case_2` outputs.

### 1. Start PostGIS

```bash
docker compose up -d postgis
```

### 2. Start the backend

```bash
docker compose up -d backend
```

The backend can also be restarted with:

```bash
docker compose restart backend
```

### 3. Verify backend health

Open:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### 4. Verify the retained event

Open:

```text
http://localhost:8000/events
```

The response should include:

```text
test_case_2
```

### 5. Verify event layers

Open:

```text
http://localhost:8000/events/test_case_2/summary
http://localhost:8000/events/test_case_2/times
http://localhost:8000/events/test_case_2/tracks
http://localhost:8000/events/test_case_2/detections
http://localhost:8000/events/test_case_2/nowcasts
```

### 6. Verify radar-frame serving

Use a scan time from:

```text
http://localhost:8000/events/test_case_2/times
```

Then request:

```text
http://localhost:8000/events/test_case_2/radar-frame?time=...
```

The response should include:

```text
image_url
bounds
```

Open the returned `image_url` directly. If the PNG does not load, the backend cannot see the radar-frame output directory.

### 7. Run the frontend

From the frontend folder:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Main API endpoints

```text
GET /health
GET /events
GET /events/{event_id}/summary
GET /events/{event_id}/times
GET /events/{event_id}/tracks
GET /events/{event_id}/detections
GET /events/{event_id}/nowcasts
GET /events/{event_id}/best-track
GET /events/{event_id}/best-track/nowcasts
GET /events/{event_id}/radar-frame?time=...
GET /radar_frames/{event_id}/{filename}
GET /debug/radar-frame-paths
```

## PostGIS tables

The processed radar outputs are loaded into PostGIS tables such as:

```text
events
event_times
radar_scans
circulation_detections
signature_tracks
nowcast_paths
ml_track_predictions
radar_frames
```

These tables allow the API to serve event metadata, scan times, detection points, track geometries, nowcast geometries, ML scores, and radar-frame metadata.

## Repository hygiene

Keep source code, configuration examples, schema files, and documentation:

```text
README.md
.env.example
.gitignore
docker-compose.yml
pipeline/
backend/
database/
frontend/
docs/
```

Avoid committing large raw or generated files:

```text
data/raw/
outputs/cases/
outputs/web/radar_frames/
outputs/candidates/
outputs/tracks/
outputs/nowcasts/
outputs/geojson/
outputs/logs/
outputs/ml/
```

Avoid committing temporary development files:

```text
__pycache__/
*.pyc
*_old.py
*_backup.py
*_fixed.py
*_working.py
files with (1), (2), (3), etc. in the name
old ZIP patch files
temporary frontend variants
```

Screenshots, short videos, architecture diagrams, and compact sample exports are preferable to full generated radar datasets for repository presentation.

## Known limitations

E.V.E. performs best on curated radar cases with:

```text
one dominant compact velocity couplet
a fairly discrete storm
clear scan-to-scan continuity
a short event window
few nearby competing circulations
moderate radar range
```

It performs worse with:

```text
multiple nearby spinups
broad rotation
large wedge or EF5-scale circulations
messy storm mode
QLCS-like structure
rapidly cycling circulations
multiple competing candidates per scan
```

Current limitations include:

- plausible false vortex tracks can occur,
- separate nearby circulations can be merged into one path,
- broad circulations can be double-counted,
- event choice strongly affects output quality,
- the ML model is a simple prioritization layer, not a public-safety warning model.

E.V.E. is a prototype and performs best on curated radar cases with compact, dominant velocity couplets. Complex storm modes, broad circulations, and multiple nearby vortex signatures can produce false tracks, merged tracks, or duplicate circulation paths.

## Project summary

E.V.E. is a PostGIS-backed radar analysis application that ingests raw NEXRAD Level II data, detects storm-relative velocity couplets, links them into ranked circulation tracks, generates short-range projected paths, applies ML-based track scoring, and visualizes the result in an interactive Leaflet web map with Doppler radar scan overlays.
