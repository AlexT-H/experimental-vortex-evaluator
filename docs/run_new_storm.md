# Running E.V.E. on a Different Storm

This guide explains how to run E.V.E. on a different historical storm event. It uses a KINX radar event from May 2024 as an example, but the same process can be adapted to other NEXRAD radar sites and event windows.

E.V.E. is designed for historical radar-event analysis. It downloads NEXRAD Level II radar files, detects storm-relative circulation candidates, links detections into tracks, generates 15 and 30 minute nowcast paths, scores tracks with a machine-learning model, loads outputs into PostGIS, and serves the results through the backend and frontend.

## Example storm configuration

The example below uses the KINX radar site for a northeast Oklahoma tornadic supercell.

```text
Radar site: KINX
Radar name: Tulsa, Oklahoma NEXRAD
Event date folder: 2024-05-07
Approximate event window: 01:30 to 03:30 UTC
Recommended sweep: 1
Suggested maximum downloads: 10
```

This event crosses the local evening-to-UTC date boundary, so the radar archive date is `2024-05-07` even though the storm occurred during the local evening of May 6, 2024.

## Before running a new storm

Before processing another event, make sure Docker is running and the project dependencies are available.

From the project root:

```bash
docker compose up -d postgis
```

Verify that PostGIS is working:

```bash
docker compose exec postgis psql -U vortex_user -d vortex_lab -c "SELECT PostGIS_Version();"
```

The project should also include the standard folders:

```text
pipeline/
backend/
frontend/
database/
data/
outputs/
models/
```

## Step 1: Choose a new event ID

Each storm example needs a unique event ID so its detections, tracks, nowcasts, radar frames, and ML outputs do not overwrite another event.

Use a clear lowercase identifier, for example:

```text
kinx_2024_05_07
```

For another storm, use a similar format:

```text
radar_yyyy_mm_dd
```

Examples:

```text
kdmx_2024_05_21
kilx_2013_11_17
kbmx_2011_04_27
```

## Step 2: Run the pipeline until the labeling step

The first run downloads radar files, detects circulation candidates, builds tracks, generates nowcasts, and prepares a label file. The run stops before final ML training so labels can be reviewed.

```bash
docker compose run --rm pipeline python pipeline/run_pipeline.py \
  --event-id kinx_2024_05_07 \
  --event-name "KINX 2024-05-07 Oklahoma Supercell" \
  --radar-site KINX \
  --date-folder 2024-05-07 \
  --start-hhmm 0130 \
  --end-hhmm 0330 \
  --max-downloads 10 \
  --sweep 1 \
  --clean-case \
  --clean-radar-frames \
  --stop-before-labels
```

This creates a case-specific output folder and a manual label file.

Expected label file:

```text
outputs/cases/kinx_2024_05_07/manual/track_labels.csv
```

## Step 3: Review and label tracks

Open the generated track outputs, radar-frame images, and diagnostic figures. Identify which track appears to be the main vortex-like or tornado-associated circulation.

The label file should follow this structure:

```csv
track_id,tornado_associated,label_notes
TRK_006,1,visually validated main circulation
TRK_001,0,weaker or secondary circulation
TRK_002,0,false, weak, or non-primary track
```

Label meaning:

```text
1 = visually validated or tornado-associated circulation track
0 = false, weak, secondary, or non-primary track
```

The model uses these labels to learn which radar-derived tracks most closely resemble the manually validated circulation tracks. It should not be interpreted as an official tornado prediction model.

## Step 4: Finish the pipeline after labeling

After the label file is reviewed and saved, finish the pipeline without cleaning the case folder.

```bash
docker compose run --rm pipeline python pipeline/run_pipeline.py \
  --event-id kinx_2024_05_07 \
  --skip-download
```

Do not use `--clean-case` after labeling unless the label file has been backed up. Cleaning the case folder removes generated outputs, including manual labels.

## Step 5: Load the event into PostGIS

After the pipeline completes, load the event outputs into the PostGIS database.

```bash
docker compose run --rm pipeline python pipeline/load_postgis.py \
  --event-id kinx_2024_05_07 \
  --event-name "KINX 2024-05-07 Oklahoma Supercell" \
  --radar-site KINX \
  --date-folder 2024-05-07
```

If the local project still uses the older numbered script name, use:

```bash
docker compose run --rm pipeline python pipeline/11_load_postgis.py \
  --event-id kinx_2024_05_07 \
  --event-name "KINX 2024-05-07 Oklahoma Supercell" \
  --radar-site KINX \
  --date-folder 2024-05-07
```

## Step 6: Export API-preview files if needed

Some project versions include an API-preview export step. This exports database-backed GeoJSON and JSON files for validation.

```bash
docker compose run --rm pipeline python pipeline/export_api_preview.py \
  --event-id kinx_2024_05_07 \
  --output-dir outputs/api_preview/kinx_2024_05_07
```

If the local project still uses the older numbered script name, use:

```bash
docker compose run --rm pipeline python pipeline/12_query_postgis_preview.py \
  --event-id kinx_2024_05_07 \
  --output-dir outputs/api_preview/kinx_2024_05_07
```

Expected API-preview files may include:

```text
event_summary.json
tracks_with_scores.geojson
nowcasts_with_scores.geojson
best_track.geojson
best_track_nowcasts.geojson
high_priority_tracks.geojson
high_priority_nowcasts.geojson
detections.geojson
```

## Step 7: Start the backend

Start the FastAPI backend:

```bash
docker compose up -d backend
```

Or run it directly from the backend folder:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check the API:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

Useful event endpoints:

```text
GET /events
GET /events/kinx_2024_05_07/summary
GET /events/kinx_2024_05_07/times
GET /events/kinx_2024_05_07/tracks
GET /events/kinx_2024_05_07/detections
GET /events/kinx_2024_05_07/nowcasts
GET /events/kinx_2024_05_07/radar-frame?time=...
```

## Step 8: View the event in the frontend

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

If the frontend is locked to a single event, update the locked event value in the frontend configuration or `App.jsx` file:

```js
const LOCKED_EVENT_ID = "kinx_2024_05_07";
```

Restart the frontend after making the change.

For a public demo, the project should remain locked to the strongest validated case. For local testing, the locked event ID can be changed to inspect a different storm.

## Step 9: Check the output

A successful run should produce:

```text
Downloaded radar files
Candidate circulation CSVs
Diagnostic figures
Track CSVs
Nowcast CSVs
ML feature tables
ML score outputs
GeoJSON layers
Radar-frame PNGs
PostGIS records
Backend API responses
Frontend map visualization
```

The frontend should show:

```text
Radar velocity imagery
Cumulative detections through the selected scan time
Full circulation tracks
15- and 30-minute nowcast paths
Priority tracks based on ML score
Selected-feature inspection
```

## Adapting the process to another radar event

To run a different storm, change these values:

```text
--event-id
--event-name
--radar-site
--date-folder
--start-hhmm
--end-hhmm
--max-downloads
--sweep
```

Example template:

```bash
docker compose run --rm pipeline python pipeline/run_pipeline.py \
  --event-id radar_yyyy_mm_dd \
  --event-name "Readable Storm Event Name" \
  --radar-site RADAR \
  --date-folder YYYY-MM-DD \
  --start-hhmm HHMM \
  --end-hhmm HHMM \
  --max-downloads 10 \
  --sweep 1 \
  --clean-case \
  --clean-radar-frames \
  --stop-before-labels
```

Then label the tracks, finish the run, load PostGIS, start the backend, and view the event in the frontend.

## Event selection recommendations

E.V.E. works best on storm cases with:

```text
one dominant compact velocity couplet
a discrete or semi-discrete storm mode
a short focused time window
stable scan-to-scan motion
limited nearby competing circulations
clear velocity and reflectivity structure
```

It may perform less reliably on:

```text
messy outbreak environments
QLCS or line-embedded circulations
many nearby competing rotations
broad weak rotation
rapid cycling storms
cases with poor radar sampling
events too close to or too far from the radar
```

For best results, choose a short window around the mature circulation rather than a long event window containing multiple storm modes.

## Responsible-use note

E.V.E. is an experimental portfolio project for radar-derived circulation detection, tracking, scoring, and visualization. It is not an official forecast, warning, or public-safety system. Official severe-weather guidance should come from the National Weather Service and local emergency authorities.
