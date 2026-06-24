# E.V.E. Methodology

This document explains the technical methodology behind **E.V.E. - The Experimental Vortex Evaluator**. It covers data ingestion, radar processing, storm-relative velocity conversion, TVS-style candidate detection, circulation tracking, nowcasting, ML scoring, PostGIS storage, API serving, frontend visualization, and limitations.

---

## Methodology summary

E.V.E. converts historical NEXRAD Level II radar files into derived geospatial storm objects:

```text
radar scans
    ↓
storm-relative velocity fields
    ↓
candidate vortex / TVS-style signatures
    ↓
cross-scan circulation tracks
    ↓
15/30-minute nowcast paths
    ↓
ML-prioritized track scores
    ↓
PostGIS geometries and API-served map layers
```

The system is designed to demonstrate radar-data engineering, geospatial analysis, and web GIS architecture. It is not intended to provide operational tornado warnings.

---

## 1. Data source and radar-event setup

### Primary radar data

E.V.E. uses historical NEXRAD Level II radar files. These files contain Doppler radar fields such as:

```text
reflectivity
velocity
spectrum_width
```

The two most important fields for the current system are:

| Field | Use |
|---|---|
| `reflectivity` | Identifies storm structure and precipitation intensity. |
| `velocity` | Shows radial wind motion toward or away from the radar and supports circulation/couplet detection. |

### Final retained case

The current public-facing portfolio demo uses:

```text
event_id: test_case_2
radar_site: KBMX
date_folder: 2011-04-27
event: Central Alabama Storm Outbreak
```

This case was retained because it produced the strongest final frontend visualization.

### Development case

The original development case used:

```text
radar_site: KINX
date_folder: 2024-05-07
event: May 6, 2024 northeast Oklahoma tornadic supercell
sweep: 1
```

KINX was useful for building and debugging the detector, but it is not the active final frontend demo case.

---

## 2. Radar preprocessing

Each radar file is opened with Py-ART. For each scan, the pipeline extracts velocity and reflectivity data from the selected sweep.

The preprocessing stage performs these tasks:

1. Open each NEXRAD Level II file.
2. Read velocity data.
3. Read reflectivity data when available.
4. Extract the selected sweep.
5. Convert radar scan time into a consistent UTC timestamp.
6. Convert radar gate positions into usable spatial coordinates.
7. Prepare arrays for velocity analysis, candidate detection, diagnostic rendering, and later map export.

### Sweep selection

During KINX development, Sweep 1 was selected because it consistently contained usable low-level velocity data. Some nearby sweeps were empty or unusable for the files that were checked.

In later cases, the selected sweep should be verified again rather than assumed.

---

## 3. Storm-relative velocity methodology

Base radial velocity includes both storm movement and internal storm-scale rotation. E.V.E. attempts to estimate the parent storm motion and subtract the storm-motion component from the base radial velocity.

Conceptually:

```text
storm-relative radial velocity = base radial velocity - storm radial motion component
```

### Storm-motion estimate

The current method estimates storm motion from reflectivity echo centroid displacement:

1. Threshold reflectivity to identify the main storm echo.
2. Compute a reflectivity-weighted echo centroid.
3. Compare the centroid position to the previous scan.
4. Convert scan-to-scan displacement into storm-motion speed and direction.
5. Smooth the estimate to reduce scan-to-scan noise.
6. Project the storm-motion vector onto the radar radial.
7. Subtract that projected component from base radial velocity.

The first scan cannot calculate motion from a previous scan, so it uses a fallback/base method until a valid displacement can be estimated.

### Why storm-relative velocity matters

Storm-relative velocity makes the candidate detector focus more on localized rotation than on broad storm translation. This is especially important when trying to identify compact inbound/outbound couplets associated with storm-scale circulation.

---

## 4. TVS-style candidate detection

### Detection goal

The detector identifies candidate vortex or TVS-style signatures from the storm-relative velocity field.

The goal is not to confirm tornadoes. The goal is to identify plausible radar-indicated circulation features that can be tracked and scored.

### Detector evolution

The detector evolved through several stages:

1. **Gate-level velocity pairing**
   - Detected nearby inbound/outbound gates.
   - Produced too many noisy candidates.

2. **Region-based detection**
   - Grouped connected inbound/outbound regions.
   - Reduced noise but could miss broader or messy rotation.

3. **Geometry filtering**
   - Added couplet diameter, tightness, size balance, spread, range, and rotation-score metrics.

4. **Hybrid detection**
   - Combined region-pair detection with local-shear scanning.

5. **Storm-context scoring**
   - Used reflectivity to emphasize candidates embedded in meaningful storm structure.

6. **Banded storm-relative TVS-style detection**
   - Added banded lobe logic, neutral-center logic, lobe opposition, radial split checks, and range-dependent Delta-V requirements.

### Final detection philosophy

The current detector looks for a compact, structured, storm-relative velocity dipole rather than a simple line of shear.

A stronger candidate generally has:

```text
strong positive velocity lobe
strong negative velocity lobe
weaker same-sign surrounding bands
neutral/gray transition near the center
opposing or semi-circular lobe geometry
reflectivity support
sufficient Delta-V
reasonable lobe separation
radial split across the candidate center
```

### Reflectivity context

Candidate patches must have enough reflectivity support. This prevents the detector from searching arbitrary clear-air pixels or ranking velocity artifacts far away from storm echo.

Reflectivity is used to evaluate:

- whether there is enough storm echo in the candidate patch,
- whether the patch has sufficient maximum reflectivity,
- whether the candidate is near meaningful storm structure.

### Range and Delta-V logic

The detector applies range-aware filtering. The exact threshold values can be tuned, but the methodology is:

- reject candidates outside the usable radar range,
- require stronger Delta-V close to the radar where clutter/static can dominate,
- allow a slightly lower threshold farther out while still rejecting weak signatures,
- reject candidates with unrealistic lobe separation or poor lobe geometry.

### Banded lobe logic

The detector checks whether each local patch contains:

- strong positive gates,
- strong negative gates,
- mid-level positive gates,
- mid-level negative gates,
- neutral gates near the center.

This is intended to capture a more vortex-like visual pattern:

```text
strong lobe → weaker same-sign band → neutral center → opposite-sign band → opposite strong lobe
```

### Radial split requirement

The detector requires the positive and negative lobes to sit on opposite sides of the radar radial through the candidate center. This reduces false detections from non-rotational shear boundaries.

### Candidate scoring

The confidence score combines:

```text
Delta-V strength
lobe opposition
banded velocity structure
neutral-center structure
radial side balance
reflectivity context
```

Low-scoring or duplicate candidates are filtered out before outputs are written.

### Candidate outputs

Per-scan candidate outputs include fields such as:

```text
radar_file
scan_time_utc
sweep
latitude
longitude
center_x_km
center_y_km
confidence_score
delta_v_kt
required_delta_v_kt
radar_range_miles
lobe_separation_km
opposition_score
banded_score
neutral_center_score
reflectivity_dbz
storm_motion_u_kt
storm_motion_v_kt
storm_motion_speed_kt
```

Outputs are saved as CSVs and diagnostic PNGs for inspection.

---

## 5. Circulation tracking methodology

### Tracking goal

The tracker links candidate detections across radar scans into coherent circulation paths.

This step answers:

- Which detections likely represent the same circulation over time?
- How long did a circulation persist?
- How smoothly did it move?
- Which tracks look most coherent and vortex-like?

### Matching logic

A candidate detection can be linked to an active track when it satisfies scan-time, distance, and movement constraints.

The tracker considers:

```text
scan time gap
spatial distance from predicted track position
implied movement speed
track continuity
competing candidate-track matches
```

The tracker uses recent track motion to estimate the next expected position. This prevents the system from treating every scan as independent.

### Track quality score

Each track receives:

```text
track_quality_score
track_quality_rank
ranked_track_name
track_quality_label
```

The score rewards:

- persistence,
- longer duration,
- higher confidence,
- higher Delta-V,
- plausible speed,
- smooth movement,
- fewer jumps,
- fewer erratic direction changes.

The score penalizes:

- single-scan detections,
- very short tracks,
- large jumps,
- unrealistic speed,
- erratic heading changes.

### Why all tracks are preserved

E.V.E. keeps all tracks for transparency, while also producing high-quality and best-track layers for cleaner visualization. This allows the web app to show both raw algorithmic output and prioritized interpretation.

---

## 6. Nowcasting methodology

### Nowcast goal

For each projectable track, E.V.E. estimates recent movement and projects a short path forward.

The current system produces only:

```text
15-minute projections
30-minute projections
```


### Nowcast fields

Each nowcast path stores:

```text
track_id
track_quality_rank
ranked_track_name
track_quality_score
track_quality_label
issued_time_utc
projection_min
valid_time_utc
start_latitude
start_longitude
end_latitude
end_longitude
speed_kt
bearing_deg
```

### Interpretation

A nowcast path is an extrapolated movement path from the detected circulation track. It should not be interpreted as a tornado path forecast or a public-safety warning.

---

## 7. ML scoring methodology

### ML purpose

The ML model scores already-detected and already-tracked circulation signatures. It does not detect tornadoes directly from raw radar imagery.  The model produces an experimental tornado-associated circulation score.


### Feature level

The model operates at the track level. Each row represents one tracked circulation.

Feature groups include:

| Feature group | Examples |
|---|---|
| Persistence and structure | detection count, duration, path length |
| Motion behavior | speed, bearing, step distance, turn angle |
| Detection strength | confidence score, Delta-V |
| Track quality metrics | quality score, rank, quality sub-scores |
| Candidate aggregation | radar range, lobe separation, opposition, banding, reflectivity |
| Derived comparisons | Delta-V growth, confidence growth, path efficiency |

### Labels

The first model uses manual/weak labels:

```text
1 = visually validated or tornado-associated circulation
0 = false, secondary, weak, or non-primary circulation
```

Future improvements could add official validation by joining tracks to tornado reports, warning polygons, or time/distance relationships.

### Model type

The current baseline model uses:

```text
RandomForestClassifier
```

It outputs:

```text
ml_score
ml_rank
ml_label
```

The ML score controls emphasis and filtering in the web app. It does not replace the detection/tracking pipeline.

**Random Forest Classification** was used for the machine-learning portion of E.V.E. because the project scores structured track-level features rather than raw radar images. Each circulation track is represented by measurable attributes such as duration, detection count, Delta-V strength, confidence score, movement behavior, and track quality. A Random Forest model is well suited for this type of small tabular dataset because it can capture nonlinear relationships between these features while still remaining relatively interpretable through feature importance.

---

## 8. PostGIS methodology

### Purpose

PostGIS stores radar-derived objects as queryable spatial records.

The database stores:

```text
events
radar scans
circulation detections
signature tracks
nowcast paths
ML predictions
radar frame metadata
```

### Geometry types

```text
circulation detections -> Point
signature tracks       -> LineString
nowcast paths          -> LineString
```

### Why this matters

PostGIS turns E.V.E. from a file-output pipeline into a backend-ready GIS system. It allows the frontend to request filtered geospatial layers through an API instead of reading raw pipeline files directly.

---

## 9. API methodology

The FastAPI backend reads from PostGIS and radar-frame directories. It serves:

- event metadata,
- radar scan times,
- track GeoJSON,
- detection GeoJSON,
- nowcast GeoJSON,
- best-track layers,
- radar-frame metadata,
- radar-frame PNG files.

Important endpoint groups:

```text
/events
/events/{event_id}/summary
/events/{event_id}/times
/events/{event_id}/tracks
/events/{event_id}/detections
/events/{event_id}/nowcasts
/events/{event_id}/radar-frame?time=...
/radar_frames/{event_id}/{filename}
```

The frontend depends on these endpoints returning consistent event IDs, geometries, timestamps, and radar-frame bounds.

---

## 10. Frontend visualization methodology

### Map architecture

The frontend uses direct Leaflet rendering rather than `react-leaflet`. This was chosen because direct Leaflet resolved tile-rendering instability encountered in earlier versions.

The frontend renders:

- static basemap tiles,
- radar-frame image overlays,
- GeoJSON detection points,
- GeoJSON track lines,
- GeoJSON nowcast lines,
- selected-feature popups and inspection panels.

### Time slider logic

The scan-time slider controls:

```text
radar velocity frame
cumulative detection visibility
```

It does not hide completed tracks or nowcast paths. Tracks and nowcasts remain visible for the full event so the portfolio viewer can understand the complete circulation structure.

### Priority layer

Priority tracks require:

```text
ml_score >= 0.90
```

This strict threshold keeps the demo focused on high-confidence ML-prioritized tracks.

### Mobile behavior

The mobile version uses a slide-out sidebar with:

- Controls button,
- backdrop,
- close button,
- Escape-key support,
- full map viewport when closed.

---

## 11. Case-selection methodology

During final cleanup, several case options were tested or considered. The project was narrowed to one polished case because that made the public demo more defensible.

### Final retained case

```text
test_case_2 — KBMX 2011-04-27 Central Alabama Storm Outbreak
```

Retained because:

- detections displayed properly,
- full tracks and nowcasts displayed clearly,
- ML-prioritized styling worked well,
- it produced the strongest portfolio demo.

### Removed from final active demo

```text
test_case_1 / KINX 2024-05-07
```

KINX was valuable for development, but it was removed from the active deployed frontend to keep the final demo focused.

### Other tested/considered cases

Some possible third-case events were considered but not retained because they were too messy for the current detector/tracker or were not the preferred visual fit. The main lesson was that the current system performs best on curated, compact, discrete cases.

---

## 12. Validation and interpretation

E.V.E. uses visual and algorithmic validation rather than official operational verification.

The current validation approach includes:

- reviewing diagnostic radar PNGs,
- checking whether candidate markers appear near plausible couplets,
- reviewing track smoothness and persistence,
- confirming whether the best/priority tracks align with the visible main circulation,
- checking whether detections, tracks, nowcasts, and radar overlays display correctly in the frontend.

This is enough for a portfolio prototype, but it is not enough for operational meteorological validation.

Future validation could include:

- time/distance joins against tornado reports,
- comparison against NWS warning polygons,
- storm survey path overlays,
- larger multi-event testing,
- systematic false-positive and false-negative review,
- independent holdout events for ML evaluation.

---

## 13. Known limitations

E.V.E. currently performs best when the radar case has:

```text
one dominant compact velocity couplet
a relatively discrete storm mode
stable scan-to-scan motion
a short event window
limited nearby competing circulations
moderate radar range
clear scan-to-scan continuity
```

It performs worse when the case has:

```text
multiple nearby circulations
QLCS-like or messy storm mode
broad rotation instead of compact TVS-style rotation
rapid cycling
large wedge/EF5-scale circulation
competing candidates in the same scan
```

Specific limitations:

- false vortex tracks can occur,
- separate circulations can be merged into one path,
- one broad circulation can be double-counted,
- candidate quality depends strongly on event selection,
- the model uses limited/manual labels,
- the ML metrics should not be overinterpreted,
- the system is not operationally validated,
- the frontend is a showcase of preprocessed/API-served layers, not the full processing engine.

Recommended limitation statement:

> E.V.E. is a prototype and performs best on curated radar cases with compact, dominant velocity couplets. It is not an operational tornado-warning system. Complex storm modes, broad circulations, and multiple nearby vortex signatures can produce false tracks, merged tracks, or duplicate circulation paths.

---

## 14. Future methodology improvements

Future versions could improve E.V.E. by adding:

- official tornado-report validation,
- NWS warning polygon overlays,
- storm survey track overlays,
- better storm-motion estimation,
- velocity dealiasing checks,
- adaptive sweep selection,
- stronger clutter filtering,
- better multi-circulation separation,
- scan-by-scan animation,
- larger multi-event ML training,
- storm reflectivity radar calculations,
- a limitations/responsible-use document,
- hosted static demo exports for lightweight deployment.
