import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const LOCKED_EVENT_ID = "test_case_2";

const EMPTY_FC = { type: "FeatureCollection", features: [] };
const DEFAULT_CENTER = [36.8, -96.2];
const DEFAULT_ZOOM = 7;
const RADAR_CENTER_ZOOM = 8;

const RADAR_SITE_CENTERS = {
  KINX: [36.1751, -95.5641],
  KBMX: [33.1719, -86.7697],
  KILX: [40.1505, -89.3368],
  KLOT: [41.6044, -88.0845],
  KDMX: [41.7311, -93.7230],
  KTLX: [35.3331, -97.2778],
};

function radarCenterForEvent(summary, events, eventId) {
  const summarySite = summary?.radar_site || "";
  const eventSite = events.find((event) => event.event_id === eventId)?.radar_site || "";
  return RADAR_SITE_CENTERS[summarySite] || RADAR_SITE_CENTERS[eventSite] || null;
}

function cleanFeatureCollection(data) {
  if (!data || data.type !== "FeatureCollection" || !Array.isArray(data.features)) {
    return EMPTY_FC;
  }

  return {
    type: "FeatureCollection",
    features: data.features.filter((feature) => feature && feature.geometry),
  };
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${path}`);
  }

  return response.json();
}

function fmt(value, digits = 2, suffix = "") {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return `${value}${suffix}`;
  return `${parsed.toFixed(digits)}${suffix}`;
}

function pct(value) {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  return `${Math.round(parsed * 100)}%`;
}

function formatScanTime(value) {
  if (!value) return "No scan selected";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return `${date.toISOString().slice(11, 19)} UTC`;
}

function apiUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

function featureName(feature) {
  const p = feature?.properties || {};
  return p.ranked_track_name || p.track_id || p.detection_id || "Map feature";
}

function getMlScore(feature) {
  const parsed = Number(feature?.properties?.ml_score ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getQualityScore(feature) {
  const parsed = Number(feature?.properties?.track_quality_score ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function featureTimeValue(feature) {
  const p = feature?.properties || {};
  return p.scan_time || p.scan_time_utc || p.source_scan_time || p.source_scan_time_utc || p.latest_scan_time || "";
}

function isFeatureAtOrBefore(feature, selectedTime) {
  if (!selectedTime) return true;

  const featureTime = featureTimeValue(feature);
  if (!featureTime) return true;

  const featureMs = new Date(featureTime).getTime();
  const selectedMs = new Date(selectedTime).getTime();

  if (!Number.isFinite(featureMs) || !Number.isFinite(selectedMs)) return true;

  return featureMs <= selectedMs;
}

function popupHtml(feature) {
  const p = feature.properties || {};
  return `
    <div class="eve-popup">
      <strong>${featureName(feature)}</strong>
      <span>ML score: ${fmt(p.ml_score, 3)}</span>
      <span>Quality: ${fmt(p.track_quality_score, 1)}</span>
      <span>Delta-V: ${fmt(p.mean_delta_v_kt ?? p.delta_v_kt, 1, " kt")}</span>
      <span>Scan: ${featureTimeValue(feature) || "—"}</span>
    </div>
  `;
}

function trackStyle() {
  return {
    color: "#facc15",
    weight: 3.4,
    opacity: 0.92,
  };
}

function priorityTrackStyle() {
  return {
    color: "#ff1744",
    weight: 6,
    opacity: 0.98,
  };
}

function bestTrackStyle() {
  return {
    color: "#ff1744",
    weight: 8,
    opacity: 1,
  };
}

function nowcastStyle() {
  return {
    color: "#ff4db8",
    weight: 4,
    opacity: 0.94,
    dashArray: "8 8",
  };
}

function bestNowcastStyle() {
  return {
    color: "#ff4db8",
    weight: 4,
    opacity: 0.94,
    dashArray: "8 8",
  };
}

function detectionPoint(feature, latlng) {
  const strong = getMlScore(feature) >= 0.7;
  const confidence = Number(feature?.properties?.confidence_score ?? 0);

  return L.circleMarker(latlng, {
    radius: Math.min(8, Math.max(4, 4 + confidence / 28)),
    color: strong ? "#ff1744" : "#facc15",
    fillColor: strong ? "#ff6b7f" : "#fde68a",
    fillOpacity: 0.84,
    opacity: 0.98,
    weight: 1.5,
  });
}

function radarBounds(frame) {
  const b = frame?.bounds;
  if (!b) return null;

  const { south, west, north, east } = b;
  if ([south, west, north, east].some((value) => value === null || value === undefined)) {
    return null;
  }

  return L.latLngBounds([[south, west], [north, east]]);
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function LayerToggle({ label, checked, onChange, disabled = false }) {
  return (
    <label className={`layer-toggle ${checked ? "active" : ""} ${disabled ? "disabled" : ""}`}>
      <input type="checkbox" checked={checked} onChange={onChange} disabled={disabled} />
      <span className="layer-switch" />
      <span>{label}</span>
    </label>
  );
}

function Legend() {
  return (
    <div className="legend-card">
      <div className="legend-title">Map Legend</div>
      <div className="legend-row">
        <span className="legend-radar" />
        <span>Radar Scan</span>
      </div>
      <div className="legend-row">
        <span className="legend-line tracks" />
        <span>All Tracks</span>
      </div>
      <div className="legend-row">
        <span className="legend-line priority" />
        <span>Priority Tracks</span>
      </div>
      <div className="legend-row">
        <span className="legend-line nowcast" />
        <span>Nowcasts</span>
      </div>
      <div className="legend-row">
        <span className="legend-point" />
        <span>Detections</span>
      </div>
    </div>
  );
}

export default function App() {
  const mapElementRef = useRef(null);
  const mapRef = useRef(null);
  const drawnLayersRef = useRef(null);
  const radarLayerRef = useRef(null);

  const [events, setEvents] = useState([]);
  const [eventId] = useState(LOCKED_EVENT_ID);

  const [summary, setSummary] = useState(null);
  const [times, setTimes] = useState([]);
  const [timeIndex, setTimeIndex] = useState(0);
  const selectedTime = times[timeIndex]?.scan_time || "";

  const [tracks, setTracks] = useState(EMPTY_FC);
  const [nowcasts, setNowcasts] = useState(EMPTY_FC);
  const [bestTrack, setBestTrack] = useState(EMPTY_FC);
  const [bestNowcasts, setBestNowcasts] = useState(EMPTY_FC);
  const [allDetections, setAllDetections] = useState(EMPTY_FC);
  const [radarFrame, setRadarFrame] = useState(null);

  const [selectedFeature, setSelectedFeature] = useState(null);
  const [loadingEvents, setLoadingEvents] = useState(true);
  const [loadingLayers, setLoadingLayers] = useState(false);
  const [loadingScan, setLoadingScan] = useState(false);
  const [error, setError] = useState("");
  const [radarError, setRadarError] = useState("");

  const [layers, setLayers] = useState({
    radar: true,
    allTracks: true,
    priority: true,
    nowcasts: true,
    detections: true,
  });

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const priorityTracks = useMemo(() => {
    return {
      type: "FeatureCollection",
      features: tracks.features.filter((feature) => {
        return getMlScore(feature) >= 0.9;
      }),
    };
  }, [tracks]);

  const cumulativeDetections = useMemo(() => {
    return {
      type: "FeatureCollection",
      features: allDetections.features.filter((feature) => isFeatureAtOrBefore(feature, selectedTime)),
    };
  }, [allDetections, selectedTime]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") setSidebarOpen(false);
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    if (!mapElementRef.current || mapRef.current) return;

    const map = L.map(mapElementRef.current, {
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: false,
      preferCanvas: true,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      subdomains: "abcd",
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    }).addTo(map);

    L.control.zoom({ position: "topright" }).addTo(map);
    L.control.scale({ position: "bottomleft" }).addTo(map);

    mapRef.current = map;

    const resizeMap = () => map.invalidateSize();
    window.setTimeout(resizeMap, 0);
    window.setTimeout(resizeMap, 250);
    window.addEventListener("resize", resizeMap);

    return () => {
      window.removeEventListener("resize", resizeMap);
      map.remove();
      mapRef.current = null;
      radarLayerRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadEvents() {
      try {
        setLoadingEvents(true);
        setError("");

        const rows = await apiGet("/events");
        if (cancelled) return;

        const eventRows = Array.isArray(rows) ? rows : [];
        setEvents(eventRows);

        if (!eventRows.some((event) => event.event_id === LOCKED_EVENT_ID)) {
          setError(`FastAPI did not return ${LOCKED_EVENT_ID}. Load that case into PostGIS first.`);
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Could not connect to FastAPI at ${API_BASE_URL}. ${err.message}`);
        }
      } finally {
        if (!cancelled) setLoadingEvents(false);
      }
    }

    loadEvents();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!eventId) return;

    let cancelled = false;

    async function loadEventBaseData() {
      try {
        setLoadingLayers(true);
        setError("");
        setRadarError("");
        setSelectedFeature(null);
        setTimes([]);
        setTimeIndex(0);

        const [
          summaryResult,
          timesResult,
          allDetectionsResult,
          tracksResult,
          nowcastsResult,
          bestTrackResult,
          bestNowcastsResult,
        ] = await Promise.all([
          apiGet(`/events/${eventId}/summary`),
          apiGet(`/events/${eventId}/times`),
          apiGet(`/events/${eventId}/detections?limit=10000`),
          apiGet(`/events/${eventId}/tracks`),
          apiGet(`/events/${eventId}/nowcasts`),
          apiGet(`/events/${eventId}/best-track`),
          apiGet(`/events/${eventId}/best-track/nowcasts`),
        ]);

        if (cancelled) return;

        const orderedTimes = Array.isArray(timesResult) ? timesResult : [];

        setSummary(summaryResult);
        setTimes(orderedTimes);
        setTimeIndex(orderedTimes.length ? 0 : 0);
        setAllDetections(cleanFeatureCollection(allDetectionsResult));
        setTracks(cleanFeatureCollection(tracksResult));
        setNowcasts(cleanFeatureCollection(nowcastsResult));
        setBestTrack(cleanFeatureCollection(bestTrackResult));
        setBestNowcasts(cleanFeatureCollection(bestNowcastsResult));
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setSummary(null);
          setTimes([]);
          setTracks(EMPTY_FC);
          setNowcasts(EMPTY_FC);
          setBestTrack(EMPTY_FC);
          setBestNowcasts(EMPTY_FC);
          setAllDetections(EMPTY_FC);
          setRadarFrame(null);
        }
      } finally {
        if (!cancelled) setLoadingLayers(false);
      }
    }

    loadEventBaseData();

    return () => {
      cancelled = true;
    };
  }, [eventId]);

  useEffect(() => {
    if (!eventId) return;

    let cancelled = false;

    async function loadSelectedRadarFrame() {
      try {
        setLoadingScan(true);
        setRadarError("");

        const radarFrameResult = selectedTime
          ? await apiGet(`/events/${eventId}/radar-frame?time=${encodeURIComponent(selectedTime)}`).catch((err) => {
              if (!cancelled) setRadarError(err.message);
              return null;
            })
          : null;

        if (cancelled) return;
        setRadarFrame(radarFrameResult);
      } catch (err) {
        if (!cancelled) {
          setRadarError(err.message);
          setRadarFrame(null);
        }
      } finally {
        if (!cancelled) setLoadingScan(false);
      }
    }

    loadSelectedRadarFrame();

    return () => {
      cancelled = true;
    };
  }, [eventId, selectedTime]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (radarLayerRef.current) {
      radarLayerRef.current.remove();
      radarLayerRef.current = null;
    }

    if (!layers.radar || !radarFrame?.image_url) return;

    const bounds = radarBounds(radarFrame);
    if (!bounds) return;

    radarLayerRef.current = L.imageOverlay(apiUrl(radarFrame.image_url), bounds, {
      opacity: 0.66,
      interactive: false,
    }).addTo(map);
  }, [layers.radar, radarFrame]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !eventId) return;

    const center = radarCenterForEvent(summary, events, eventId);
    if (!center) return;

    window.setTimeout(() => {
      map.invalidateSize();
      map.setView(center, RADAR_CENTER_ZOOM, { animate: true });
    }, 90);
  }, [eventId, summary?.radar_site, events]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (drawnLayersRef.current) {
      drawnLayersRef.current.remove();
      drawnLayersRef.current = null;
    }

    const group = L.featureGroup();

    function addCollection(collection, options) {
      if (!collection.features.length) return;

      const geoJson = L.geoJSON(collection, {
        ...options,
        onEachFeature: (feature, layer) => {
          layer.bindPopup(popupHtml(feature));
          layer.on("click", () => setSelectedFeature(feature));
        },
      });

      geoJson.addTo(group);
    }

    if (layers.allTracks) {
      addCollection(tracks, { style: trackStyle });
    }

    if (layers.priority) {
      addCollection(priorityTracks, { style: priorityTrackStyle });
      addCollection(bestTrack, { style: bestTrackStyle });
    }

    if (layers.nowcasts) {
      addCollection(nowcasts, { style: nowcastStyle });
      addCollection(bestNowcasts, { style: bestNowcastStyle });
    }

    if (layers.detections) {
      addCollection(cumulativeDetections, { pointToLayer: detectionPoint });
    }

    group.addTo(map);
    drawnLayersRef.current = group;

    window.setTimeout(() => {
      map.invalidateSize();
    }, 80);

    return () => {
      group.remove();
    };
  }, [
    eventId,
    tracks,
    priorityTracks,
    nowcasts,
    bestTrack,
    bestNowcasts,
    cumulativeDetections,
    layers,
    radarFrame,
  ]);

  const best = summary?.best_track || null;
  const selectedProps = selectedFeature?.properties || null;
  const radarAvailable = Boolean(radarFrame?.image_url);

  function toggleLayer(key) {
    setLayers((current) => ({
      ...current,
      [key]: !current[key],
    }));
  }

  function stepTime(delta) {
    setTimeIndex((current) => {
      if (!times.length) return 0;
      return Math.min(times.length - 1, Math.max(0, current + delta));
    });
  }

  return (
    <div className={`eve-app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <button
        type="button"
        className="mobile-sidebar-toggle"
        aria-label="Open E-V-E controls"
        aria-expanded={sidebarOpen}
        onClick={() => setSidebarOpen(true)}
      >
        Controls
      </button>

      <button
        type="button"
        className="sidebar-backdrop"
        aria-label="Close E-V-E controls"
        onClick={() => setSidebarOpen(false)}
      />

      <aside className="sidebar">
        <div className="sidebar-mobile-header">
          <span>Controls</span>
          <button type="button" onClick={() => setSidebarOpen(false)}>
            Close
          </button>
        </div>

        <div className="brand-block">
          <h1>E-V-E</h1>
          <div className="kicker">The Experimental Vortex Evaluator</div>
          <h3></h3>
          <p>
            E-V-E uses scan-by-scan Doppler velocity data to identify storm circulations and build intelligent vortex projections
            using meteorological, GIS and ML principles.
          </p>
        </div>

        <section className="sidebar-section">
          <div className="field-label">Storm example</div>
          <div className="locked-event-card">
            <strong>Central Alabama Storm Outbreak</strong>
            <span>{summary?.event_name || events.find((event) => event.event_id === LOCKED_EVENT_ID)?.event_name || "test_case_2"}</span>
          </div>

          <div className="connection-row">
            <span className={`connection-dot ${error ? "bad" : "good"}`} />
            <span>
              {loadingEvents || loadingLayers || loadingScan
                ? "Loading data"
                : error
                  ? "API issue"
                  : "API connected"}
            </span>
          </div>

          {error && <div className="error-card">{error}</div>}
          {layers.radar && !radarAvailable && radarError && (
            <div className="error-card">
              Radar overlay unavailable for this scan. Run radar frame rendering and reload PostGIS.
            </div>
          )}
        </section>

        <section className="sidebar-section time-section">
          <div className="section-header">
            <h2>Radar Time</h2>
            <span>{times.length ? `${timeIndex + 1}/${times.length}` : "0/0"}</span>
          </div>

          <div className="time-card">
            <strong>{formatScanTime(selectedTime)}</strong>
            <span>{times[timeIndex]?.radar_file || "No radar scan loaded"}</span>
          </div>

          <input
            className="time-slider"
            type="range"
            min="0"
            max={Math.max(0, times.length - 1)}
            value={Math.min(timeIndex, Math.max(0, times.length - 1))}
            disabled={!times.length}
            onChange={(event) => setTimeIndex(Number(event.target.value))}
          />

          <div className="time-buttons">
            <button onClick={() => stepTime(-1)} disabled={!times.length || timeIndex <= 0}>
              Previous
            </button>
            <button onClick={() => stepTime(1)} disabled={!times.length || timeIndex >= times.length - 1}>
              Next
            </button>
          </div>

          <p className="time-note">
            Tracks and nowcasts show the full selected test case. Detections are cumulative through the selected scan.
          </p>
        </section>

        <section className="sidebar-section">
          <div className="section-header">
            <h2>Event Summary</h2>
            <span>{summary?.radar_site || "Radar"}</span>
          </div>

          <div className="metric-grid">
            <MetricCard label="Radar Used" value={summary?.radar_site || "—"} />
            <MetricCard label="Date" value={summary?.date_folder || "—"} />
            <MetricCard label="Tracks" value={summary?.counts?.signature_tracks ?? "—"} />
            <MetricCard label="Radar Frames" value={summary?.counts?.radar_frames ?? "—"} />
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-header">
            <h2>Layers</h2>
            <span>Map</span>
          </div>

          <div className="layer-grid compact">
            <LayerToggle
              label="Radar Scan"
              checked={layers.radar}
              disabled={!radarAvailable}
              onChange={() => toggleLayer("radar")}
            />
            <LayerToggle
              label="All Tracks"
              checked={layers.allTracks}
              onChange={() => toggleLayer("allTracks")}
            />
            <LayerToggle
              label="Priority Tracks"
              checked={layers.priority}
              onChange={() => toggleLayer("priority")}
            />
            <LayerToggle
              label="Nowcasts"
              checked={layers.nowcasts}
              onChange={() => toggleLayer("nowcasts")}
            />
            <LayerToggle
              label="Detections"
              checked={layers.detections}
              onChange={() => toggleLayer("detections")}
            />
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-header">
            <h2>Best ML-Ranked Track</h2>
            <span>Priority</span>
          </div>

          <div className="best-card">
            <div className="best-card-top">
              <div>
                <div className="small-kicker">Primary circulation</div>
                <strong>{best?.ranked_track_name || best?.track_id || "—"}</strong>
              </div>
              <div className="score-pill">{pct(best?.ml_score)}</div>
            </div>

            <div className="score-bar">
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, Number(best?.ml_score || 0) * 100))}%`,
                }}
              />
            </div>

            <DetailRow label="ML score" value={fmt(best?.ml_score, 3)} />
            <DetailRow label="ML label" value={best?.ml_label || "—"} />
            <DetailRow label="Quality score" value={fmt(best?.track_quality_score, 1)} />
            <DetailRow label="Quality label" value={best?.track_quality_label || "—"} />
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-header">
            <h2>Selected Feature</h2>
            <span>Inspect</span>
          </div>

          {selectedProps ? (
            <div className="details-card">
              <DetailRow label="Name" value={featureName(selectedFeature)} />
              <DetailRow label="Track ID" value={selectedProps.track_id || "—"} />
              <DetailRow label="ML score" value={fmt(selectedProps.ml_score, 3)} />
              <DetailRow label="ML rank" value={selectedProps.ml_rank ?? "—"} />
              <DetailRow label="Quality" value={fmt(selectedProps.track_quality_score, 1)} />
              <DetailRow
                label="Duration"
                value={
                  selectedProps.duration_min !== undefined
                    ? fmt(selectedProps.duration_min, 1, " min")
                    : "—"
                }
              />
              <DetailRow
                label="Speed"
                value={
                  selectedProps.speed_kt !== undefined
                    ? fmt(selectedProps.speed_kt, 1, " kt")
                    : "—"
                }
              />
              <DetailRow
                label="Bearing"
                value={
                  selectedProps.bearing_deg !== undefined
                    ? fmt(selectedProps.bearing_deg, 0, "°")
                    : "—"
                }
              />
              <DetailRow
                label="Delta-V"
                value={
                  selectedProps.mean_delta_v_kt !== undefined
                    ? fmt(selectedProps.mean_delta_v_kt, 1, " kt")
                    : selectedProps.delta_v_kt !== undefined
                      ? fmt(selectedProps.delta_v_kt, 1, " kt")
                      : "—"
                }
              />
            </div>
          ) : (
            <div className="empty-card">
              Click a track, nowcast path, or detection point to inspect it.
            </div>
          )}
        </section>
      </aside>

      <main className="map-panel">
        <header className="map-header">
          <div>
            <div className="map-kicker">FastAPI → PostGIS Live Preview</div>
            <h2>{summary?.event_name || eventId || "E-V-E Event"}</h2>
          </div>
          <div className="api-pill">{selectedTime ? formatScanTime(selectedTime) : API_BASE_URL}</div>
        </header>

        <div className="map-frame">
          <div ref={mapElementRef} className="leaflet-map" />
          <Legend />
        </div>
      </main>
    </div>
  );
}
