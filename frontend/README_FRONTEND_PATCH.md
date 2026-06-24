# E-V-E Frontend Patch — Persistent Tracks, Cumulative Detections, Radar-Centered Map

## Files to replace/add

Copy these into your frontend project:

```text
src/App.jsx
src/App.css
src/index.css
src/main.jsx
public/favicon.svg
index.html
```

`App.css`, `index.css`, and `main.jsx` are included unchanged so the patch can be dropped into your frontend folder cleanly.

## What changed

1. Tracks and nowcasts are loaded for the full selected test case and stay visible for the entire event.
2. Detections remain cumulative through the selected radar scan.
3. Radar velocity still changes with the time slider.
4. When a test case is selected, the map centers on that event's Doppler radar site.
5. The uploaded E-V-E SVG is used as the browser favicon.

## Radar centers included

```text
KINX, KBMX, KILX, KLOT, KDMX, KTLX
```

## Install

From your project root, copy the files into the frontend folder. Example:

```bash
cp src/App.jsx frontend/src/App.jsx
cp src/App.css frontend/src/App.css
cp src/index.css frontend/src/index.css
cp src/main.jsx frontend/src/main.jsx
cp index.html frontend/index.html
mkdir -p frontend/public
cp public/favicon.svg frontend/public/favicon.svg
```

Then run:

```bash
cd frontend
npm install
npm run dev
```
