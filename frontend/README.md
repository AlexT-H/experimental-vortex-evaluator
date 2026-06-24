# E-V-E Frontend

This frontend restores the original `src` look and layout while keeping the newer radar/time controls.

Included behavior:

- original dark E-V-E dashboard styling
- radar velocity layer toggle
- radar scan time slider
- previous/next scan buttons
- no automatic map pan/zoom when the selected scan time changes
- cumulative detection display through the selected scan time

The app expects the backend at:

```text
http://localhost:8000
```

Override with:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## Run

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```
