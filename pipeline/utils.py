from pathlib import Path
from datetime import datetime, timezone
import re


def parse_nexrad_scan_metadata(radar_path, event_id, sweep=None):
    """
    Parse NEXRAD-style filenames such as:
    KINX20240507_020733_V06

    Returns fields that should be attached to detections, tracks,
    nowcasts, and event_times.
    """
    name = Path(str(radar_path)).name

    match = re.search(r"([A-Z]{4})(\\d{8})[_-](\\d{6})", name)
    if not match:
        raise ValueError(f"Could not parse radar scan time from filename: {name}")

    radar_site = match.group(1)
    date_part = match.group(2)
    time_part = match.group(3)

    scan_dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
    scan_dt = scan_dt.replace(tzinfo=timezone.utc)

    return {
        "event_id": event_id,
        "scan_time": scan_dt.isoformat().replace("+00:00", "Z"),
        "radar_site": radar_site,
        "radar_file": name,
        "sweep": sweep,
    }
