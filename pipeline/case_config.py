from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventCase:
    event_id: str
    radar_site: str
    date_folder: str
    start_hhmm: str = ""
    end_hhmm: str = ""
    sweep: int = 1
    max_downloads: int = 10
    event_name: str = ""
    description: str = ""

    @property
    def year(self) -> str:
        return self.date_folder.split("-")[0]

    @property
    def month(self) -> str:
        return self.date_folder.split("-")[1]

    @property
    def day(self) -> str:
        return self.date_folder.split("-")[2]


CASES: dict[str, EventCase] = {
    "test_case_1": EventCase(
        event_id="test_case_1",
        radar_site="KINX",
        date_folder="2024-05-07",
        start_hhmm="0130",
        end_hhmm="0330",
        sweep=1,
        max_downloads=10,
        event_name="KINX 2024-05-07 Main Case",
        description="Tulsa area Oklahoma circulation tracking case.",
    ),
    "test_case_2": EventCase(
        event_id="test_case_2",
        radar_site="KBMX",
        date_folder="2011-04-27",
        start_hhmm="2130",
        end_hhmm="2315",
        sweep=1,
        max_downloads=24,
        event_name="KBMX 2011-04-27 Test Case",
        description="Central Alabama historic tornado supercell case.",
    ),
    "test_case_3": EventCase(
        event_id="test_case_3",
        radar_site="KILX",
        date_folder="2013-11-17",
        start_hhmm="1705",
        end_hhmm="1735",
        sweep=1,
        max_downloads=8,
        event_name="KILX 2013-11-17 Washington IL EF4 Case",
        description="Central Illinois storm outbreak.",
    )
}

KNOWN_CASES = CASES


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def case_dir(event_id: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "outputs" / "cases" / event_id


def known_case(event_id: str) -> EventCase:
    if event_id not in CASES:
        raise KeyError(f"Unknown E-V-E case: {event_id}. Known cases: {', '.join(sorted(CASES))}")
    return CASES[event_id]


def resolve_case(
    event_id: str,
    event_name: str | None = None,
    radar_site: str | None = None,
    date_folder: str | None = None,
    start_hhmm: str | None = None,
    end_hhmm: str | None = None,
    max_downloads: int | None = None,
    sweep: int | None = None,
    description: str | None = None,
) -> EventCase:
    base = CASES.get(
        event_id,
        EventCase(
            event_id=event_id,
            radar_site=radar_site or "",
            date_folder=date_folder or "",
            start_hhmm=start_hhmm or "",
            end_hhmm=end_hhmm or "",
            max_downloads=max_downloads or 10,
            sweep=sweep or 1,
            event_name=event_name or event_id,
            description=description or "",
        ),
    )

    return EventCase(
        event_id=event_id,
        radar_site=radar_site or base.radar_site,
        date_folder=date_folder or base.date_folder,
        start_hhmm=start_hhmm or base.start_hhmm,
        end_hhmm=end_hhmm or base.end_hhmm,
        sweep=base.sweep if sweep is None else int(sweep),
        max_downloads=base.max_downloads if max_downloads is None else int(max_downloads),
        event_name=event_name or base.event_name,
        description=description or base.description,
    )
