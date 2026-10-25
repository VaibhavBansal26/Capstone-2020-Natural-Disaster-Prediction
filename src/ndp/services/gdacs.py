"""Recent disaster alerts from GDACS, the Global Disaster Alert and Coordination System
(a UN OCHA / European Commission cooperation). No key needed.

The alert levels (Green, Orange, Red) are GDACS's own assessments of humanitarian impact. This
app only relays them; it does not compute or second-guess them, and nothing here is an official
warning for any particular place.

Endpoint (verified against the live service): the event-list `SEARCH` endpoint returns a GeoJSON
FeatureCollection, newest first, at most 100 features per page (`pagenumber` pages through).
`alertlevel=Orange;Red` is therefore requested separately from `alertlevel=Green`, otherwise the
many Green wildfire and earthquake entries would push the rare Orange and Red ones off page one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from ndp.services import ServiceError
from ndp.services.geo import USER_AGENT

log = logging.getLogger("ndp.gdacs")

EVENT_LIST_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
EVENT_TYPES = "EQ;TC;FL;VO;DR;WF"
WINDOW_DAYS = 30
CACHE_SECONDS = 600
PARTIAL_CACHE_SECONDS = 60  # when only part of the answer could be fetched, retry soon
TIMEOUT = 15.0

KIND_NAMES = {
    "EQ": "Earthquake",
    "TC": "Tropical cyclone",
    "FL": "Flood",
    "VO": "Volcano",
    "DR": "Drought",
    "WF": "Wildfire",
}
LEVEL_RANK = {"Red": 0, "Orange": 1, "Green": 2}


@dataclass(frozen=True)
class GdacsEvent:
    id: str
    kind: str
    kind_name: str
    name: str
    alert_level: str
    country: str
    lat: float | None
    lon: float | None
    from_date: datetime | None
    to_date: datetime | None
    url: str
    summary: str


_clock = time.monotonic
_cache: tuple[float, float, list[GdacsEvent]] | None = None  # (stored_at, ttl, events)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_date(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _positions(coords) -> list[tuple[float, float]]:
    """Flatten any GeoJSON `coordinates` nesting into (lon, lat) pairs, ignoring junk."""
    if not isinstance(coords, (list, tuple)):
        return []
    if len(coords) >= 2 and all(
        isinstance(c, (int, float)) and not isinstance(c, bool) for c in coords[:2]
    ):
        return [(float(coords[0]), float(coords[1]))]
    out: list[tuple[float, float]] = []
    for part in coords:
        out.extend(_positions(part))
    return out


def _geometry_positions(geometry) -> list[tuple[float, float]]:
    if not isinstance(geometry, dict):
        return []
    if geometry.get("type") == "GeometryCollection":
        out: list[tuple[float, float]] = []
        for g in geometry.get("geometries") or []:
            out.extend(_geometry_positions(g))
        return out
    return _positions(geometry.get("coordinates"))


def representative_point(feature: dict) -> tuple[float | None, float | None]:
    """(lat, lon) for a feature: the point itself, or the centre of the bounding box of a line or
    polygon geometry (it ignores antimeridian wrapping, fine for a map marker). Falls back to the
    feature's `bbox`. (None, None) when nothing usable is present or the numbers are out of
    range."""
    pts = _geometry_positions(feature.get("geometry"))
    if not pts:
        bbox = feature.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            pts = _positions([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
    if not pts:
        return None, None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    lon, lat = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None
    return lat, lon


def _text(value) -> str:
    return " ".join(str(value).split()) if isinstance(value, (str, int, float)) else ""


def _summary(props: dict, kind: str) -> str:
    sev = props.get("severitydata")
    text = _text(sev.get("severitytext")) if isinstance(sev, dict) else ""
    # GDACS floods report "Magnitude 0", which says nothing; fall back to the event description.
    if text and not text.startswith("Magnitude 0"):
        return text
    return _text(props.get("description")) or KIND_NAMES.get(kind, kind)


def parse_event(feature) -> GdacsEvent | None:
    """One GeoJSON feature to a GdacsEvent, or None when it lacks the essentials."""
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        return None
    kind = _text(props.get("eventtype")).upper()
    event_id = _text(props.get("eventid"))
    level = _text(props.get("alertlevel")).capitalize()
    if kind not in KIND_NAMES or not event_id or level not in LEVEL_RANK:
        return None
    lat, lon = representative_point(feature)
    country = _text(props.get("country"))
    if not country and isinstance(props.get("affectedcountries"), list):
        country = ", ".join(
            _text(c.get("countryname")) for c in props["affectedcountries"] if isinstance(c, dict)
        ).strip(", ")
    urls = props.get("url") if isinstance(props.get("url"), dict) else {}
    report = _text(urls.get("report")) if urls else ""
    if not report.startswith("https://"):
        report = ""
    kind_name = KIND_NAMES[kind]
    return GdacsEvent(
        id=f"{kind}-{event_id}",
        kind=kind,
        kind_name=kind_name,
        name=_text(props.get("name")) or _text(props.get("description")) or kind_name,
        alert_level=level,
        country=country,
        lat=lat,
        lon=lon,
        from_date=_parse_date(props.get("fromdate")),
        to_date=_parse_date(props.get("todate")),
        url=report,
        summary=_summary(props, kind),
    )


def parse_events(payload) -> list[GdacsEvent]:
    """All usable events of a FeatureCollection; malformed features are skipped."""
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError("response has no 'features' list")
    return [e for e in (parse_event(f) for f in features) if e is not None]


def sort_events(events: list[GdacsEvent]) -> list[GdacsEvent]:
    """Red, then Orange, then Green; within a level the most recent activity first."""
    floor = datetime.min.replace(tzinfo=UTC)

    def recency(e: GdacsEvent) -> datetime:
        return e.to_date or e.from_date or floor

    return sorted(events, key=lambda e: (LEVEL_RANK[e.alert_level], -recency(e).timestamp()))


def _fetch_levels(client: httpx.Client, levels: str, max_pages: int) -> list[GdacsEvent]:
    start = (_utcnow() - timedelta(days=WINDOW_DAYS)).date().isoformat()
    end = _utcnow().date().isoformat()
    events: list[GdacsEvent] = []
    for page in range(1, max_pages + 1):
        resp = client.get(
            EVENT_LIST_URL,
            params={
                "eventlist": EVENT_TYPES,
                "fromdate": start,
                "todate": end,
                "alertlevel": levels,
                "pagenumber": page,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        events.extend(parse_events(payload))
        if len(payload.get("features", [])) < 100:  # a full page means there may be more
            break
    return events


def _load() -> tuple[list[GdacsEvent], bool]:
    """(events, complete). Orange and Red are required; Green is best effort."""
    try:
        with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            important = _fetch_levels(client, "Orange;Red", max_pages=3)
            try:
                green = _fetch_levels(client, "Green", max_pages=1)
                complete = True
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("GDACS Green events unavailable: %s", exc)
                green, complete = [], False
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError(f"The GDACS alert service is unavailable right now ({exc}).") from exc
    unique: dict[str, GdacsEvent] = {}
    for e in important + green:
        unique.setdefault(e.id, e)
    return sort_events(list(unique.values())), complete


def recent_alerts(limit: int = 30) -> list[GdacsEvent]:
    """Events of the last ~30 days (earthquakes, cyclones, floods, volcanoes, droughts, wildfires),
    Red first, then Orange, then Green, most recent first within a level. Cached for 10 minutes."""
    global _cache
    now = _clock()
    if _cache is None or now - _cache[0] > _cache[1]:
        events, complete = _load()
        _cache = (now, CACHE_SECONDS if complete else PARTIAL_CACHE_SECONDS, events)
    return _cache[2][: max(0, limit)]


def clear_cache() -> None:
    global _cache
    _cache = None
