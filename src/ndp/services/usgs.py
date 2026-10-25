"""Recent significant earthquakes from the USGS GeoJSON summary feeds (no key needed)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from ndp.services import ServiceError

FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson"
CACHE_SECONDS = 300


@dataclass(frozen=True)
class QuakeEvent:
    id: str
    magnitude: float
    place: str
    lat: float
    lon: float
    depth_km: float
    time: datetime
    url: str


_cache: tuple[float, list[QuakeEvent]] | None = None


def collapse_duplicates(events: list[QuakeEvent]) -> list[QuakeEvent]:
    """The feed can list one earthquake several times (different contributing networks, e.g. the
    Pacific Tsunami Warning Center). Treat events within 90 seconds, 50 km and 0.5 magnitude of
    each other as the same quake and keep the preferred one: a core-network id ("us...") first,
    then the strongest. Without this a watcher would alert twice for one earthquake."""
    from ndp.ml.hazard import haversine_km

    def preference(e: QuakeEvent) -> tuple[int, float]:
        return (0 if e.id.startswith("us") else 1, -e.magnitude)

    kept: list[QuakeEvent] = []
    for e in sorted(events, key=preference):
        duplicate = any(
            abs((e.time - k.time).total_seconds()) <= 90
            and abs(e.magnitude - k.magnitude) <= 0.5
            and float(haversine_km(e.lat, e.lon, k.lat, k.lon)) <= 50
            for k in kept
        )
        if not duplicate:
            kept.append(e)
    return kept


def recent_events(limit: int = 8) -> list[QuakeEvent]:
    """Strongest events of the past week (magnitude 4.5 and up), strongest first."""
    global _cache
    now = time.monotonic()
    if _cache is None or now - _cache[0] > CACHE_SECONDS:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(FEED_URL)
                resp.raise_for_status()
                features = resp.json()["features"]
            events = []
            for f in features:
                props, (lon, lat, depth) = f["properties"], f["geometry"]["coordinates"][:3]
                if props.get("mag") is None:
                    continue
                events.append(
                    QuakeEvent(
                        id=f["id"],
                        magnitude=float(props["mag"]),
                        place=props.get("place") or "unknown location",
                        lat=float(lat),
                        lon=float(lon),
                        depth_km=max(float(depth or 0.0), 0.0),
                        time=datetime.fromtimestamp(props["time"] / 1000, tz=UTC),
                        url=props.get("url") or "",
                    )
                )
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise ServiceError(f"The USGS feed is unavailable right now ({exc}).") from exc
        events = collapse_duplicates(events)
        events.sort(key=lambda e: e.magnitude, reverse=True)
        _cache = (now, events)
    return _cache[1][:limit]
