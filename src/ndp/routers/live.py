"""Live hazards page: GDACS alerts, Open-Meteo forecast, real-event watcher status.

Each block is its own HTMX partial that loads after the page, so a slow upstream never delays
the page itself. Nothing on this page is an official warning."""

from __future__ import annotations

import html
import time

from fastapi import APIRouter, Form, Request

from ndp.services import ServiceError, gdacs, geo, openmeteo, watcher
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/live")

DEFAULT_PLACE = "Mumbai, India"
MAX_PLACE_LEN = 200
LEVEL_TONE = {"Red": "danger", "Orange": "warn", "Green": "ok"}
LEVEL_COLOR = {"Red": "#be123c", "Orange": "#ea580c", "Green": "#16a34a"}
LEVEL_RADIUS = {"Red": 11, "Orange": 9, "Green": 6}
KIND_ICON_ORDER = ("EQ", "TC", "FL", "VO", "DR", "WF")

_GEOCODE_TTL = 3600.0
_geocode_cache: dict[str, tuple[float, geo.Place]] = {}


def _geocode(query: str) -> geo.Place:
    """Nominatim asks for modest request rates and this runs on every page load, so remember
    lookups for an hour."""
    key = " ".join(query.lower().split())
    hit = _geocode_cache.get(key)
    if hit and time.monotonic() - hit[0] < _GEOCODE_TTL:
        return hit[1]
    place = geo.geocode(query)
    if len(_geocode_cache) > 200:
        _geocode_cache.clear()
    _geocode_cache[key] = (time.monotonic(), place)
    return place


def _short_countries(country: str, keep: int = 3) -> tuple[str, str]:
    """(display text, spoken text) for a possibly very long country list."""
    parts = [p.strip() for p in country.split(",") if p.strip()]
    if len(parts) <= keep:
        return country, " and ".join(parts) if len(parts) > 1 else country
    more = len(parts) - keep
    shown = ", ".join(parts[:keep])
    return f"{shown} and {more} more", f"{shown} and {more} more countries"


def _when(e: gdacs.GdacsEvent) -> str:
    start, end = e.from_date, e.to_date
    if start and end and start.date() != end.date():
        same_year = start.year == end.year
        left = start.strftime("%d %b" if same_year else "%d %b %Y")
        return f"{left} to {end.strftime('%d %b %Y')}"
    when = end or start
    return when.strftime("%d %b %Y") if when else "date unknown"


def _event_view(e: gdacs.GdacsEvent) -> dict:
    country_text, country_spoken = _short_countries(e.country) if e.country else ("", "")
    where = f" in {country_spoken}" if country_spoken else ""
    speak = f"{e.alert_level} alert. {e.name.rstrip('.')}{where}. {e.summary.rstrip('.')}."
    return {
        "e": e,
        "tone": LEVEL_TONE.get(e.alert_level, "muted"),
        "country": country_text,
        "when": _when(e),
        "speak": speak,
    }


def _gdacs_context(events: list[gdacs.GdacsEvent]) -> dict:
    views = [_event_view(e) for e in events]
    markers = [
        {
            "lat": v["e"].lat,
            "lon": v["e"].lon,
            "title": v["e"].name,
            "color": LEVEL_COLOR.get(v["e"].alert_level, "#2563eb"),
            "radius": LEVEL_RADIUS.get(v["e"].alert_level, 6),
            "lines": [
                f"{v['e'].alert_level} alert, {v['e'].kind_name}",
                v["country"],
                v["when"],
                v["e"].summary,
            ],
            "popup": (
                '<p><a class="underline" target="_blank" rel="noopener" href="'
                + html.escape(v["e"].url)
                + '">GDACS report</a></p>'
                if v["e"].url
                else ""
            ),
        }
        for v in views
        if v["e"].lat is not None and v["e"].lon is not None
    ]
    counts = {level: sum(1 for e in events if e.alert_level == level) for level in LEVEL_TONE}
    return {
        "events": views,
        "markers": markers,
        "counts": counts,
        "unmapped": len(views) - len(markers),
    }


def _bars(hours, attr: str, floor: float, threshold: float | None = None) -> dict:
    """Bar-chart data grouped by local day, heights as percentages of a fixed scale."""
    values = [getattr(h, attr) or 0.0 for h in hours]
    scale = max([floor, *values])
    groups: list[dict] = []
    for h, v in zip(hours, values, strict=True):
        if not groups or groups[-1]["day"] != h.time.date():
            groups.append({"day": h.time.date(), "label": h.time.strftime("%a %d %b"), "bars": []})
        groups[-1]["bars"].append(
            {
                "pct": max(v / scale * 100, 2.0 if v > 0 else 0.0),
                "over": threshold is not None and round(v, 1) >= threshold,
                "tip": f"{h.time.strftime('%a %H:%M')}: {v:g}",
            }
        )
    return {"groups": groups, "scale": scale, "count": len(hours), "empty": not any(values)}


def _hours_label(hours: int) -> str:
    return f"{hours} hours" if hours != 24 else "24 hours"


def _forecast_context(place: geo.Place, fc: openmeteo.Forecast) -> dict:
    spoken = [
        f"Weather outlook for {place.name.split(',')[0]}, from model forecast data, "
        f"not an official warning.",
        f"In the next {_hours_label(len(fc.hours))}, the wettest 24 hours bring up to "
        f"{fc.precip_24h_max_mm:g} millimetres of rain."
        + (
            f" Wind gusts reach {fc.max_gust_kmh:g} kilometres per hour."
            if fc.max_gust_kmh is not None
            else ""
        ),
        *(a.text for a in fc.advisories),
    ]
    return {
        "place": place,
        "fc": fc,
        "rain_chart": _bars(fc.hours, "precip_mm", 5.0),
        "gust_chart": _bars(fc.hours, "gust_kmh", 70.0, openmeteo.GUST_KMH),
        "speak": " ".join(spoken),
        "gust_threshold": openmeteo.GUST_KMH,
    }


def _page_context() -> dict:
    return {"default_place": DEFAULT_PLACE, "watcher": watcher.status()}


@router.get("", name="live_page")
def live_page(request: Request):
    return render(request, "live/page.html", **_page_context())


@router.get("/gdacs", name="live_gdacs")
def live_gdacs(request: Request):
    try:
        events = gdacs.recent_alerts(30)
    except ServiceError as exc:
        return render(request, "live/_gdacs.html", error=str(exc))
    return render(request, "live/_gdacs.html", **_gdacs_context(events))


@router.post("/forecast", name="live_forecast")
def live_forecast(request: Request, place: str = Form("")):
    query = " ".join(place.split())
    inputs = {"place": query or DEFAULT_PLACE}
    result: dict
    if not query:
        result = {"inputs": inputs, "error": "Enter a place name, for example Mumbai, India."}
    elif len(query) > MAX_PLACE_LEN:
        result = {"inputs": inputs, "error": f"Place names are limited to {MAX_PLACE_LEN} letters."}
    else:
        try:
            found = _geocode(query)
            fc = openmeteo.forecast(found.lat, found.lon, days=3)
            result = {"inputs": inputs, **_forecast_context(found, fc)}
        except ServiceError as exc:
            result = {"inputs": inputs, "error": str(exc)}
    return page_or_partial(
        request, "live/page.html", "live/_forecast.html", _page_context, **result
    )


@router.get("/watcher", name="live_watcher")
def live_watcher(request: Request):
    return render(request, "live/_watcher.html", watcher=watcher.status())
