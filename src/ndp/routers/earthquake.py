from __future__ import annotations

import logging
from collections import deque
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Form, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_date, parse_float
from ndp.ml import earthquake as eq
from ndp.ml import registry
from ndp.services import ServiceError, alerting, geo, usgs
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/earthquake")
USGS_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
PLOTS = [
    ("Magnitude distribution", "generated/earthquake/e1.png"),
    ("Events by month", "generated/earthquake/e2.png"),
    ("Events by year", "generated/earthquake/e3.png"),
    ("Magnitude vs depth", "generated/earthquake/e4.png"),
]

log = logging.getLogger("ndp.earthquake")


def _auto_alert(entry: dict, lat: float, lon: float, depth: float, on, magnitude: float) -> None:
    """Background job: look up the place and nearby services, then alert the configured
    contacts. Mirrors the 2020 behaviour (alert at M >= 6) but is opt-in via NDP_AUTO_ALERT and
    includes the hospitals and police found around the epicentre in the message."""
    try:
        place = geo.reverse_geocode(lat, lon)
        try:
            places = geo.nearby_services(lat, lon, get_settings().alert_radius_km)
        except ServiceError:
            places = []  # still alert; the facility list is a bonus, not a precondition
        body = alerting.build_earthquake_message(
            place.name, lat, lon, magnitude, depth, on.isoformat(), places
        )
        outcome = alerting.dispatch(place.name, body)
        if outcome.channels:
            entry["alert_status"] = "sent: " + ", ".join(outcome.channels)
        elif outcome.would_send:
            entry["alert_status"] = "dry run: " + ", ".join(outcome.would_send)
        else:
            entry["alert_status"] = "not sent (no channel configured)"
    except Exception:  # background task: never let it vanish silently
        log.exception("automatic earthquake alert failed")
        entry["alert_status"] = "failed (see server log)"


# Replaces the legacy MySQL table that only ever stored the last few queries.
_recent: deque[dict] = deque(maxlen=8)


def _stats() -> dict | None:
    try:
        return eq.stats()
    except registry.ModelNotTrained:
        return None


def _page_context() -> dict:
    return {
        "meta": registry.metadata(eq.MODEL),
        "trained": registry.is_trained(eq.MODEL),
        "recent": list(_recent),
        "stats": _stats(),
        "usgs_feed_url": USGS_FEED_URL,
        "today": date.today().isoformat(),
    }


def _prefill(lat: str | None, lon: str | None, depth: str | None) -> dict[str, str]:
    """Valid numeric query parameters become form defaults; anything else is ignored."""
    limits = {"lat": (lat, -90, 90), "lon": (lon, -180, 180), "depth": (depth, 0, 800)}
    out: dict[str, str] = {}
    for key, (raw, lo, hi) in limits.items():
        try:
            if raw is not None and lo <= parse_float(raw, key) <= hi:
                out[key] = f"{float(raw):g}"
        except FormError:
            continue
    return out


@router.get("", name="earthquake_page")
def earthquake_page(
    request: Request, lat: str | None = None, lon: str | None = None, depth: str | None = None
):
    context = _page_context()
    if prefill := _prefill(lat, lon, depth):
        context["inputs"] = prefill
    return render(request, "earthquake/page.html", **context)


@router.get("/events", name="earthquake_events")
def earthquake_events(request: Request):
    """The week's strongest real earthquakes, each with a one-click nearby-help lookup.
    Loaded by HTMX so a slow USGS response never delays the page."""
    try:
        events = usgs.recent_events(8)
    except ServiceError as exc:
        return render(request, "earthquake/_events.html", error=str(exc))
    return render(request, "earthquake/_events.html", events=events)


@router.get("/graphs", name="earthquake_graphs")
def earthquake_graphs(request: Request):
    return render(
        request,
        "earthquake/graphs.html",
        stats=_stats(),
        plots=PLOTS,
        meta=registry.metadata(eq.MODEL),
        trained=registry.is_trained(eq.MODEL),
    )


@router.post("/predict", name="earthquake_predict")
def earthquake_predict(
    request: Request,
    background_tasks: BackgroundTasks,
    lat: str = Form(""),
    lon: str = Form(""),
    depth: str = Form(""),
    date_: str = Form("", alias="date"),
):
    inputs = {"lat": lat, "lon": lon, "depth": depth, "date": date_}
    try:
        f_lat = parse_float(lat, "Latitude", -90, 90)
        f_lon = parse_float(lon, "Longitude", -180, 180)
        f_depth = parse_float(depth, "Depth (km)", 0, 800)
        on = parse_date(date_)
        result = eq.predict(f_lat, f_lon, f_depth, on)
    except FormError as exc:
        return page_or_partial(
            request,
            "earthquake/page.html",
            "earthquake/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    entry = {
        "lat": f_lat,
        "lon": f_lon,
        "depth": f_depth,
        "date": on.isoformat(),
        "magnitude": round(result.magnitude, 2),
        "alert": result.alert,
        "alert_status": None,
    }
    _recent.appendleft(entry)
    auto_alert = bool(result.alert and get_settings().auto_alert)
    if auto_alert:
        entry["alert_status"] = "queued"
        background_tasks.add_task(_auto_alert, entry, f_lat, f_lon, f_depth, on, result.magnitude)
    return page_or_partial(
        request,
        "earthquake/page.html",
        "earthquake/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
        auto_alert=auto_alert,
    )
