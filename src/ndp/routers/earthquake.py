from __future__ import annotations

from collections import deque

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_float
from ndp.ml import catalogue, hazard, registry
from ndp.services import ServiceError, usgs
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/earthquake")
USGS_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
DEFAULT_RADIUS_KM = 300
RADIUS_MIN_KM = 25
RADIUS_MAX_KM = 2000
# Plots written by `hazard.train()`; shown on the page that uses the hazard model.
HAZARD_PLOTS = [
    ("Significant earthquakes, 1965 to 2016", "generated/earthquake/hz1_seismicity.png"),
    (
        "Magnitude-frequency relation (Gutenberg-Richter)",
        "generated/earthquake/hz2_gutenberg_richter.png",
    ),
]
# Plots of the legacy catalogue statistics (e1 to e4), kept on the graphical-analysis page.
PLOTS = [
    ("Magnitude distribution", "generated/earthquake/e1.png"),
    ("Events by month", "generated/earthquake/e2.png"),
    ("Events by year", "generated/earthquake/e3.png"),
    ("Magnitude vs depth", "generated/earthquake/e4.png"),
]

# Replaces the legacy MySQL table that only ever stored the last few queries.
_recent: deque[dict] = deque(maxlen=8)


def _stats() -> dict | None:
    try:
        return catalogue.stats()
    except registry.ModelNotTrained:
        return None


def _page_context() -> dict:
    return {
        "meta": registry.metadata(hazard.MODEL),
        "trained": registry.is_trained(hazard.MODEL),
        "recent": list(_recent),
        "stats": _stats(),
        "plots": HAZARD_PLOTS,
        "plots_title": "How the hazard is estimated",
        "usgs_feed_url": USGS_FEED_URL,
        "default_radius_km": DEFAULT_RADIUS_KM,
    }


def _prefill(lat: str | None, lon: str | None, radius_km: str | None) -> dict[str, str]:
    """Valid numeric query parameters become form defaults; anything else is ignored.
    (`depth` used to be a model input; it is accepted in links but no longer used.)"""
    limits = {
        "lat": (lat, -90, 90),
        "lon": (lon, -180, 180),
        "radius_km": (radius_km, RADIUS_MIN_KM, RADIUS_MAX_KM),
    }
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
    request: Request,
    lat: str | None = None,
    lon: str | None = None,
    depth: str | None = None,  # accepted for old links; ignored
    radius_km: str | None = None,
):
    context = _page_context()
    if prefill := _prefill(lat, lon, radius_km):
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
        meta=registry.metadata(hazard.MODEL),
        trained=registry.is_trained(hazard.MODEL),
    )


@router.post("/predict", name="earthquake_predict")
def earthquake_predict(
    request: Request,
    lat: str = Form(""),
    lon: str = Form(""),
    radius_km: str = Form(str(DEFAULT_RADIUS_KM)),
):
    inputs = {"lat": lat, "lon": lon, "radius_km": radius_km}
    try:
        f_lat = parse_float(lat, "Latitude", -90, 90)
        f_lon = parse_float(lon, "Longitude", -180, 180)
        f_radius = parse_float(
            radius_km or str(DEFAULT_RADIUS_KM), "Radius (km)", RADIUS_MIN_KM, RADIUS_MAX_KM
        )
    except FormError as exc:
        return page_or_partial(
            request,
            "earthquake/page.html",
            "earthquake/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    result = hazard.predict(f_lat, f_lon, f_radius)
    p10 = (result.prob_10y or {}).get("6.0")
    _recent.appendleft(
        {
            "lat": f_lat,
            "lon": f_lon,
            "radius_km": f_radius,
            "level": result.level,
            "elevated": result.elevated,
            "prob_10y_m6": p10,
        }
    )
    return page_or_partial(
        request,
        "earthquake/page.html",
        "earthquake/_result.html",
        _page_context,
        inputs={"lat": f_lat, "lon": f_lon, "radius_km": f_radius},
        result=result,
    )
