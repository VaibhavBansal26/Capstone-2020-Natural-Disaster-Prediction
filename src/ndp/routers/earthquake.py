from __future__ import annotations

from collections import deque
from datetime import date

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_date, parse_float
from ndp.ml import earthquake as eq
from ndp.ml import registry
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/earthquake")
USGS_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
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


@router.get("", name="earthquake_page")
def earthquake_page(request: Request):
    return render(request, "earthquake/page.html", **_page_context())


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
    _recent.appendleft(
        {
            "lat": f_lat,
            "lon": f_lon,
            "depth": f_depth,
            "date": on.isoformat(),
            "magnitude": round(result.magnitude, 2),
            "alert": result.alert,
        }
    )
    return page_or_partial(
        request,
        "earthquake/page.html",
        "earthquake/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )
