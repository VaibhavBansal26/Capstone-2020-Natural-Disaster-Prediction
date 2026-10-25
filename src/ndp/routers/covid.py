"""COVID-19 pages: live totals, the 2020 India case curve, test centres and maps.

Mounted unless NDP_ENABLE_COVID=false."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Form, HTTPException, Request

from ndp.forms import FormError, parse_date, parse_float
from ndp.ml import covid, registry
from ndp.ml.features import DataMissing
from ndp.services import ServiceError, covidstats, geo
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/covid")
PLOTS = [
    ("Confirmed cases in India (JHU)", "generated/covid/actual.png"),
    ("Model fit and 90-day extrapolation", "generated/covid/forecast.png"),
]


def _centres_count() -> int | None:
    try:
        return len(covid.test_centres())
    except DataMissing:
        return None


def _page_context() -> dict:
    meta = registry.metadata(covid.MODEL) or {}
    return {
        "meta": meta or None,
        "trained": registry.is_trained(covid.MODEL),
        "plots": PLOTS,
        "centres_count": _centres_count(),
        "last_data_date": meta.get("last_date"),
        "today": date.today().isoformat(),
    }


@router.get("", name="covid_page")
def covid_page(request: Request):
    return render(request, "covid/page.html", **_page_context())


@router.post("/predict", name="covid_predict")
def covid_predict(request: Request, date_: str = Form("", alias="date")):
    inputs = {"date": date_}
    try:
        result = covid.predict(parse_date(date_))
    except FormError as exc:
        return page_or_partial(
            request,
            "covid/page.html",
            "covid/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "covid/page.html",
        "covid/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )


@router.get("/maps", name="covid_maps")
def covid_maps(request: Request):
    return render(request, "covid/maps.html", meta=None, trained=True)


def _feature(lat: float, lon: float, props: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


@router.get("/data/{layer}", name="covid_geojson")
def covid_geojson(layer: str) -> dict:
    try:
        return _geojson(layer)
    except DataMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _geojson(layer: str) -> dict:
    if layer == "centres":
        feats = [
            _feature(
                c["lat"], c["lon"], {"name": c["name"], "popup": f"{c['name']}<br>{c['address']}"}
            )
            for c in covid.test_centres()
        ]
    elif layer == "states":
        feats = [
            _feature(
                s["lat"],
                s["lon"],
                {
                    "name": s["state"],
                    "active": s["active"],
                    "confirmed": s["confirmed"],
                    "deaths": s["deaths"],
                    "recovered": s["recovered"],
                    "popup": (
                        f"<b>{s['state']}</b><br>Active {s['active']}<br>Confirmed {s['confirmed']}"
                        f"<br>Deaths {s['deaths']}<br>Recovered {s['recovered']}"
                    ),
                },
            )
            for s in covid.state_snapshot()
        ]
    elif layer == "cities":
        feats = [
            _feature(
                c["lat"],
                c["lon"],
                {
                    "name": c["city"],
                    "active": c["active"],
                    "confirmed": c["confirmed"],
                    "deaths": c["deaths"],
                    "popup": (
                        f"<b>{c['city']}</b><br>Active {c['active']}<br>Confirmed {c['confirmed']}"
                        f"<br>Deaths {c['deaths']}"
                    ),
                },
            )
            for c in covid.city_snapshot()
        ]
    else:
        raise HTTPException(status_code=404, detail="unknown layer")
    return {"type": "FeatureCollection", "features": feats}


@router.get("/stats", name="covid_stats")
def covid_stats(request: Request):
    """Live world and India totals; loaded by HTMX when the page opens so it never blocks."""
    try:
        rows = [covidstats.world(), covidstats.india()]
    except ServiceError as exc:
        return render(request, "covid/_stats.html", error=str(exc))
    return render(request, "covid/_stats.html", rows=rows)


@router.post("/centres", name="covid_centres")
def covid_centres(
    request: Request, place: str = Form(""), lat: str = Form(""), lon: str = Form("")
):
    """Nearest ICMR test centres to a place name or a coordinate."""
    inputs = {"place": place, "lat": lat, "lon": lon}
    try:
        if place.strip():
            found = geo.geocode(place.strip())
            origin = {"name": found.name, "lat": found.lat, "lon": found.lon}
        else:
            f_lat = parse_float(lat, "Latitude", -90, 90)
            f_lon = parse_float(lon, "Longitude", -180, 180)
            origin = {"name": f"{f_lat:.3f}, {f_lon:.3f}", "lat": f_lat, "lon": f_lon}
        centres = covid.nearest_centres(origin["lat"], origin["lon"], 5)
    except (FormError, ServiceError) as exc:
        return render(request, "covid/_centres.html", inputs=inputs, error=str(exc))
    except DataMissing as exc:
        return render(request, "covid/_centres.html", inputs=inputs, error=str(exc))
    return render(request, "covid/_centres.html", inputs=inputs, origin=origin, centres=centres)
