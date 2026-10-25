"""Archived 2020 COVID module. Mounted only when NDP_ENABLE_COVID=true."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Form, HTTPException, Request

from ndp.forms import FormError, parse_date
from ndp.ml import covid, registry
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/covid")
PLOTS = [
    ("Confirmed cases in India (JHU)", "generated/covid/actual.png"),
    ("Model fit and 90-day extrapolation", "generated/covid/forecast.png"),
]


def _page_context() -> dict:
    meta = registry.metadata(covid.MODEL) or {}
    return {
        "meta": meta or None,
        "trained": registry.is_trained(covid.MODEL),
        "plots": PLOTS,
        "centres_count": len(covid.test_centres()),
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
