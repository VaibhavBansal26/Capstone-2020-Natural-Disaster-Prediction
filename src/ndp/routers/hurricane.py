from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_date, parse_float
from ndp.ml import hurricane, registry
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/hurricane")
PLOTS = [
    ("Predicted vs actual wind", "generated/hurricane/h1.png"),
    ("Maximum wind over time", "generated/hurricane/h2.png"),
    ("Wind distribution", "generated/hurricane/h3.png"),
]


def _page_context() -> dict:
    return {
        "meta": registry.metadata(hurricane.MODEL),
        "trained": registry.is_trained(hurricane.MODEL),
        "plots": PLOTS,
        "today": date.today().isoformat(),
    }


@router.get("", name="hurricane_page")
def hurricane_page(request: Request):
    return render(request, "hurricane/page.html", **_page_context())


@router.post("/predict", name="hurricane_predict")
def hurricane_predict(
    request: Request, date_: str = Form("", alias="date"), lat: str = Form(""), lon: str = Form("")
):
    inputs = {"date": date_, "lat": lat, "lon": lon}
    try:
        result = hurricane.predict(
            parse_date(date_),
            parse_float(lat, "Latitude", -90, 90),
            parse_float(lon, "Longitude", -180, 180),
        )
    except FormError as exc:
        return page_or_partial(
            request,
            "hurricane/page.html",
            "hurricane/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "hurricane/page.html",
        "hurricane/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )
