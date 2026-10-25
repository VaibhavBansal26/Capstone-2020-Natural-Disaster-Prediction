from __future__ import annotations

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_float
from ndp.ml import registry, tsunami
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/tsunami")
PLOTS = [
    ("Maximum wave height vs magnitude", "generated/tsunami/t1.png"),
    ("Events per decade", "generated/tsunami/t2.png"),
    ("Source locations", "generated/tsunami/t3.png"),
]


def _page_context() -> dict:
    return {
        "meta": registry.metadata(tsunami.MODEL),
        "trained": registry.is_trained(tsunami.MODEL),
        "plots": PLOTS,
    }


@router.get("", name="tsunami_page")
def tsunami_page(request: Request):
    return render(request, "tsunami/page.html", **_page_context())


@router.post("/predict", name="tsunami_predict")
def tsunami_predict(
    request: Request, lat: str = Form(""), lon: str = Form(""), max_height: str = Form("")
):
    inputs = {"lat": lat, "lon": lon, "max_height": max_height}
    try:
        result = tsunami.predict(
            parse_float(lat, "Latitude", -90, 90),
            parse_float(lon, "Longitude", -180, 180),
            parse_float(max_height, "Maximum wave height (m)", 0, 600),
        )
    except FormError as exc:
        return page_or_partial(
            request,
            "tsunami/page.html",
            "tsunami/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "tsunami/page.html",
        "tsunami/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )
