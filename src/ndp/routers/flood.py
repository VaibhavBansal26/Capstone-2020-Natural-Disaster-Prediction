from __future__ import annotations

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_float
from ndp.ml import flood, registry
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/flood")
PLOTS = [
    ("Hourly rainfall", "generated/flood/f1.png"),
    ("River level", "generated/flood/f2.png"),
    ("Level vs cumulative rainfall", "generated/flood/f3.png"),
    ("Fit on training data", "generated/flood/f4.png"),
    ("Fit on held-out data", "generated/flood/f5.png"),
]


def _page_context() -> dict:
    return {
        "meta": registry.metadata(flood.MODEL),
        "trained": registry.is_trained(flood.MODEL),
        "plots": PLOTS,
        "threshold_m": flood.FLOOD_THRESHOLD_M,
    }


@router.get("", name="flood_page")
def flood_page(request: Request):
    return render(request, "flood/page.html", **_page_context())


@router.post("/predict", name="flood_predict")
def flood_predict(request: Request, cumulative_rainfall: str = Form("")):
    inputs = {"cumulative_rainfall": cumulative_rainfall}
    try:
        result = flood.predict(
            parse_float(cumulative_rainfall, "Cumulative rainfall (mm)", 0, 5000)
        )
    except FormError as exc:
        return page_or_partial(
            request,
            "flood/page.html",
            "flood/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "flood/page.html",
        "flood/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )
