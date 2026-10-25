from __future__ import annotations

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_float
from ndp.ml import registry, storm
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/storm")
PLOTS = [
    ("Decision tree (top levels)", "generated/storm/tree.png"),
    ("Class distribution", "generated/storm/classes.png"),
]


def _page_context() -> dict:
    meta = registry.metadata(storm.MODEL)
    return {
        "meta": meta,
        "trained": registry.is_trained(storm.MODEL),
        "plots": PLOTS,
        "classes": (meta or {}).get("classes", []),
    }


@router.get("", name="storm_page")
def storm_page(request: Request):
    return render(request, "storm/page.html", **_page_context())


@router.post("/predict", name="storm_predict")
def storm_predict(
    request: Request,
    temperature: str = Form(""),
    pressure: str = Form(""),
    humidity: str = Form(""),
    wind_direction: str = Form(""),
    wind_speed: str = Form(""),
):
    inputs = {
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "wind_direction": wind_direction,
        "wind_speed": wind_speed,
    }
    try:
        result = storm.predict(
            parse_float(temperature, "Temperature (°F)", -100, 150),
            parse_float(pressure, "Pressure (hPa)", 800, 1100),
            parse_float(humidity, "Humidity (%)", 0, 100),
            parse_float(wind_direction, "Wind direction (°)", 0, 360),
            parse_float(wind_speed, "Wind speed", 0, 200),
        )
    except FormError as exc:
        return page_or_partial(
            request,
            "storm/page.html",
            "storm/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "storm/page.html",
        "storm/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )
