from __future__ import annotations

from fastapi import APIRouter, Form, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_float, parse_text
from ndp.services import ServiceError
from ndp.services.weather import predict_city_weather
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/weather")


def _page_context() -> dict:
    return {"configured": bool(get_settings().openweather_api_key), "meta": None, "trained": True}


@router.get("", name="weather_page")
def weather_page(request: Request):
    return render(request, "weather/page.html", **_page_context())


@router.post("/predict", name="weather_predict")
def weather_predict(
    request: Request, city: str = Form(""), lat: str = Form(""), lon: str = Form("")
):
    inputs = {"city": city, "lat": lat, "lon": lon}
    try:
        result = predict_city_weather(
            parse_text(city, "City", 80),
            parse_float(lat, "Latitude", -90, 90),
            parse_float(lon, "Longitude", -180, 180),
        )
    except (FormError, ServiceError) as exc:
        return page_or_partial(
            request,
            "weather/page.html",
            "weather/_result.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "weather/page.html",
        "weather/_result.html",
        _page_context,
        inputs=inputs,
        result=result,
    )
