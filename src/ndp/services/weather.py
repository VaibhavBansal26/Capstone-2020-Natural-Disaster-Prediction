"""Live weather lookups via OpenWeatherMap and the legacy k-nearest-neighbours check.

The original app trained a KNN on the ten weather stations nearest to a point and used it
to "predict" the weather in a named city, then compared that with the city's reported
weather. That is kept as a demonstration, but the API key now comes from settings and
nothing is accumulated across requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from sklearn.neighbors import KNeighborsClassifier

from ndp.config import get_settings
from ndp.services import ServiceError

FIND_URL = "https://api.openweathermap.org/data/2.5/find"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


@dataclass(frozen=True)
class WeatherResult:
    prediction: str
    actual: str
    matched: bool
    station_count: int
    labels: list[str]
    city: str
    lat: float
    lon: float
    temperature_c: float | None = None


def _features(point: dict) -> list[float]:
    main, wind, clouds = point["main"], point.get("wind", {}), point.get("clouds", {})
    return [
        float(main["temp"]),
        float(main["pressure"]),
        float(main["humidity"]),
        float(wind.get("speed", 0.0)),
        float(wind.get("deg", 0.0)),
        float(clouds.get("all", 0.0)),
    ]


def predict_city_weather(city: str, lat: float, lon: float) -> WeatherResult:
    key = get_settings().openweather_api_key
    if not key:
        raise ServiceError("OpenWeatherMap is not configured (set NDP_OPENWEATHER_API_KEY).")
    try:
        with httpx.Client(timeout=15.0) as client:
            nearby = client.get(FIND_URL, params={"lat": lat, "lon": lon, "cnt": 10, "appid": key})
            nearby.raise_for_status()
            current = client.get(WEATHER_URL, params={"q": city, "appid": key, "units": "metric"})
            current.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200]
        raise ServiceError(f"OpenWeatherMap returned {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise ServiceError(f"OpenWeatherMap request failed: {exc}") from exc

    stations = nearby.json().get("list", [])
    if len(stations) < 2:
        raise ServiceError("Not enough nearby weather stations to fit a classifier.")
    rows = [_features(p) for p in stations]
    labels = [p["weather"][0]["main"] for p in stations]

    clf = KNeighborsClassifier(n_neighbors=min(5, len(rows)))
    clf.fit(rows, labels)

    here = current.json()
    # The city's own reading is in metric units; the station features are in Kelvin
    # (default units), so re-request-free conversion keeps the feature space consistent.
    here_kelvin = {**here, "main": {**here["main"], "temp": here["main"]["temp"] + 273.15}}
    here_features = _features(here_kelvin)
    prediction = str(clf.predict([here_features])[0])
    actual = str(here["weather"][0]["main"])
    return WeatherResult(
        prediction=prediction,
        actual=actual,
        matched=prediction == actual,
        station_count=len(rows),
        labels=sorted(set(labels)),
        city=here.get("name", city),
        lat=lat,
        lon=lon,
        temperature_c=float(here["main"]["temp"]),
    )
