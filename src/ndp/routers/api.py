"""JSON API mirroring the HTML prediction forms (interactive docs at /docs)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ndp.ml import flood, hazard, hurricane, storm, tsunami

router = APIRouter(prefix="/api/v1", tags=["predict"])


class HazardIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(default=300.0, ge=25, le=2000)


class TsunamiIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    max_height_m: float = Field(ge=0, le=600)


class FloodForecastIn(BaseModel):
    """Hoppers Crossing gauge (Werribee River, Melbourne) only."""

    level_now_m: float = Field(ge=0, le=15)
    level_1h_ago_m: float = Field(ge=0, le=15)
    level_3h_ago_m: float = Field(ge=0, le=15)
    rain_3h_mm: float = Field(ge=0, le=500)
    rain_6h_mm: float = Field(ge=0, le=500)
    rain_24h_mm: float = Field(ge=0, le=500)


class HurricaneIn(BaseModel):
    date: date
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class StormIn(BaseModel):
    temperature_f: float
    pressure_hpa: float
    humidity_pct: float = Field(ge=0, le=100)
    wind_direction_deg: float = Field(ge=0, le=360)
    wind_speed: float = Field(ge=0)


@router.post("/earthquake/hazard", name="api_earthquake_hazard")
def api_earthquake_hazard(body: HazardIn) -> dict:
    """Historical seismic hazard around a point (a rate, not a prediction of when)."""
    return asdict(hazard.predict(body.lat, body.lon, body.radius_km))


@router.post("/tsunami/predict", name="api_tsunami")
def api_tsunami(body: TsunamiIn) -> dict:
    return asdict(tsunami.predict(body.lat, body.lon, body.max_height_m))


@router.post("/flood/forecast", name="api_flood_forecast")
def api_flood_forecast(body: FloodForecastIn) -> dict:
    """1, 3 and 6 hour river-level forecast for the Hoppers Crossing gauge only."""
    return asdict(
        flood.predict(
            body.level_now_m,
            body.level_1h_ago_m,
            body.level_3h_ago_m,
            body.rain_3h_mm,
            body.rain_6h_mm,
            body.rain_24h_mm,
        )
    )


@router.post("/hurricane/predict", name="api_hurricane")
def api_hurricane(body: HurricaneIn) -> dict:
    return asdict(hurricane.predict(body.date, body.lat, body.lon))


@router.post("/storm/predict", name="api_storm")
def api_storm(body: StormIn) -> dict:
    return asdict(
        storm.predict(
            body.temperature_f,
            body.pressure_hpa,
            body.humidity_pct,
            body.wind_direction_deg,
            body.wind_speed,
        )
    )
