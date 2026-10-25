"""JSON API mirroring the HTML prediction forms. Phase 3 (agent tools) builds on this."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ndp.ml import earthquake as eq
from ndp.ml import flood, hurricane, storm, tsunami

router = APIRouter(prefix="/api/v1", tags=["predict"])


class EarthquakeIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    depth_km: float = Field(ge=0, le=800)
    date: date


class TsunamiIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    max_height_m: float = Field(ge=0, le=600)


class FloodIn(BaseModel):
    cumulative_rainfall_mm: float = Field(ge=0, le=5000)


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


@router.post("/earthquake/predict", name="api_earthquake")
def api_earthquake(body: EarthquakeIn) -> dict:
    return asdict(eq.predict(body.lat, body.lon, body.depth_km, body.date))


@router.post("/tsunami/predict", name="api_tsunami")
def api_tsunami(body: TsunamiIn) -> dict:
    return asdict(tsunami.predict(body.lat, body.lon, body.max_height_m))


@router.post("/flood/predict", name="api_flood")
def api_flood(body: FloodIn) -> dict:
    return asdict(flood.predict(body.cumulative_rainfall_mm))


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
