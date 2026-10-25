"""Live hazards page: GDACS alerts, Open-Meteo forecast, real-event watcher status."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/live")
