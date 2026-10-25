"""Short-range weather outlook from Open-Meteo (no key needed) with rule-of-thumb advisories.

Everything derived here is a *heuristic computed from model forecast data*. It is not an
official warning and not a forecast issued by any national weather service. The rainfall
classes are the India Meteorological Department's 24-hour classes; the gust level is the lower
bound of gale force (about Beaufort 8).

Endpoint (verified against the live service): GET https://api.open-meteo.com/v1/forecast with
`hourly=precipitation,wind_gusts_10m,temperature_2m`, `timezone=auto`, `forecast_days=N`. Hourly
times are local to the location and the day starts at 00:00, so the window is cut to start at
the current local hour. Errors come back as HTTP 400 with `{"error": true, "reason": "..."}`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import httpx

from ndp.services import ServiceError
from ndp.services.geo import USER_AGENT

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "precipitation,wind_gusts_10m,temperature_2m"
CACHE_SECONDS = 15 * 60
CACHE_MAX = 128
TIMEOUT = 12.0
MAX_DAYS = 7

# India Meteorological Department 24-hour rainfall classes (mm). Rainfall is reported to 0.1 mm,
# so the lower bound of each class is the figure after rounding to one decimal.
HEAVY_MM = 64.5  # heavy 64.5-115.5
VERY_HEAVY_MM = 115.6  # very heavy 115.6-204.4
EXTREMELY_HEAVY_MM = 204.5  # extremely heavy >= 204.5
GUST_KMH = 63.0

DISCLAIMER = (
    "Rule-of-thumb advisories computed from Open-Meteo model forecast data using India "
    "Meteorological Department rainfall classes. This is not an official warning; follow your "
    "national weather service and local authorities."
)


@dataclass(frozen=True)
class ForecastHour:
    time: datetime  # local time at the location, naive
    precip_mm: float | None
    gust_kmh: float | None
    temp_c: float | None


@dataclass(frozen=True)
class Advisory:
    level: str  # "info" | "watch" | "warning"
    text: str


@dataclass(frozen=True)
class DailySummary:
    day: date
    precip_mm: float
    max_gust_kmh: float | None
    temp_min_c: float | None
    temp_max_c: float | None


@dataclass(frozen=True)
class Forecast:
    lat: float
    lon: float
    timezone: str
    days: int
    hours: list[ForecastHour]
    precip_24h_max_mm: float
    precip_72h_total_mm: float
    max_gust_kmh: float | None
    temp_min_c: float | None
    temp_max_c: float | None
    advisories: list[Advisory]
    daily: list[DailySummary] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


_clock = time.monotonic
_cache: dict[tuple[float, float, int], tuple[float, Forecast]] = {}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value else None  # NaN guard


def rolling_max(values: list[float | None], width: int = 24) -> float:
    """Largest sum over any `width` consecutive hours (the whole series if it is shorter)."""
    vals = [v or 0.0 for v in values]
    if not vals:
        return 0.0
    if len(vals) <= width:
        return round(sum(vals), 1)
    window = sum(vals[:width])
    best = window
    for i in range(width, len(vals)):
        window += vals[i] - vals[i - width]
        best = max(best, window)
    return round(best, 1)


def assess(precip_24h_max_mm: float, max_gust_kmh: float | None, hours: int) -> list[Advisory]:
    """Advisories from the two derived figures. Pure function, so the thresholds are testable."""
    rain = round(precip_24h_max_mm, 1)
    gust = round(max_gust_kmh, 1) if max_gust_kmh is not None else None
    out: list[Advisory] = []
    if rain >= EXTREMELY_HEAVY_MM:
        out.append(
            Advisory(
                "warning",
                f"Extremely heavy rainfall class (204.5 mm or more in 24 hours): the model shows "
                f"up to {rain:g} mm in its wettest 24 hours.",
            )
        )
    elif rain >= VERY_HEAVY_MM:
        out.append(
            Advisory(
                "warning",
                f"Very heavy rainfall class (115.6 to 204.4 mm in 24 hours): the model shows up "
                f"to {rain:g} mm in its wettest 24 hours.",
            )
        )
    elif rain >= HEAVY_MM:
        out.append(
            Advisory(
                "watch",
                f"Heavy rainfall class (64.5 to 115.5 mm in 24 hours): the model shows up to "
                f"{rain:g} mm in its wettest 24 hours.",
            )
        )
    if gust is not None and gust >= GUST_KMH:
        out.append(
            Advisory(
                "watch",
                f"Gale-strength gusts (63 km/h or more): the model shows gusts up to "
                f"{gust:g} km/h.",
            )
        )
    if not out:
        out.append(
            Advisory(
                "info",
                f"No heavy rainfall (64.5 mm or more in 24 hours) and no gale-strength gusts "
                f"(63 km/h or more) in the model forecast for the next {hours} hours.",
            )
        )
    return out


def _daily(hours: list[ForecastHour]) -> list[DailySummary]:
    by_day: dict[date, list[ForecastHour]] = {}
    for h in hours:
        by_day.setdefault(h.time.date(), []).append(h)
    out = []
    for day, items in by_day.items():
        gusts = [h.gust_kmh for h in items if h.gust_kmh is not None]
        temps = [h.temp_c for h in items if h.temp_c is not None]
        out.append(
            DailySummary(
                day=day,
                precip_mm=round(sum(h.precip_mm or 0.0 for h in items), 1),
                max_gust_kmh=max(gusts) if gusts else None,
                temp_min_c=min(temps) if temps else None,
                temp_max_c=max(temps) if temps else None,
            )
        )
    return out


def parse_forecast(
    payload: dict, lat: float, lon: float, days: int, now: datetime | None = None
) -> Forecast:
    """Build a Forecast from an Open-Meteo response. Raises ValueError for unusable payloads."""
    if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
        raise ValueError("response has no hourly data")
    hourly = payload["hourly"]
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise ValueError("response has no hourly times")

    def column(name: str) -> list:
        col = hourly.get(name)
        return col if isinstance(col, list) else []

    precip, gusts, temps = (
        column("precipitation"),
        column("wind_gusts_10m"),
        column("temperature_2m"),
    )
    offset = _num(payload.get("utc_offset_seconds")) or 0.0
    local_now = ((now or _utcnow()).astimezone(UTC) + timedelta(seconds=offset)).replace(
        tzinfo=None
    )
    start_hour = local_now.replace(minute=0, second=0, microsecond=0)

    rows: list[ForecastHour] = []
    for i, raw in enumerate(times):
        try:
            when = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        if when.tzinfo is not None:
            when = when.replace(tzinfo=None)
        if when < start_hour:
            continue
        rows.append(
            ForecastHour(
                time=when,
                precip_mm=_num(precip[i]) if i < len(precip) else None,
                gust_kmh=_num(gusts[i]) if i < len(gusts) else None,
                temp_c=_num(temps[i]) if i < len(temps) else None,
            )
        )
    rows = rows[: days * 24]
    if not rows:
        raise ValueError("no forecast hours at or after the current time")

    gust_values = [h.gust_kmh for h in rows if h.gust_kmh is not None]
    temp_values = [h.temp_c for h in rows if h.temp_c is not None]
    max_gust = max(gust_values) if gust_values else None
    p24 = rolling_max([h.precip_mm for h in rows], 24)
    return Forecast(
        lat=float(payload.get("latitude", lat)),
        lon=float(payload.get("longitude", lon)),
        timezone=str(payload.get("timezone") or "local time"),
        days=days,
        hours=rows,
        precip_24h_max_mm=p24,
        precip_72h_total_mm=round(sum(h.precip_mm or 0.0 for h in rows[:72]), 1),
        max_gust_kmh=max_gust,
        temp_min_c=min(temp_values) if temp_values else None,
        temp_max_c=max(temp_values) if temp_values else None,
        advisories=assess(p24, max_gust, len(rows)),
        daily=_daily(rows),
    )


def forecast(lat: float, lon: float, days: int = 3, now: datetime | None = None) -> Forecast:
    """Hourly outlook for the next `days` days (1 to 7) from the current local hour. Cached for 15
    minutes per location (rounded to 0.01 degree). `now` exists for tests."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ServiceError("Latitude must be within -90 to 90 and longitude within -180 to 180.")
    days = max(1, min(int(days), MAX_DAYS))
    key = (round(lat, 2), round(lon, 2), days)
    hit = _cache.get(key)
    if hit and _clock() - hit[0] <= CACHE_SECONDS:
        return hit[1]
    try:
        with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(
                FORECAST_URL,
                params={
                    "latitude": f"{lat:.4f}",
                    "longitude": f"{lon:.4f}",
                    "hourly": HOURLY,
                    "timezone": "auto",
                    # The first day starts at 00:00, so ask for one extra day to still have a full
                    # window after cutting to the current hour.
                    "forecast_days": days + 1,
                },
            )
            if resp.status_code == 400:
                try:
                    reason = resp.json().get("reason", "bad request")
                except ValueError:
                    reason = "bad request"
                raise ServiceError(f"Open-Meteo rejected the request ({reason}).")
            resp.raise_for_status()
            payload = resp.json()
        result = parse_forecast(payload, lat, lon, days, now)
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError(f"The Open-Meteo forecast is unavailable right now ({exc}).") from exc
    if len(_cache) >= CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (_clock(), result)
    return result


def clear_cache() -> None:
    _cache.clear()
