"""Live COVID-19 totals from disease.sh (free, no key). Cached for ten minutes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from ndp.services import ServiceError

BASE = "https://disease.sh/v3/covid-19"
CACHE_SECONDS = 600


@dataclass(frozen=True)
class Totals:
    label: str
    cases: int
    today_cases: int
    deaths: int
    today_deaths: int
    recovered: int
    active: int
    tests: int | None
    updated: datetime


_cache: dict[str, tuple[float, Totals]] = {}


def _fetch(path: str, label: str) -> Totals:
    now = time.monotonic()
    hit = _cache.get(path)
    if hit and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{BASE}/{path}")
            resp.raise_for_status()
            d = resp.json()
        totals = Totals(
            label=label,
            cases=int(d["cases"]),
            today_cases=int(d.get("todayCases", 0)),
            deaths=int(d["deaths"]),
            today_deaths=int(d.get("todayDeaths", 0)),
            recovered=int(d.get("recovered", 0)),
            active=int(d.get("active", 0)),
            tests=int(d["tests"]) if d.get("tests") else None,
            updated=datetime.fromtimestamp(d["updated"] / 1000, tz=UTC),
        )
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        raise ServiceError(f"Live COVID-19 statistics are unavailable right now ({exc}).") from exc
    _cache[path] = (now, totals)
    return totals


def world() -> Totals:
    return _fetch("all", "World")


def india() -> Totals:
    return _fetch("countries/india", "India")
