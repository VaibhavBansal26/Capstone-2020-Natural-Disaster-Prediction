"""Earthquake hazard from the historical catalogue (replaces the magnitude regression).

This is statistics, not machine learning that forecasts: given a point and a radius it counts
the M >= 5.5 earthquakes the USGS catalogue (1965-2016) recorded inside that circle, turns the
count into an annual rate, extends it to larger magnitudes with the Gutenberg-Richter law and
converts rates to probabilities with a Poisson model.

* b-value: Aki (1965) maximum likelihood with Utsu's correction for magnitudes rounded to
  ``dm`` = 0.1, ``b = log10(e) / (mean(M) - (Mc - dm / 2))``, used when the circle holds at
  least 50 events; otherwise the global b-value is used (``b_is_global``).
* rate for ``M >= m``: ``annual_rate_m55 * 10 ** (-b * (m - 5.5))``.
* probability of at least one such event in ``T`` years: ``1 - exp(-rate * T)``.

What it is not: it says nothing about *when* an earthquake will happen, it assumes the rate in
the 52-year catalogue is the long-run rate, it ignores clustering and aftershocks, and the
catalogue is incomplete for old and remote events. Regions with few events get noisy rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ndp.ml import plots, registry
from ndp.ml.features import load_csv

MODEL = "earthquake_hazard"
CSV = "database.csv"
MC = 5.5
MAGNITUDE_STEP = 0.1
MIN_EVENTS_LOCAL_B = 50
MIN_RADIUS_KM = 25.0
MAX_RADIUS_KM = 2000.0
DEFAULT_RADIUS_KM = 300.0
EARTH_RADIUS_KM = 6371.0088
THRESHOLDS = (6.0, 7.0, 8.0)
LARGE_EVENT_MAGNITUDE = 6.0
_SECONDS_PER_YEAR = 365.25 * 86400.0
_EPOCH = np.datetime64("1970-01-01T00:00:00")


@dataclass(frozen=True)
class HazardEstimate:
    lat: float
    lon: float
    radius_km: float
    n_events: int
    years: float
    annual_rate_m55: float
    b_value: float
    b_is_global: bool
    annual_rate: dict[str, float]
    prob_1y: dict[str, float]
    prob_10y: dict[str, float]
    max_magnitude: float | None
    max_date: str | None
    depth_median_km: float | None
    nearest_event: dict | None
    level: str
    elevated: bool


# --------------------------------------------------------------------------- maths


def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to arrays of points (vectorised).

    The longitude difference is wrapped into [-180, 180) so pairs on either side of the
    antimeridian (e.g. 179.5 and -179.5) are 1 degree apart, not 359; near the poles the
    formula degrades gracefully because it only uses sines and cosines of the latitudes.
    """
    phi0 = math.radians(lat0)
    phi = np.radians(np.asarray(lats, dtype=float))
    dlon = np.radians((np.asarray(lons, dtype=float) - lon0 + 180.0) % 360.0 - 180.0)
    a = np.sin((phi - phi0) / 2.0) ** 2 + math.cos(phi0) * np.cos(phi) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def aki_b_value(magnitudes: np.ndarray, mc: float = MC, dm: float = MAGNITUDE_STEP) -> float:
    """Aki-Utsu maximum-likelihood Gutenberg-Richter b-value for events with ``M >= mc``."""
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[mags >= mc - 1e-9]
    if mags.size < 2:
        raise ValueError("need at least two events at or above the completeness magnitude")
    excess = float(mags.mean()) - (mc - dm / 2.0)
    if excess <= 0:
        raise ValueError("mean magnitude is not above the completeness cut-off")
    return math.log10(math.e) / excess


def poisson_probability(annual_rate: float, years: float) -> float:
    """P(at least one event in ``years``) for a Poisson process; exact for tiny rates."""
    return float(-math.expm1(-max(annual_rate, 0.0) * years))


def gutenberg_richter_rate(annual_rate_m55: float, b: float, magnitude: float) -> float:
    return float(annual_rate_m55 * 10.0 ** (-b * (magnitude - MC)))


def level_for(prob_10y_m6: float) -> str:
    if prob_10y_m6 < 0.05:
        return "low"
    if prob_10y_m6 < 0.25:
        return "moderate"
    if prob_10y_m6 < 0.60:
        return "high"
    return "very high"


def normalise_lon(lon: float) -> float:
    return float((lon + 180.0) % 360.0 - 180.0)


def clamp_radius(radius_km: float) -> float:
    return float(min(max(float(radius_km), MIN_RADIUS_KM), MAX_RADIUS_KM))


def _epoch_to_date(epoch: float) -> str:
    return str((_EPOCH + np.timedelta64(int(epoch), "s")).astype("datetime64[D]"))


def _epoch_to_year(epoch: float) -> int:
    return int(str(_EPOCH + np.timedelta64(int(epoch), "s"))[:4])


def estimate_from_catalogue(
    catalogue: dict[str, np.ndarray],
    *,
    years: float,
    b_global: float,
    lat: float,
    lon: float,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> HazardEstimate:
    """The hazard computation on an in-memory catalogue (arrays ``lat lon depth mag epoch``)."""
    if not (math.isfinite(lat) and math.isfinite(lon)) or not -90.0 <= lat <= 90.0:
        raise ValueError("lat must be within -90..90 and lon must be finite")
    lon = normalise_lon(lon)
    radius = clamp_radius(radius_km)

    dist = haversine_km(lat, lon, catalogue["lat"], catalogue["lon"])
    inside = (dist <= radius) & (catalogue["mag"] >= MC - 1e-9)
    mags = catalogue["mag"][inside]
    n_events = int(mags.size)

    annual_rate_m55 = n_events / years if years > 0 else 0.0
    if n_events >= MIN_EVENTS_LOCAL_B:
        b_value, b_is_global = aki_b_value(mags), False
    else:
        b_value, b_is_global = float(b_global), True

    annual_rate = {
        f"{m:.1f}": gutenberg_richter_rate(annual_rate_m55, b_value, m) for m in THRESHOLDS
    }
    prob_1y = {k: poisson_probability(v, 1.0) for k, v in annual_rate.items()}
    prob_10y = {k: poisson_probability(v, 10.0) for k, v in annual_rate.items()}
    level = level_for(prob_10y["6.0"])

    max_magnitude = max_date = depth_median = nearest = None
    if n_events:
        epochs = catalogue["epoch"][inside]
        top = int(np.argmax(mags))
        max_magnitude = float(mags[top])
        max_date = _epoch_to_date(float(epochs[top]))
        depth_median = float(np.median(catalogue["depth"][inside]))
        large = np.flatnonzero(mags >= LARGE_EVENT_MAGNITUDE)
        if large.size:
            d_inside = dist[inside]
            closest = int(large[np.argmin(d_inside[large])])
            nearest = {
                "distance_km": float(d_inside[closest]),
                "magnitude": float(mags[closest]),
                "year": _epoch_to_year(float(epochs[closest])),
            }

    return HazardEstimate(
        lat=float(lat),
        lon=lon,
        radius_km=radius,
        n_events=n_events,
        years=float(years),
        annual_rate_m55=float(annual_rate_m55),
        b_value=float(b_value),
        b_is_global=b_is_global,
        annual_rate=annual_rate,
        prob_1y=prob_1y,
        prob_10y=prob_10y,
        max_magnitude=max_magnitude,
        max_date=max_date,
        depth_median_km=depth_median,
        nearest_event=nearest,
        level=level,
        elevated=level in ("high", "very high"),
    )


# --------------------------------------------------------------------------- training


def load_catalogue() -> dict[str, np.ndarray]:
    """Real earthquakes (not nuclear tests or explosions) of the USGS catalogue as arrays."""
    from ndp.ml.features import earthquake_date_to_epoch

    df = load_csv(CSV)
    df = df[df["Type"] == "Earthquake"]
    df = df.dropna(subset=["Date", "Latitude", "Longitude", "Depth", "Magnitude"])
    df = df[df["Magnitude"] >= MC - 1e-9]
    epoch = np.array([earthquake_date_to_epoch(d) for d in df["Date"]], dtype=np.float64)
    order = np.argsort(epoch, kind="stable")
    return {
        "lat": df["Latitude"].to_numpy(dtype=np.float64)[order],
        "lon": df["Longitude"].to_numpy(dtype=np.float64)[order],
        "depth": df["Depth"].to_numpy(dtype=np.float64)[order],
        "mag": df["Magnitude"].to_numpy(dtype=np.float64)[order],
        "epoch": epoch[order],
    }


def _year_fraction(epoch: np.ndarray) -> np.ndarray:
    return 1970.0 + epoch / _SECONDS_PER_YEAR


def _backtest(catalogue: dict[str, np.ndarray], split_year: int = 2000, cell_deg: int = 5) -> dict:
    """Do 5-degree cells that were seismically active before ``split_year`` capture the
    M >= 6 earthquakes after it, compared with how much of the globe they cover?"""
    year = _year_fraction(catalogue["epoch"])
    lat_idx = np.clip(((catalogue["lat"] + 90) // cell_deg).astype(int), 0, 180 // cell_deg - 1)
    lon_idx = np.clip(
        (((catalogue["lon"] + 180) % 360) // cell_deg).astype(int), 0, 360 // cell_deg - 1
    )
    cell = lat_idx * (360 // cell_deg) + lon_idx
    past = (year < split_year) & (catalogue["mag"] >= MC)
    future = (year >= split_year) & (catalogue["mag"] >= LARGE_EVENT_MAGNITUDE)
    active = np.unique(cell[past])

    n_lat, n_lon = 180 // cell_deg, 360 // cell_deg
    centres = -90 + cell_deg * (np.arange(n_lat) + 0.5)
    area_row = np.cos(np.radians(centres))
    active_lat = active // n_lon
    area_share = float(area_row[active_lat].sum() / (area_row.sum() * n_lon))
    captured = float(np.isin(cell[future], active).mean()) if future.any() else float("nan")
    return {
        "train_period": f"before {split_year}",
        "test_period": f"{split_year} onwards",
        "test_events_m6": int(future.sum()),
        "cells_active_before": int(active.size),
        "share_of_globe_in_active_cells": area_share,
        "share_of_test_events_in_active_cells": captured,
        "lift": captured / area_share if area_share else None,
    }


def _plots(catalogue: dict[str, np.ndarray], b_global: float, years: float) -> None:
    plt.figure(figsize=(9, 4.6))
    order = np.argsort(catalogue["mag"])
    scatter = plt.scatter(
        catalogue["lon"][order],
        catalogue["lat"][order],
        c=catalogue["mag"][order],
        s=np.where(catalogue["mag"][order] >= 7, 18, 3),
        cmap="plasma_r",
        alpha=0.7,
        linewidths=0,
    )
    plt.colorbar(scatter, label="Magnitude")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Earthquake epicentres, M >= 5.5, USGS catalogue ({len(catalogue['mag']):,} events)")
    plots.save_current("earthquake", "hz1_seismicity.png")

    mags = catalogue["mag"].astype(float)
    grid = np.round(np.arange(MC, float(mags.max()) + 1e-9, MAGNITUDE_STEP), 1)
    counts = np.array([(mags >= m - 1e-9).sum() for m in grid])
    shown = counts > 0
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(
        grid[shown], np.log10(counts[shown]), s=14, color="#2563eb", label="Observed N(>= M)"
    )
    fit_x = np.array([MC, float(grid[shown].max())])
    fit_y = math.log10(len(mags)) - b_global * (fit_x - MC)
    plt.plot(
        fit_x, fit_y, color="#dc2626", lw=1.5, label=f"Gutenberg-Richter fit, b = {b_global:.2f}"
    )
    plt.xlabel("Magnitude")
    plt.ylabel("log10 of number of events with magnitude >= M")
    plt.title(f"Gutenberg-Richter law, {years:.0f} years of catalogue")
    plt.legend()
    plots.save_current("earthquake", "hz2_gutenberg_richter.png")


def train() -> dict:
    catalogue = load_catalogue()
    year = _year_fraction(catalogue["epoch"])
    year_min, year_max = int(np.floor(year.min())), int(np.floor(year.max()))
    years_span = float(year_max - year_min + 1)  # the catalogue covers whole calendar years
    b_global = aki_b_value(catalogue["mag"])
    _plots(catalogue, b_global, years_span)
    metadata = {
        "n_events": int(catalogue["mag"].size),
        "years_span": years_span,
        "mc": MC,
        "b_value_global": b_global,
        "year_min": year_min,
        "year_max": year_max,
        "annual_rate_global_m55": float(catalogue["mag"].size / years_span),
        "method": "Aki-Utsu b-value, Gutenberg-Richter extrapolation, Poisson probability",
        "split": "none (statistical rates; see backtest for a before/after-2000 check)",
        "backtest": _backtest(catalogue),
        "min_events_local_b": MIN_EVENTS_LOCAL_B,
        "target": "annual rate of M >= 5.5 within a radius",
    }
    registry.save(MODEL, catalogue, metadata)
    return metadata


# --------------------------------------------------------------------------- prediction


def predict(lat: float, lon: float, radius_km: float = DEFAULT_RADIUS_KM) -> HazardEstimate:
    catalogue = registry.load(MODEL)
    meta = registry.metadata(MODEL) or {}
    years = float(meta.get("years_span") or 0.0)
    if years <= 0:
        year = _year_fraction(catalogue["epoch"])
        years = float(np.floor(year.max()) - np.floor(year.min()) + 1)
    b_global = float(meta.get("b_value_global") or aki_b_value(catalogue["mag"]))
    return estimate_from_catalogue(
        catalogue, years=years, b_global=b_global, lat=lat, lon=lon, radius_km=radius_km
    )
