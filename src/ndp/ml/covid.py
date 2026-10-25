"""COVID-19 (India) module: cumulative-case forecast and 2020 map snapshots.

Trained only with ``ndp-train --include-covid`` (or ``NDP_ENABLE_COVID=true``).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ndp.ml import plots, registry
from ndp.ml.features import honest_metadata, load_csv, regression_metrics, single_row

MODEL = "covid_india_cases"
EPOCH = dt.date(2020, 1, 22)
JHU_CSV = "time_series_covid19_confirmed_global.csv"
CENTRES_CSV = "TestCentres_with_geolocation.csv"
STATE_CSV = "state.csv"
CITY_CSV = "main.csv"
COUNTRY = "India"
FEATURES = ["day_index"]
TARGET = "confirmed"
FORECAST_DAYS = 90


@dataclass(frozen=True)
class CovidPrediction:
    day_index: int
    cases: int
    last_data_date: dt.date


def load_series() -> pd.Series:
    """Cumulative confirmed cases for India indexed by date (all provinces summed)."""
    raw = load_csv(JHU_CSV)
    india = raw[raw["Country/Region"] == COUNTRY]
    if india.empty:
        raise ValueError(f"no rows for {COUNTRY!r} in {JHU_CSV}")
    date_cols = raw.columns[4:]
    totals = india[date_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=0)
    dates = pd.to_datetime(date_cols, format="%m/%d/%y")
    return pd.Series(totals.to_numpy(dtype=float), index=dates, name=TARGET)


def _build_pipeline() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=3000, random_state=0),
    )


def train() -> dict:
    series = load_series()
    days = np.array([(d.date() - EPOCH).days for d in series.index], dtype=float)
    X = pd.DataFrame({"day_index": days})
    y = series.to_numpy()
    # Honest check first: fit on the first 80% of days, score on the most recent 20%.
    cut = int(len(y) * 0.8)
    holdout_model = _build_pipeline().fit(X.iloc[:cut], y[:cut])
    holdout_pred = holdout_model.predict(X.iloc[cut:])
    split = (
        f"chronological, last 20% of days held out ({series.index[cut].date()} to "
        f"{series.index[-1].date()}); the deployed model is then refit on all days"
    )
    honest = honest_metadata(y[:cut], y[cut:], holdout_pred, split)
    persistence = regression_metrics(y[cut:], np.full(len(y) - cut, y[cut - 1]))
    trend = np.polyfit(days[:cut], y[:cut], 1)
    linear = regression_metrics(y[cut:], np.polyval(trend, days[cut:]))

    model = _build_pipeline()
    model.fit(X, y)
    fitted = model.predict(X)
    r2_in_sample = float(r2_score(y, fitted))
    last_date = series.index[-1].date()

    plt.figure(figsize=(8, 4))
    plt.plot(series.index, y, label="confirmed (JHU)", color="#2563eb")
    plt.plot(series.index, fitted, label="model fit", color="#dc2626", lw=1)
    plt.title(f"{COUNTRY}: cumulative confirmed cases")
    plt.ylabel("Cases")
    plt.legend()
    plots.save_current("covid", "actual.png")

    future_days = np.arange(days[-1] + 1, days[-1] + 1 + FORECAST_DAYS)
    future_dates = [EPOCH + dt.timedelta(days=int(d)) for d in future_days]
    forecast = np.clip(model.predict(pd.DataFrame({"day_index": future_days})), 0, None)
    plt.figure(figsize=(8, 4))
    plt.plot(series.index, y, label="observed", color="#2563eb")
    plt.plot(future_dates, forecast, label=f"forecast (+{FORECAST_DAYS} days)", color="#dc2626")
    plt.title(f"{COUNTRY}: {FORECAST_DAYS}-day forecast")
    plt.ylabel("Cases")
    plt.legend()
    plots.save_current("covid", "forecast.png")

    metadata = {
        "last_date": last_date.isoformat(),
        "n_days": int(len(series)),
        **honest,
        "r2_in_sample": r2_in_sample,
        "last_value_rmse": persistence["rmse"],
        "linear_trend_rmse": linear["rmse"],
        "note": (
            "r2 and rmse are on the held-out final 20% of days; r2_in_sample is a curve fit. "
            "Cumulative counts trend upward, so the mean baseline is easy to beat; compare "
            "with last_value_rmse and linear_trend_rmse. Far-ahead forecasts are extrapolation."
        ),
        "n_rows": int(len(series)),
        "features": FEATURES,
        "target": TARGET,
        "epoch": EPOCH.isoformat(),
    }
    registry.save(MODEL, {"model": model, "last_date": last_date}, metadata)
    return metadata


def predict(on: dt.date) -> CovidPrediction:
    artifact = registry.load(MODEL)
    day_index = (on - EPOCH).days
    raw = float(artifact["model"].predict(single_row(FEATURES, [day_index]))[0])
    cases = max(0, int(round(raw)))
    return CovidPrediction(day_index=day_index, cases=cases, last_data_date=artifact["last_date"])


def _finite(*values: float) -> bool:
    return all(isinstance(v, int | float) and math.isfinite(v) for v in values)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def test_centres() -> list[dict]:
    df = load_csv(CENTRES_CSV)
    out: list[dict] = []
    for row in df.itertuples(index=False):
        lat, lon = _float(row.Latitude), _float(row.Longitude)
        if not _finite(lat, lon):
            continue
        out.append(
            {
                "name": str(row.LabName).strip(),
                "address": str(row.Addresses).strip(),
                "city": str(row.City).strip(),
                "lat": lat,
                "lon": lon,
            }
        )
    return out


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def nearest_centres(lat: float, lon: float, n: int = 5) -> list[dict]:
    """ICMR test centres on file, closest first, each with a `distance_km` field."""
    ranked = [
        {**c, "distance_km": _haversine_km(lat, lon, c["lat"], c["lon"])} for c in test_centres()
    ]
    ranked.sort(key=lambda c: c["distance_km"])
    return ranked[:n]


def state_snapshot() -> list[dict]:
    df = load_csv(STATE_CSV)
    out: list[dict] = []
    for row in df.itertuples(index=False):
        lat, lon = _float(row.Latitude), _float(row.Longitude)
        if not _finite(lat, lon):
            continue
        out.append(
            {
                "state": str(row.State).strip(),
                "lat": lat,
                "lon": lon,
                "active": _int(row.Active),
                "confirmed": _int(row.Confirm),
                "deaths": _int(row.Death),
                "recovered": _int(row.Recovered),
            }
        )
    return out


def city_snapshot() -> list[dict]:
    df = load_csv(CITY_CSV)
    out: list[dict] = []
    for row in df.itertuples(index=False):
        lat, lon = _float(row.Latitude), _float(row.Longitude)
        if not _finite(lat, lon):
            continue
        out.append(
            {
                "city": str(row.City).strip(),
                "lat": lat,
                "lon": lon,
                "active": _int(row.Active),
                "confirmed": _int(row.Confirm),
                "deaths": _int(row.Death),
            }
        )
    return out
