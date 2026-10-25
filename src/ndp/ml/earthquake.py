"""Earthquake magnitude regression (USGS significant earthquakes 1965-2016).

Illustrative only, not used by the earthquake page (that page uses :mod:`ndp.ml.hazard`).
It is kept because the model comparison and the tests use it, and because it is a useful
example of a model with almost no skill: the catalogue only holds M >= 5.5 events, location
and depth say very little about the size of the next one, and the held-out R^2 is about 0.
Its metadata records the baseline, the skill and the chronological split so the page can say so.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ndp.ml import plots, registry
from ndp.ml.features import earthquake_date_to_epoch, honest_metadata, load_csv, single_row

MODEL = "earthquake_magnitude"
STATS = "earthquake_stats"
CSV = "database.csv"
FEATURES = ["Latitude", "Longitude", "Depth", "date_epoch"]
TARGET = "Magnitude"
ALERT_MAGNITUDE = 6.0


@dataclass(frozen=True)
class EarthquakePrediction:
    magnitude: float
    severity: str
    alert: bool


def severity_for(magnitude: float) -> str:
    if magnitude < 4:
        return "minor"
    if magnitude < 5:
        return "light"
    if magnitude < 6:
        return "moderate"
    return "strong"


def load_frame() -> pd.DataFrame:
    df = load_csv(CSV)
    df = df.dropna(subset=["Date", "Latitude", "Longitude", "Depth", TARGET]).copy()
    df["date_epoch"] = df["Date"].map(earthquake_date_to_epoch)
    df["datetime"] = pd.to_datetime(df["date_epoch"], unit="s")
    return df.sort_values("date_epoch", kind="stable").reset_index(drop=True)


def _build_pipeline() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=500, random_state=0),
    )


def _compute_stats(df: pd.DataFrame) -> dict:
    years = df["datetime"].dt.year
    months = df["datetime"].dt.month
    by_year = {int(y): int(n) for y, n in years.value_counts().sort_index().items()}
    by_month = {m: int((months == m).sum()) for m in range(1, 13)}
    busiest_year, busiest_count = max(by_year.items(), key=lambda kv: kv[1])
    bins: dict[str, int] = {}
    for lo in range(5, 10):
        hi = lo + 1
        mask = (df[TARGET] >= lo) & (df[TARGET] < hi) if hi < 10 else (df[TARGET] >= lo)
        bins[f"{lo}-{hi}"] = int(mask.sum())
    return {
        "min_mag": float(df[TARGET].min()),
        "max_mag": float(df[TARGET].max()),
        "mean_mag": float(df[TARGET].mean()),
        "n_events": int(len(df)),
        "busiest_year": int(busiest_year),
        "busiest_year_count": int(busiest_count),
        "by_year": by_year,
        "by_month": by_month,
        "magnitude_bins": bins,
    }


def _plots(df: pd.DataFrame, stats: dict) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(df[TARGET], bins=np.arange(5.5, 9.6, 0.1), color="#2563eb")
    plt.title("Magnitude distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Events")
    plots.save_current("earthquake", "e1.png")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(7, 4))
    plt.bar(months, [stats["by_month"][m] for m in range(1, 13)], color="#2563eb")
    plt.title("Events by month")
    plt.ylabel("Events")
    plots.save_current("earthquake", "e2.png")

    plt.figure(figsize=(9, 4))
    years = sorted(stats["by_year"])
    plt.bar(years, [stats["by_year"][y] for y in years], color="#2563eb")
    plt.title("Events by year")
    plt.xlabel("Year")
    plt.ylabel("Events")
    plots.save_current("earthquake", "e3.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(df["Depth"], df[TARGET], s=3, alpha=0.3)
    plt.title("Magnitude vs depth")
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plots.save_current("earthquake", "e4.png")


def train() -> dict:
    df = load_frame()
    X = df[FEATURES]
    y = df[TARGET]
    cut = int(len(df) * 0.8)  # time-ordered data: hold out the most recent 20%
    X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    model = _build_pipeline()
    model.fit(X_train, y_train)
    split = (
        "chronological, last 20% held out "
        f"({df['datetime'].iloc[cut]:%Y-%m-%d} to {df['datetime'].iloc[-1]:%Y-%m-%d})"
    )
    metrics = honest_metadata(y_train, y_test, model.predict(X_test), split)

    stats = _compute_stats(df)
    _plots(df, stats)

    metadata = {
        **metrics,
        "illustrative": True,
        "note": "Illustrative only: the earthquake page uses earthquake_hazard instead.",
        "n_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "n_iter": int(model[-1].n_iter_),
    }
    registry.save(MODEL, model, metadata)
    registry.save(STATS, stats, {"n_events": stats["n_events"]})
    return metadata


def predict(lat: float, lon: float, depth_km: float, on: dt.date) -> EarthquakePrediction:
    model = registry.load(MODEL)
    epoch = (dt.datetime(on.year, on.month, on.day) - dt.datetime(1970, 1, 1)).total_seconds()
    row = single_row(FEATURES, [lat, lon, depth_km, epoch])
    magnitude = float(model.predict(row)[0])
    return EarthquakePrediction(
        magnitude=magnitude,
        severity=severity_for(magnitude),
        alert=magnitude >= ALERT_MAGNITUDE,
    )


def stats() -> dict:
    return registry.load(STATS)
