"""Hurricane maximum sustained wind regression (NOAA HURDAT2 Atlantic + Pacific)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ndp.ml import plots, registry
from ndp.ml.features import (
    date_to_int,
    load_csv,
    parse_hurdat_coord,
    regression_metrics,
    single_row,
)

MODEL = "hurricane_wind"
CSVS = ("hurricane-atlantic.csv", "hurricane-pacific.csv")
FEATURES = ["Date", "Latitude", "Longitude"]
TARGET = "Maximum Wind"
HURRICANE_KT = 64

# Saffir-Simpson on sustained wind in knots: (upper bound exclusive, label)
_CATEGORIES = (
    (34, "Tropical depression"),
    (64, "Tropical storm"),
    (83, "Category 1"),
    (96, "Category 2"),
    (113, "Category 3"),
    (137, "Category 4"),
)


@dataclass(frozen=True)
class HurricanePrediction:
    max_wind_kt: float
    category: str
    is_hurricane: bool


def category_for(max_wind_kt: float) -> str:
    for upper, label in _CATEGORIES:
        if max_wind_kt < upper:
            return label
    return "Category 5"


def load_frame() -> pd.DataFrame:
    """Both basins, coordinates parsed to signed floats, -99 sentinel winds dropped."""
    frames = [load_csv(name) for name in CSVS]
    df = pd.concat(frames, ignore_index=True)[FEATURES + [TARGET]].copy()
    df["Latitude"] = df["Latitude"].map(parse_hurdat_coord)
    df["Longitude"] = df["Longitude"].map(parse_hurdat_coord)
    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET] >= 0]
    df["Date"] = df["Date"].astype(int)
    return df.reset_index(drop=True)


def _build_pipeline() -> Pipeline:
    return make_pipeline(StandardScaler(), LinearRegression())


def _plots(df: pd.DataFrame, y_test: pd.Series, y_pred) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, y_test, s=4, alpha=0.3)
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], color="#dc2626")
    plt.title("Predicted vs actual maximum wind (test set)")
    plt.xlabel("Predicted (kt)")
    plt.ylabel("Actual (kt)")
    plots.save_current("hurricane", "h1.png")

    when = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    sample = df.assign(when=when).dropna(subset=["when"])
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=0)
    plt.figure(figsize=(9, 4))
    plt.scatter(sample["when"], sample[TARGET], s=2, alpha=0.3)
    plt.title("Maximum wind over time")
    plt.xlabel("Date")
    plt.ylabel("Maximum wind (kt)")
    plots.save_current("hurricane", "h2.png")

    plt.figure(figsize=(6, 4))
    plt.hist(df[TARGET], bins=range(0, int(df[TARGET].max()) + 10, 5), color="#2563eb")
    plt.title("Maximum wind distribution")
    plt.xlabel("Maximum wind (kt)")
    plt.ylabel("Observations")
    plots.save_current("hurricane", "h3.png")


def train() -> dict:
    df = load_frame()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    model = _build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    _plots(df, y_test, y_pred)
    metadata = {
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "n_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
    }
    registry.save(MODEL, model, metadata)
    return metadata


def predict(on: dt.date, lat: float, lon: float) -> HurricanePrediction:
    model = registry.load(MODEL)
    row = single_row(FEATURES, [date_to_int(on), lat, lon])
    wind = float(model.predict(row)[0])
    return HurricanePrediction(
        max_wind_kt=wind, category=category_for(wind), is_hurricane=wind >= HURRICANE_KT
    )
