"""Hurricane maximum sustained wind regression (NOAA HURDAT2 Atlantic + Pacific).

Features are position and time of year only (``Latitude``, ``Longitude``, day of year and
``abs(Latitude)``); the year is deliberately not a feature. The split is chronological:
trained on observations before 2005, evaluated on 2005 onwards. A gradient-boosted model of
position alone cannot know how strong a particular storm is, so expect modest skill.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ndp.ml import plots, registry
from ndp.ml.features import (
    honest_metadata,
    load_csv,
    parse_hurdat_coord,
)

MODEL = "hurricane_wind"
CSVS = ("hurricane-atlantic.csv", "hurricane-pacific.csv")
FEATURES = ["Date", "Latitude", "Longitude"]  # raw columns (also used by ``compare``)
MODEL_FEATURES = ["lat", "lon", "day_of_year", "abs_lat"]
SPLIT_YEAR = 2005
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


def model_frame(lat, lon, day_of_year) -> pd.DataFrame:
    """The model's feature matrix; accepts array-likes."""
    frame = pd.DataFrame({"lat": lat, "lon": lon, "day_of_year": day_of_year})
    frame["abs_lat"] = frame["lat"].abs()
    return frame[MODEL_FEATURES].astype(float)


def _day_of_year(dates: pd.Series) -> pd.Series:
    when = pd.to_datetime(dates.astype(str), format="%Y%m%d", errors="coerce")
    return when.dt.day_of_year


def _build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0
    )


def _plots(df: pd.DataFrame, y_test: pd.Series, y_pred) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, y_test, s=4, alpha=0.3)
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], color="#dc2626")
    plt.title("Predicted vs actual maximum wind (2005 onwards, held out)")
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
    df = df.assign(day_of_year=_day_of_year(df["Date"])).dropna(subset=["day_of_year"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    year = df["Date"] // 10000
    train_mask = (year < SPLIT_YEAR).to_numpy()
    X = model_frame(df["Latitude"], df["Longitude"], df["day_of_year"])
    y = df[TARGET]
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    model = _build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    split = (
        f"chronological: trained on {int(year.min())} to {SPLIT_YEAR - 1}, "
        f"tested on {SPLIT_YEAR} to {int(year.max())}"
    )
    metrics = honest_metadata(y_train, y_test, y_pred, split)
    _plots(df, y_test, y_pred)
    metadata = {
        **metrics,
        "n_rows": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "features": MODEL_FEATURES,
        "target": TARGET,
        "split_year": SPLIT_YEAR,
    }
    registry.save(MODEL, model, metadata)
    return metadata


def predict(on: dt.date, lat: float, lon: float) -> HurricanePrediction:
    model = registry.load(MODEL)
    row = model_frame([lat], [lon], [on.timetuple().tm_yday])
    wind = float(model.predict(row)[0])
    return HurricanePrediction(
        max_wind_kt=wind, category=category_for(wind), is_hurricane=wind >= HURRICANE_KT
    )
