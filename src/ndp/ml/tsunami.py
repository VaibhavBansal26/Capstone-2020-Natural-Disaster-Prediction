"""Tsunami source-magnitude regression (NOAA NCEI historical tsunami database)."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ndp.ml import plots, registry
from ndp.ml.features import load_csv, regression_metrics, single_row

MODEL = "tsunami_magnitude"
CSV = "tsunami.csv"
FEATURES = ["LATITUDE", "LONGITUDE", "MAXIMUM_HEIGHT"]
TARGET = "PRIMARY_MAGNITUDE"
RISK_MAGNITUDE = 6.5


@dataclass(frozen=True)
class TsunamiPrediction:
    magnitude: float
    risk: bool


def load_frame() -> pd.DataFrame:
    """Features fillna(mean) as legacy did; rows with no recorded magnitude are dropped."""
    raw = load_csv(CSV)
    df = raw[FEATURES + [TARGET, "YEAR"]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    return df.reset_index(drop=True)


def _build_pipeline() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100,), max_iter=1500, random_state=0),
    )


def _plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[TARGET], df["MAXIMUM_HEIGHT"], s=8, alpha=0.5)
    plt.yscale("symlog")
    plt.title("Maximum water height vs source magnitude")
    plt.xlabel("Primary magnitude")
    plt.ylabel("Maximum height (m)")
    plots.save_current("tsunami", "t1.png")

    decades = (pd.to_numeric(df["YEAR"], errors="coerce").dropna() // 10 * 10).astype(int)
    decades = decades[decades >= 1800]
    counts = decades.value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(str), counts.values, color="#2563eb")
    plt.xticks(rotation=60)
    plt.title("Tsunami events per decade (since 1800)")
    plt.ylabel("Events")
    plots.save_current("tsunami", "t2.png")

    plt.figure(figsize=(8, 4))
    plt.scatter(df["LONGITUDE"], df["LATITUDE"], s=6, alpha=0.5, c=df[TARGET], cmap="viridis")
    plt.colorbar(label="Magnitude")
    plt.title("Tsunami source locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plots.save_current("tsunami", "t3.png")


def train() -> dict:
    df = load_frame()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = _build_pipeline()
    model.fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    _plots(df)
    metadata = {
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "n_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "n_iter": int(model[-1].n_iter_),
    }
    registry.save(MODEL, model, metadata)
    return metadata


def predict(lat: float, lon: float, max_height_m: float) -> TsunamiPrediction:
    model = registry.load(MODEL)
    magnitude = float(model.predict(single_row(FEATURES, [lat, lon, max_height_m]))[0])
    return TsunamiPrediction(magnitude=magnitude, risk=magnitude >= RISK_MAGNITUDE)
