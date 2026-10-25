"""Tsunami source-magnitude regression (NOAA NCEI historical tsunami database).

Random forest on location and the observed maximum water height. Events are independent
historical records (not a forecast series), so the hold-out is a shuffled 75/25 split and the
headline stability figure is 5-fold cross-validation over all rows. Note that the maximum
height is an *outcome* of the event, so this estimates a source magnitude from an observation,
it does not predict a tsunami in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from ndp.ml import plots, registry
from ndp.ml.features import cv_summary, honest_metadata, load_csv, single_row

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
    """Numeric features (missing values left as NaN for the model's imputer); rows with no
    recorded magnitude are dropped."""
    raw = load_csv(CSV)
    df = raw[FEATURES + [TARGET, "YEAR"]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    return df.reset_index(drop=True)


def _build_model() -> Pipeline:
    """Mean imputation lives inside the pipeline, so each training fold (and the final model)
    learns the mean from its own training rows only; filling before the split would leak."""
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
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
    model = _build_model().fit(X_train, y_train)
    metrics = honest_metadata(
        y_train,
        y_test,
        model.predict(X_test),
        "random 75/25 split (independent historical events) plus 5-fold cross-validation",
    )
    cv = cross_val_score(
        _build_model(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2"
    )
    _plots(df)
    metadata = {
        **metrics,
        **cv_summary(cv),
        "cv_folds": 5,
        "n_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "note": "MAXIMUM_HEIGHT is an observed outcome: this estimates source magnitude, not risk.",
    }
    model[-1].n_jobs = 1  # single-row requests: thread start-up would dominate
    registry.save(MODEL, model, metadata)
    return metadata


def predict(lat: float, lon: float, max_height_m: float) -> TsunamiPrediction:
    model = registry.load(MODEL)
    magnitude = float(model.predict(single_row(FEATURES, [lat, lon, max_height_m]))[0])
    return TsunamiPrediction(magnitude=magnitude, risk=magnitude >= RISK_MAGNITUDE)
