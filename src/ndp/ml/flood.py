"""River-level regression for the Hoppers Crossing gauge (cumulative rainfall -> level)."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ndp.ml import plots, registry
from ndp.ml.features import load_csv, regression_metrics, single_row

MODEL = "flood_level"
FLOOD_THRESHOLD_M = 1.44

RAINFALL_CSV = "Hoppers Crossing-Hourly-Rainfall.csv"
RIVER_CSV = "Hoppers Crossing-Hourly-River-Level.csv"
RAIN_COL = "Cumulative rainfall (mm)"
CURRENT_COL = "Current rainfall (mm)"
LEVEL_COL = "Level (m)"
TIME_COL = "Date/Time"
FEATURES = [RAIN_COL]
TARGET = LEVEL_COL


@dataclass(frozen=True)
class FloodPrediction:
    level_m: float
    is_flood: bool
    threshold_m: float


def load_merged() -> pd.DataFrame:
    """Outer-merge rainfall and river level on Date/Time; fill the gaps with 0 as legacy did."""
    rain = load_csv(RAINFALL_CSV)
    river = load_csv(RIVER_CSV)
    merged = pd.merge(rain, river, how="outer", on=[TIME_COL])
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL], errors="coerce")
    merged = merged.sort_values(TIME_COL).reset_index(drop=True)
    for col in (CURRENT_COL, RAIN_COL, LEVEL_COL):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def _plot_series(frame: pd.DataFrame, col: str, title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(frame[TIME_COL], frame[col], lw=0.6, color="#2563eb")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plots.save_current("flood", filename)


def train() -> dict:
    df = load_merged()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))

    _plot_series(df, RAIN_COL, "Cumulative rainfall", "Rainfall (mm)", "f1.png")
    _plot_series(df, LEVEL_COL, "River level", "Level (m)", "f2.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(df[RAIN_COL], df[LEVEL_COL], s=4, alpha=0.4)
    plt.title("River level vs cumulative rainfall")
    plt.xlabel("Cumulative rainfall (mm)")
    plt.ylabel("Level (m)")
    plots.save_current("flood", "f3.png")

    line_x = pd.DataFrame({RAIN_COL: [X[RAIN_COL].min(), X[RAIN_COL].max()]})
    line_y = model.predict(line_x)
    for split_name, Xs, ys, filename in (
        ("Training set", X_train, y_train, "f4.png"),
        ("Test set", X_test, y_test, "f5.png"),
    ):
        plt.figure(figsize=(6, 4))
        plt.scatter(Xs[RAIN_COL], ys, s=4, alpha=0.4, color="#64748b")
        plt.plot(line_x[RAIN_COL], line_y, color="#dc2626")
        plt.title(f"Rainfall vs river level ({split_name})")
        plt.xlabel("Cumulative rainfall (mm)")
        plt.ylabel("Level (m)")
        plots.save_current("flood", filename)

    metadata = {
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "n_rows": int(len(df)),
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "features": FEATURES,
        "target": TARGET,
        "threshold_m": FLOOD_THRESHOLD_M,
    }
    registry.save(MODEL, model, metadata)
    return metadata


def predict(cumulative_rainfall_mm: float) -> FloodPrediction:
    model = registry.load(MODEL)
    level = float(model.predict(single_row(FEATURES, [cumulative_rainfall_mm]))[0])
    return FloodPrediction(
        level_m=level, is_flood=level > FLOOD_THRESHOLD_M, threshold_m=FLOOD_THRESHOLD_M
    )
