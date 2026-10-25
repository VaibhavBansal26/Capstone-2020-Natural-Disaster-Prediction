"""USGS significant-earthquake catalogue 1965-2016: loading and descriptive statistics.

This module only describes the catalogue (counts by magnitude, month and year, and the plots
on the earthquake "Graphical analysis" page). The hazard estimate for a place lives in
:mod:`ndp.ml.hazard`. A magnitude regression used to live here as well; it had no skill
(held-out R-squared about 0, because the catalogue only holds M >= 5.5 events and location and
depth say almost nothing about the size of a future one) and was removed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ndp.ml import plots, registry
from ndp.ml.features import earthquake_date_to_epoch, load_csv

STATS = "earthquake_catalogue"
CSV = "database.csv"
TARGET = "Magnitude"


def load_frame() -> pd.DataFrame:
    df = load_csv(CSV)
    if "Type" in df.columns:  # the file also lists nuclear explosions, explosions and a rock burst
        df = df[df["Type"] == "Earthquake"]
    df = df.dropna(subset=["Date", "Latitude", "Longitude", "Depth", TARGET]).copy()
    df["date_epoch"] = df["Date"].map(earthquake_date_to_epoch)
    df["datetime"] = pd.to_datetime(df["date_epoch"], unit="s")
    return df.sort_values("date_epoch", kind="stable").reset_index(drop=True)


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
    stats = _compute_stats(df)
    _plots(df, stats)
    metadata = {
        "n_events": stats["n_events"],
        "n_rows": stats["n_events"],
        "note": "Descriptive statistics of the catalogue; not a predictive model.",
    }
    registry.save(STATS, stats, metadata)
    return metadata


def stats() -> dict:
    return registry.load(STATS)
