"""Hourly river-level forecast for the Hoppers Crossing gauge (1, 3 and 6 hours ahead).

The data are two hourly CSVs (rainfall, river level) from the same station, 2009-2018. They
are merged on a complete hourly grid: any hour that is missing in either file stays missing
(NaN) and every training row whose lookback window or target touches a gap is dropped, so
nothing is interpolated or invented. Lookback features only ever look backwards in time.

Honest framing: river level is highly persistent (lag-1 autocorrelation about 0.985), so the
bar to clear is *persistence*, "the level in h hours equals the level now". The model
predicts the *change* in level over the horizon and adds it to the current level; the
metadata records, per horizon, whether it actually beats persistence on a chronological
hold-out. This is a model of one gauge only and is not an official flood warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ndp.ml import plots, registry
from ndp.ml.features import load_csv, regression_metrics, skill_score

MODEL = "flood_level"
FLOOD_THRESHOLD_M = 1.44
HORIZONS = (1, 3, 6)
HEADLINE_HORIZON = 3
RISE_EVENT_M = 0.10  # a rise this large within the horizon counts as an event
TEST_FRACTION = 0.2
STUCK_SENSOR_HOURS = 168  # an identical reading for a whole week is a stuck logger, not a river
_HOUR = pd.Timedelta(hours=1)

RAINFALL_CSV = "Hoppers Crossing-Hourly-Rainfall.csv"
RIVER_CSV = "Hoppers Crossing-Hourly-River-Level.csv"
TIME_COL = "Date/Time"
CURRENT_COL = "Current rainfall (mm)"
RAIN_COL = "Cumulative rainfall (mm)"  # resets daily at 09:00; used only by ``compare``
LEVEL_COL = "Level (m)"

# Input features, in the order of the public ``predict`` signature.
FEATURES = ["level_now", "level_1h_ago", "level_3h_ago", "rain_3h", "rain_6h", "rain_24h"]
# Differences are derived from the inputs above (so nothing extra is asked of the caller).
MODEL_FEATURES = [*FEATURES, "d_level_1h", "d_level_3h"]


@dataclass(frozen=True)
class HorizonForecast:
    hours: int
    level_m: float
    is_flood: bool
    threshold_m: float
    persistence_level_m: float


@dataclass(frozen=True)
class FloodForecast:
    horizons: list[HorizonForecast]
    any_flood: bool
    peak_level_m: float


# --------------------------------------------------------------------------- data


def load_hourly() -> pd.DataFrame:
    """Rainfall and level on a gap-preserving hourly grid, indexed by timestamp.

    * Timestamps are naive local time, hourly on the hour, newest first in the files.
    * A few timestamps are duplicated (the duplicate rows carry the same value or only NaN):
      the first non-null value wins.
    * ``Level (m) == 0`` is the logger's outage code (sensor maintenance, 14 runs in nine
      years), not a river level, so it becomes NaN.
    * Runs of an identical level lasting ``STUCK_SENSOR_HOURS`` or more (about 10,000 hours in
      2011-2013 sit at exactly 1.40 m or 1.54 m) are a stuck sensor and become NaN.
    * Hours absent from either file are NaN (outer join on the hourly grid).
    """
    rain = load_csv(RAINFALL_CSV)
    river = load_csv(RIVER_CSV)
    out = []
    for frame, cols in ((rain, [CURRENT_COL, RAIN_COL]), (river, [LEVEL_COL])):
        frame = frame.copy()
        frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="coerce")
        frame = frame.dropna(subset=[TIME_COL])
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        out.append(frame.groupby(TIME_COL)[cols].first())
    merged = out[0].join(out[1], how="outer").sort_index()
    merged = merged.asfreq("h")
    merged.loc[merged[LEVEL_COL] <= 0, LEVEL_COL] = np.nan
    before = int(merged[LEVEL_COL].notna().sum())
    merged[LEVEL_COL] = mask_stuck(merged[LEVEL_COL])
    merged.attrs["n_stuck_masked"] = before - int(merged[LEVEL_COL].notna().sum())
    merged.index.name = TIME_COL
    return merged


def mask_stuck(level: pd.Series, min_hours: int = STUCK_SENSOR_HOURS) -> pd.Series:
    """NaN out runs of ``min_hours`` or more consecutive identical readings."""
    run_id = (level != level.shift()).cumsum()
    run_length = level.groupby(run_id).transform("size")
    return level.mask(level.notna() & (run_length >= min_hours))


def load_merged() -> pd.DataFrame:
    """Flat frame (``Date/Time`` column) with NaN for missing hours; used by ``compare``."""
    return load_hourly().reset_index()


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-hour features and horizon targets from a regular hourly frame.

    ``hourly`` needs a regular hourly ``DatetimeIndex`` and columns ``Current rainfall (mm)``
    and ``Level (m)`` (NaN = missing). Everything at row ``t`` uses data from ``t`` and earlier:

    * ``level_now``, ``level_1h_ago``, ``level_3h_ago``: ``shift(0/1/3)``;
    * ``rain_3h``, ``rain_6h``, ``rain_24h``: backward-looking rolling sums that include the
      hour ``t`` itself and require every hour in the window (no partial sums);
    * ``y_<h>``: the level ``h`` hours *later* (``shift(-h)``), the only forward-looking column,
      used as the target and never as a feature.

    Rows with any NaN feature or target are dropped, which also removes every row whose window
    or horizon touches a gap in the record.
    """
    if not isinstance(hourly.index, pd.DatetimeIndex) or (
        len(hourly) > 1 and not (hourly.index.to_series().diff().dropna() == _HOUR).all()
    ):
        raise ValueError("expected a regular hourly DatetimeIndex (one row per hour, sorted)")
    level = hourly[LEVEL_COL].astype(float)
    rain = hourly[CURRENT_COL].astype(float)
    feats = pd.DataFrame(index=hourly.index)
    feats["level_now"] = level
    feats["level_1h_ago"] = level.shift(1)
    feats["level_3h_ago"] = level.shift(3)
    for window in (3, 6, 24):
        feats[f"rain_{window}h"] = rain.rolling(window, min_periods=window).sum()
    for horizon in HORIZONS:
        feats[f"y_{horizon}"] = level.shift(-horizon)
    return feats.dropna()


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["d_level_1h"] = out["level_now"] - out["level_1h_ago"]
    out["d_level_3h"] = out["level_now"] - out["level_3h_ago"]
    return out


def chronological_split(
    features: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = earliest rows, test = latest ``test_fraction``; the last ``max(HORIZONS)``
    training hours are dropped so no training target overlaps the test period."""
    cut = int(len(features) * (1 - test_fraction))
    train = features.iloc[: max(cut - max(HORIZONS), 0)]
    test = features.iloc[cut:]
    return train, test


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=0
    )


# --------------------------------------------------------------------------- training


def _flood_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Precision / recall of the "level above the flood threshold" call."""
    truth = actual > FLOOD_THRESHOLD_M
    called = predicted > FLOOD_THRESHOLD_M
    tp = int(np.sum(truth & called))
    precision = tp / int(called.sum()) if called.any() else None
    recall = tp / int(truth.sum()) if truth.any() else None
    return {"precision": precision, "recall": recall}


def _plot_history(hourly: pd.DataFrame, test_start: pd.Timestamp) -> None:
    level = hourly[LEVEL_COL]
    plt.figure(figsize=(9, 4))
    plt.plot(level.index, level.values, lw=0.5, color="#2563eb", label="River level")
    plt.axhline(FLOOD_THRESHOLD_M, color="#dc2626", lw=1, ls="--", label="1.44 m threshold")
    plt.axvspan(test_start, level.index[-1], color="#f59e0b", alpha=0.15, label="Held-out period")
    plt.ylim(0.8, min(level.max() + 0.1, 3.5))
    plt.title("Hoppers Crossing river level (hourly; gaps and outages left blank)")
    plt.xlabel("Date")
    plt.ylabel("Level (m)")
    plt.legend(loc="upper left", fontsize=8)
    plots.save_current("flood", "f1_level_history.png")


def _plot_forecast(test: pd.DataFrame, model_pred: np.ndarray, horizon: int) -> str:
    """One month of the held-out period: the 30 days with the largest level swing."""
    frame = pd.DataFrame(
        {"actual": test[f"y_{horizon}"], "model": model_pred, "persistence": test["level_now"]},
        # plotted at the time each forecast is *valid for* (issue time + horizon)
        index=test.index + pd.Timedelta(hours=horizon),
    )
    window = 24 * 30
    swing = frame["actual"].rolling(window).max() - frame["actual"].rolling(window).min()
    end = swing.idxmax() if swing.notna().any() else frame.index[-1]
    month = frame.loc[end - pd.Timedelta(hours=window) : end]
    plt.figure(figsize=(9, 4))
    plt.plot(month.index, month["actual"], color="#111827", lw=1.6, label="Actual")
    plt.plot(month.index, month["persistence"], color="#9ca3af", lw=1.1, label="Persistence")
    plt.plot(month.index, month["model"], color="#2563eb", lw=1.1, label="Model")
    plt.axhline(FLOOD_THRESHOLD_M, color="#dc2626", lw=0.8, ls="--")
    plt.title(
        f"{horizon} h ahead: held-out month with the largest swing "
        f"({month.index[0]:%d %b %Y} to {month.index[-1]:%d %b %Y})"
    )
    plt.xlabel("Time the forecast is valid for")
    plt.ylabel("Level (m)")
    plt.legend(loc="upper left", fontsize=8)
    plt.gcf().autofmt_xdate()
    plots.save_current("flood", "f2_forecast_vs_actual.png")
    return f"{month.index[0]:%Y-%m-%d} to {month.index[-1]:%Y-%m-%d}"


def _plot_skill(per_horizon: dict[int, dict]) -> None:
    hours = list(per_horizon)
    x = np.arange(len(hours))
    plt.figure(figsize=(6, 4))
    plt.bar(
        x - 0.2,
        [per_horizon[h]["persistence_rmse"] for h in hours],
        width=0.4,
        color="#9ca3af",
        label="Persistence",
    )
    plt.bar(
        x + 0.2,
        [per_horizon[h]["rmse"] for h in hours],
        width=0.4,
        color="#2563eb",
        label="Model",
    )
    plt.xticks(x, [f"{h} h" for h in hours])
    plt.ylabel("RMSE on held-out period (m)")
    plt.title("Forecast error by horizon (lower is better)")
    plt.legend()
    plots.save_current("flood", "f3_skill_by_horizon.png")


def train() -> dict:
    hourly = load_hourly()
    n_stuck = int(hourly.attrs.get("n_stuck_masked", 0))
    feats = build_features(hourly)
    train_df, test_df = chronological_split(feats)
    train_x, test_x = add_derived(train_df), add_derived(test_df)

    models: dict[int, HistGradientBoostingRegressor] = {}
    per_horizon: dict[int, dict] = {}
    test_preds: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        target = f"y_{horizon}"
        # predict the change over the horizon; level_now is added back at prediction time
        model = _new_model().fit(train_x[MODEL_FEATURES], train_x[target] - train_x["level_now"])
        pred = test_x["level_now"].to_numpy() + model.predict(test_x[MODEL_FEATURES])
        actual = test_x[target].to_numpy()
        persistence = test_x["level_now"].to_numpy()
        scores = regression_metrics(actual, pred)
        base = regression_metrics(actual, persistence)
        flood_model = _flood_scores(actual, pred)
        flood_persist = _flood_scores(actual, persistence)
        rise_truth = (actual - persistence) >= RISE_EVENT_M
        rise_called = (pred - persistence) >= RISE_EVENT_M
        rise_tp = int(np.sum(rise_truth & rise_called))
        per_horizon[horizon] = {
            "rmse": scores["rmse"],
            "mae": scores["mae"],
            "r2": scores["r2"],
            "persistence_rmse": base["rmse"],
            "persistence_mae": base["mae"],
            "skill_vs_persistence": skill_score(scores["rmse"], base["rmse"]),
            "beats_persistence": bool(scores["rmse"] < base["rmse"]),
            "flood_precision": flood_model["precision"],
            "flood_recall": flood_model["recall"],
            "persistence_flood_precision": flood_persist["precision"],
            "persistence_flood_recall": flood_persist["recall"],
            "flood_prevalence": float(np.mean(actual > FLOOD_THRESHOLD_M)),
            "rise_event_m": RISE_EVENT_M,
            "rise_events": int(rise_truth.sum()),
            "rise_precision": rise_tp / int(rise_called.sum()) if rise_called.any() else None,
            "rise_recall": rise_tp / int(rise_truth.sum()) if rise_truth.any() else None,
        }
        models[horizon] = model
        test_preds[horizon] = pred

    _plot_history(hourly, test_x.index[0])
    month = _plot_forecast(test_x, test_preds[HEADLINE_HORIZON], HEADLINE_HORIZON)
    _plot_skill(per_horizon)

    head = per_horizon[HEADLINE_HORIZON]
    year_prevalence = (
        hourly[LEVEL_COL].dropna().gt(FLOOD_THRESHOLD_M).groupby(lambda ts: ts.year).mean()
    )
    level_known = hourly[LEVEL_COL].dropna()
    overall_share = float((level_known > FLOOD_THRESHOLD_M).mean())
    metadata = {
        "r2": head["r2"],
        "rmse": head["rmse"],
        "baseline_rmse": head["persistence_rmse"],
        "skill": head["skill_vs_persistence"],
        "baseline": "persistence (level in h hours = level now)",
        "headline_horizon_h": HEADLINE_HORIZON,
        "horizons": {str(h): per_horizon[h] for h in HORIZONS},
        "split": (
            f"chronological, last {int(TEST_FRACTION * 100)}% of usable hours held out "
            f"({test_x.index[0]:%Y-%m-%d} to {test_x.index[-1]:%Y-%m-%d}), "
            f"{max(HORIZONS)} h gap before the test period"
        ),
        "n_rows": int(len(feats)),
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "n_hours_in_record": int(len(hourly)),
        "n_hours_missing_level": int(hourly[LEVEL_COL].isna().sum()),
        "n_hours_stuck_sensor_masked": n_stuck,
        "features": FEATURES,
        "model_features": MODEL_FEATURES,
        "target": "level at t + h (modelled as the change from level_now)",
        "threshold_m": FLOOD_THRESHOLD_M,
        "horizons_h": list(HORIZONS),
        "forecast_plot_window": month,
        "share_above_threshold_by_year": {
            str(year): float(share) for year, share in year_prevalence.items()
        },
        "notes": [
            "Skill is measured against persistence, not against the average.",
            "The 1.44 m threshold sits at the median of the whole record "
            f"({overall_share:.0%}"
            " of all hours are above it) and the baseline level drifts upwards over the years, "
            "so flood precision/recall in the held-out period say little about real flood events.",
        ],
    }
    registry.save(MODEL, {"models": models, "features": MODEL_FEATURES}, metadata)
    return metadata


# --------------------------------------------------------------------------- prediction


def predict(
    level_now_m: float,
    level_1h_ago_m: float,
    level_3h_ago_m: float,
    rain_3h_mm: float,
    rain_6h_mm: float,
    rain_24h_mm: float,
) -> FloodForecast:
    artifact = registry.load(MODEL)
    row = pd.DataFrame(
        [
            [
                level_now_m,
                level_1h_ago_m,
                level_3h_ago_m,
                rain_3h_mm,
                rain_6h_mm,
                rain_24h_mm,
            ]
        ],
        columns=FEATURES,
        dtype=float,
    )
    row = add_derived(row)[artifact["features"]]
    horizons: list[HorizonForecast] = []
    for hours in HORIZONS:
        level = float(level_now_m) + float(artifact["models"][hours].predict(row)[0])
        horizons.append(
            HorizonForecast(
                hours=hours,
                level_m=level,
                is_flood=level > FLOOD_THRESHOLD_M,
                threshold_m=FLOOD_THRESHOLD_M,
                persistence_level_m=float(level_now_m),
            )
        )
    return FloodForecast(
        horizons=horizons,
        any_flood=any(h.is_flood for h in horizons),
        peak_level_m=max(h.level_m for h in horizons),
    )
