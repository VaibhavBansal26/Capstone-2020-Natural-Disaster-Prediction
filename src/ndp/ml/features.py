"""Shared feature helpers used by every trainer and predict function."""

from __future__ import annotations

import datetime as dt
import errno
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ndp.config import get_settings

_EPOCH = dt.datetime(1970, 1, 1)
_EARTHQUAKE_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%dT%H:%M:%S.%fZ")


class DataMissing(FileNotFoundError):
    """A raw dataset is not present in ``settings.data_dir``."""

    def __init__(self, filename: str):
        path = get_settings().data_dir / filename
        super().__init__(
            errno.ENOENT,
            f"dataset '{filename}' is missing; run `uv run ndp-fetch-data` to download it",
            str(path),
        )
        self.dataset = filename


def load_csv(filename: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Read ``filename`` from ``settings.data_dir``; raise :class:`DataMissing` when absent."""
    path = get_settings().data_dir / filename
    if not path.exists():
        raise DataMissing(filename)
    return pd.read_csv(path, **read_csv_kwargs)


def earthquake_date_to_epoch(s: str) -> float:
    """Seconds since 1970-01-01 for the two date formats found in the USGS ``database.csv``."""
    text = str(s).strip()
    for fmt in _EARTHQUAKE_DATE_FORMATS:
        try:
            parsed = dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        return (parsed - _EPOCH).total_seconds()
    raise ValueError(f"unrecognised earthquake date: {s!r}")


def parse_hurdat_coord(s: str) -> float:
    """HURDAT2 coordinate strings: ``"28.0N" -> 28.0``, ``"94.8W" -> -94.8``."""
    text = str(s).strip().upper()
    if not text:
        raise ValueError("empty HURDAT2 coordinate")
    hemisphere = text[-1]
    if hemisphere in "NSEW":
        value = float(text[:-1])
        return -value if hemisphere in "SW" else value
    return float(text)


def tolerance_accuracy(
    y_true: Sequence[float], y_pred: Sequence[float], tol_frac: float = 0.1
) -> float:
    """Legacy 'accuracy' for regressors: % of rows with |err| < tol_frac * max(|y_true|)."""
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if truth.size == 0:
        return 0.0
    tolerance = tol_frac * float(np.max(np.abs(truth)))
    hits = np.abs(pred - truth) < tolerance
    return float(hits.mean() * 100.0)


def date_to_int(d: dt.date) -> int:
    """``datetime.date(2005, 8, 29) -> 20050829`` (the HURDAT2 ``Date`` encoding)."""
    return d.year * 10000 + d.month * 100 + d.day


def single_row(features: Sequence[str], values: Sequence[float]) -> pd.DataFrame:
    """One-row DataFrame carrying the training feature names (keeps sklearn quiet)."""
    if len(features) != len(values):
        raise ValueError(f"expected {len(features)} values for {list(features)}, got {len(values)}")
    return pd.DataFrame([[float(v) for v in values]], columns=list(features))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "r2": float(r2_score(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "mae": float(mean_absolute_error(truth, pred)),
    }
