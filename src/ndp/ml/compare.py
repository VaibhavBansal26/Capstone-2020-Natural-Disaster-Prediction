"""Side-by-side comparison of four regressors and a baseline on each hazard dataset.

The last entry, "Baseline (predict the mean)", always answers with the training-set mean, so a
reader can see whether any model is better than guessing the average. Splits here are shuffled
(75/25) as in the original comparison; for time-ordered hazards (flood, hurricane, earthquake)
this flatters every model, which is exactly why the dedicated modules use chronological splits.

One artifact per hazard (``compare_<hazard>``) holds the fitted pipelines; the
per-model metrics live in its metadata so the page can render them without
loading the models.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ndp.ml import plots, registry
from ndp.ml.features import (
    baseline_metrics,
    regression_metrics,
    single_row,
    skill_score,
    tolerance_accuracy,
)

BASELINE_NAME = "Baseline (predict the mean)"
MODEL_NAMES = ["Random forest", "Neural network", "SVM", "Linear regression", BASELINE_NAME]
DEFAULT_MAX_ROWS = 15000


@dataclass(frozen=True)
class InputField:
    name: str
    label: str
    kind: Literal["float", "date"] = "float"


@dataclass(frozen=True)
class HazardSpec:
    name: str
    csv_loader: Callable[[], pd.DataFrame]
    features: list[str]
    target: str
    input_fields: list[InputField]


@dataclass(frozen=True)
class ModelResult:
    name: str
    prediction: float
    accuracy_pct: float
    r2: float
    rmse: float


def _earthquake_frame() -> pd.DataFrame:
    from ndp.ml import catalogue

    return catalogue.load_frame()[["Latitude", "Longitude", "Depth", "Magnitude"]]


def _tsunami_frame() -> pd.DataFrame:
    from ndp.ml import tsunami

    return tsunami.load_frame()[tsunami.FEATURES + [tsunami.TARGET]]


def _flood_frame() -> pd.DataFrame:
    from ndp.ml import flood

    # hours with missing readings or logger outages are NaN and dropped by ``train``
    return flood.load_merged()[[flood.CURRENT_COL, flood.RAIN_COL, flood.LEVEL_COL]]


def _hurricane_frame() -> pd.DataFrame:
    from ndp.ml import hurricane

    return hurricane.load_frame()[hurricane.FEATURES + [hurricane.TARGET]]


HAZARDS: dict[str, HazardSpec] = {
    "earthquake": HazardSpec(
        name="Earthquake",
        csv_loader=_earthquake_frame,
        features=["Latitude", "Longitude", "Depth"],
        target="Magnitude",
        input_fields=[
            InputField("lat", "Latitude"),
            InputField("lon", "Longitude"),
            InputField("depth", "Depth (km)"),
        ],
    ),
    "tsunami": HazardSpec(
        name="Tsunami",
        csv_loader=_tsunami_frame,
        features=["LATITUDE", "LONGITUDE", "MAXIMUM_HEIGHT"],
        target="PRIMARY_MAGNITUDE",
        input_fields=[
            InputField("lat", "Latitude"),
            InputField("lon", "Longitude"),
            InputField("max_height", "Maximum water height (m)"),
        ],
    ),
    "flood": HazardSpec(
        name="Flood",
        csv_loader=_flood_frame,
        features=["Current rainfall (mm)", "Cumulative rainfall (mm)"],
        target="Level (m)",
        input_fields=[
            InputField("current_rainfall", "Current rainfall (mm)"),
            InputField("cumulative_rainfall", "Cumulative rainfall (mm)"),
        ],
    ),
    "hurricane": HazardSpec(
        name="Hurricane",
        csv_loader=_hurricane_frame,
        features=["Date", "Latitude", "Longitude"],
        target="Maximum Wind",
        input_fields=[
            InputField("date", "Date", kind="date"),
            InputField("lat", "Latitude"),
            InputField("lon", "Longitude"),
        ],
    ),
}


def artifact_name(hazard: str) -> str:
    return f"compare_{hazard}"


def build_models() -> dict[str, Pipeline]:
    estimators = {
        "Random forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0),
        "Neural network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1500, random_state=0),
        "SVM": SVR(),
        "Linear regression": LinearRegression(),
        BASELINE_NAME: DummyRegressor(strategy="mean"),
    }
    return {name: make_pipeline(StandardScaler(), estimators[name]) for name in MODEL_NAMES}


def _spec(hazard: str) -> HazardSpec:
    try:
        return HAZARDS[hazard]
    except KeyError:
        raise KeyError(f"unknown hazard {hazard!r}; expected one of {sorted(HAZARDS)}") from None


def train(hazard: str, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    spec = _spec(hazard)
    df = spec.csv_loader()[spec.features + [spec.target]].dropna()
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    df = df.reset_index(drop=True)
    X = df[spec.features]
    y = df[spec.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    models = build_models()
    metrics: dict[str, dict[str, float]] = {}
    baseline_rmse = baseline_metrics(y_train, y_test)["baseline_rmse"]
    for name in MODEL_NAMES:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = regression_metrics(y_test, y_pred)
        metrics[name] = {
            "accuracy_pct": tolerance_accuracy(y_test, y_pred),
            "r2": scores["r2"],
            "rmse": scores["rmse"],
            "skill": skill_score(scores["rmse"], baseline_rmse),
        }

    plots.bar_comparison(
        f"{spec.name}: accuracy within 10% of max {spec.target}",
        {
            ("Baseline (mean)" if name == BASELINE_NAME else name): metrics[name]["accuracy_pct"]
            for name in MODEL_NAMES
        },
        "compare",
        f"{hazard}_accuracy.png",
        "Accuracy (%)",
    )

    artifact = {"models": models, "features": list(spec.features), "target": spec.target}
    metadata = {
        "metrics": metrics,
        "baseline": BASELINE_NAME,
        "baseline_rmse": baseline_rmse,
        "split": "random 75/25 split (shuffled), same rows for every model",
        "n_rows": int(len(df)),
        "features": list(spec.features),
        "target": spec.target,
        "hazard": hazard,
    }
    registry.save(artifact_name(hazard), artifact, metadata)
    return metadata


def train_all(max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, dict]:
    return {hazard: train(hazard, max_rows=max_rows) for hazard in HAZARDS}


def predict(hazard: str, values: Sequence[float]) -> list[ModelResult]:
    _spec(hazard)  # validates the hazard key
    artifact = registry.load(artifact_name(hazard))
    meta = registry.metadata(artifact_name(hazard)) or {}
    all_metrics = meta.get("metrics", {})
    row = single_row(artifact["features"], values)
    results: list[ModelResult] = []
    for name in MODEL_NAMES:
        model = artifact["models"][name]
        scores = all_metrics.get(name, {})
        results.append(
            ModelResult(
                name=name,
                prediction=float(model.predict(row)[0]),
                accuracy_pct=float(scores.get("accuracy_pct", float("nan"))),
                r2=float(scores.get("r2", float("nan"))),
                rmse=float(scores.get("rmse", float("nan"))),
            )
        )
    return results


def metrics(hazard: str) -> dict | None:
    meta = registry.metadata(artifact_name(hazard))
    if not meta:
        return None
    return meta.get("metrics")
