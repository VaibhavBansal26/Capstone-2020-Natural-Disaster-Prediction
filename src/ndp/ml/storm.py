"""Storm / weather-description classifier from hourly surface observations.

A decision tree and a random forest are compared by 5-fold stratified cross-validated accuracy
on the training split and the better one is kept. The file has no timestamp column, so a
chronological split is impossible; the hold-out is a stratified 80/20 split and temporal
leakage between neighbouring hourly rows cannot be ruled out. The baseline is "always answer
the most common class".
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from ndp.ml import plots, registry
from ndp.ml.features import load_csv, single_row

MODEL = "storm_classifier"
CSV = "weather_data.csv"
FEATURES = ["Temperature", "Pressure", "Humidity", "Wind Direction", "Wind Speed"]
TARGET = "Weather Description"
TOP_N = 3


@dataclass(frozen=True)
class StormPrediction:
    label: str
    top: list[tuple[str, float]]


def load_frame() -> pd.DataFrame:
    df = load_csv(CSV)[FEATURES + [TARGET]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = df[TARGET].astype(str)
    return df.dropna().reset_index(drop=True)


def _candidates() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(min_samples_split=20, random_state=99),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=99
        ),
    }


def train() -> dict:
    df = load_frame()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=99, stratify=y
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=99)
    cv_scores = {
        name: cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy")
        for name, est in _candidates().items()
    }
    chosen = max(cv_scores, key=lambda name: float(cv_scores[name].mean()))
    model = _candidates()[chosen].fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    majority = y_train.value_counts().idxmax()
    baseline_accuracy = float((y_test == majority).mean())
    classes = [str(c) for c in model.classes_]

    tree = (
        model if chosen == "decision_tree" else _candidates()["decision_tree"].fit(X_train, y_train)
    )
    plt.figure(figsize=(16, 8))
    plot_tree(
        tree,
        max_depth=3,
        feature_names=FEATURES,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    note = "" if chosen == "decision_tree" else " (illustration; the forest is the deployed model)"
    plt.title(f"Decision tree, top three levels{note}")
    plots.save_current("storm", "tree.png")

    counts = y.value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values, color="#2563eb")
    plt.title("Weather description class distribution")
    plt.ylabel("Observations")
    plt.xticks(rotation=15)
    plots.save_current("storm", "classes.png")

    metadata = {
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        # same idea as the regressors' skill: share of the baseline's errors removed
        "skill": float((accuracy - baseline_accuracy) / (1 - baseline_accuracy))
        if baseline_accuracy < 1
        else 0.0,
        "model_type": chosen,
        "cv_accuracy_mean": float(cv_scores[chosen].mean()),
        "cv_accuracy_std": float(cv_scores[chosen].std()),
        "cv_accuracy_by_model": {k: float(v.mean()) for k, v in cv_scores.items()},
        "split": (
            "stratified random 80/20 split (no timestamps in the file); model chosen by "
            "5-fold cross-validation on the training part"
        ),
        "classes": classes,
        "n_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
    }
    if chosen == "decision_tree":
        metadata["tree_depth"] = int(model.get_depth())
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # single-row requests: thread start-up would dominate
    registry.save(MODEL, model, metadata)
    return metadata


def predict(
    temperature: float,
    pressure: float,
    humidity: float,
    wind_direction: float,
    wind_speed: float,
) -> StormPrediction:
    model = registry.load(MODEL)
    row = single_row(FEATURES, [temperature, pressure, humidity, wind_direction, wind_speed])
    probabilities = model.predict_proba(row)[0]
    order = np.argsort(probabilities)[::-1]
    top = [
        (str(model.classes_[i]), float(probabilities[i]))
        for i in order[:TOP_N]
        if probabilities[i] > 0
    ]
    label = str(model.classes_[order[0]])
    return StormPrediction(label=label, top=top)
