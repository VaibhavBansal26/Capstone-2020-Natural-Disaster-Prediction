"""Storm / weather-description classifier from hourly surface observations."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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


def train() -> dict:
    df = load_frame()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=99, stratify=y
    )
    model = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    model.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    classes = [str(c) for c in model.classes_]

    plt.figure(figsize=(16, 8))
    plot_tree(
        model,
        max_depth=3,
        feature_names=FEATURES,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Storm classifier (top three levels)")
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
        "classes": classes,
        "n_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "tree_depth": int(model.get_depth()),
    }
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
