"""Fast ML unit tests: no heavy datasets, no network; finish in well under a minute."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from ndp.ml import compare, features, flood, registry, sentiment, storm

# --------------------------------------------------------------------------- features


def test_parse_hurdat_coord():
    assert features.parse_hurdat_coord("28.0N") == 28.0
    assert features.parse_hurdat_coord("94.8W") == -94.8
    assert features.parse_hurdat_coord("12.5S") == -12.5
    assert features.parse_hurdat_coord(" 106.3E ") == 106.3
    assert features.parse_hurdat_coord("45.0") == 45.0
    with pytest.raises(ValueError):
        features.parse_hurdat_coord("")


def test_earthquake_date_to_epoch_both_formats():
    epoch = dt.datetime(1970, 1, 1)
    assert features.earthquake_date_to_epoch("01/01/1970") == 0.0
    expected = (dt.datetime(1965, 1, 2) - epoch).total_seconds()
    assert features.earthquake_date_to_epoch("01/02/1965") == expected
    iso = features.earthquake_date_to_epoch("1975-02-23T02:58:41.000Z")
    assert iso == (dt.datetime(1975, 2, 23, 2, 58, 41) - epoch).total_seconds()
    with pytest.raises(ValueError):
        features.earthquake_date_to_epoch("not a date")


def test_tolerance_accuracy():
    y_true = [10.0, 20.0, 30.0, 40.0]
    assert features.tolerance_accuracy(y_true, y_true) == 100.0
    # tolerance is 0.1 * 40 = 4; errors 1, 1, 5, 5 -> two hits
    assert features.tolerance_accuracy(y_true, [11.0, 21.0, 35.0, 45.0]) == 50.0
    assert features.tolerance_accuracy([], []) == 0.0


def test_date_to_int():
    assert features.date_to_int(dt.date(2005, 8, 29)) == 20050829


def test_single_row_keeps_feature_names():
    row = features.single_row(["a", "b"], [1, 2.5])
    assert list(row.columns) == ["a", "b"] and row.shape == (1, 2)
    with pytest.raises(ValueError):
        features.single_row(["a"], [1, 2])


def test_load_csv_missing_raises_typed_error():
    with pytest.raises(features.DataMissing) as exc:
        features.load_csv("definitely-not-here.csv")
    assert isinstance(exc.value, FileNotFoundError)
    assert "ndp-fetch-data" in str(exc.value)


# --------------------------------------------------------------------------- sentiment


def test_score_text_signs():
    positive = sentiment.score_text(
        "This is a wonderful, beautiful and happy day. Everyone is very good."
    )
    negative = sentiment.score_text(
        "This is a horrible, terrible and sad disaster. Everything is very bad."
    )
    assert positive > 0
    assert negative < 0
    assert sentiment.score_text("") == 0.0


def test_cal_score_rules():
    assert sentiment.cal_score("good", "a") > 0
    assert sentiment.cal_score("bad", "a") < 0
    assert sentiment.cal_score("zzzzqqq", "a") == 0.0


def test_filtered_text_strips_numbers_and_stopwords():
    sentiment.ensure_corpora()
    out = sentiment.filtered_text([("The", "DT"), ("123", "CD"), ("Happy", "JJ"), (",", ",")])
    assert out == [("happy", "JJ")]


def test_disasters_are_in_contract_order():
    assert [d.slug for d in sentiment.DISASTERS][:2] == [
        "japan-2016-earthquake",
        "japan-2011-tsunami",
    ]
    assert len(sentiment.DISASTERS) == 7


# --------------------------------------------------------------------------- storm (real small csv)


def test_storm_train_and_predict():
    meta = storm.train()
    assert meta["n_rows"] == 1020
    assert 0.5 < meta["accuracy"] <= 1.0
    assert 0 < meta["baseline_accuracy"] < meta["accuracy"]  # beats "always the commonest class"
    assert meta["model_type"] in {"decision_tree", "random_forest"}
    assert 0.5 < meta["cv_accuracy_mean"] <= 1.0
    assert "split" in meta and "skill" in meta
    assert set(meta["classes"]) == {"thunderstorm", "sky is clear", "snow", "rain"}
    assert meta["features"] == storm.FEATURES
    result = storm.predict(42.134, 1013, 93, 60, 4)
    assert result.label in meta["classes"]
    assert 1 <= len(result.top) <= 3
    assert result.top[0][0] == result.label
    assert all(0 < p <= 1 for _, p in result.top)
    assert math.isclose(sum(p for _, p in result.top), 1.0, rel_tol=0.05) or len(result.top) < 3


# --------------------------------------------------------------------------- flood


def test_flood_train_and_predict():
    meta = flood.train()
    assert meta["n_rows"] > 60000  # after dropping gaps, outages and stuck-sensor runs
    assert meta["features"] == [
        "level_now",
        "level_1h_ago",
        "level_3h_ago",
        "rain_3h",
        "rain_6h",
        "rain_24h",
    ]
    assert set(meta["horizons"]) == {"1", "3", "6"}
    for scores in meta["horizons"].values():
        for key in ("rmse", "r2", "persistence_rmse", "skill_vs_persistence", "flood_precision"):
            assert key in scores
        assert scores["rmse"] > 0 and scores["persistence_rmse"] > 0
    # headline = 3 h horizon, baseline = persistence
    assert meta["baseline_rmse"] == meta["horizons"]["3"]["persistence_rmse"]
    assert meta["skill"] == meta["horizons"]["3"]["skill_vs_persistence"]
    assert "chronological" in meta["split"]

    calm = flood.predict(1.45, 1.45, 1.45, 0.0, 0.0, 0.0)
    assert [h.hours for h in calm.horizons] == [1, 3, 6]
    assert all(h.threshold_m == flood.FLOOD_THRESHOLD_M == 1.44 for h in calm.horizons)
    assert all(h.persistence_level_m == 1.45 for h in calm.horizons)
    assert calm.peak_level_m == max(h.level_m for h in calm.horizons)
    assert calm.any_flood == any(h.is_flood for h in calm.horizons)
    rising = flood.predict(1.60, 1.50, 1.40, 30.0, 45.0, 80.0)
    # the inputs actually reach the model
    assert [h.level_m for h in rising.horizons] != [h.level_m for h in calm.horizons]


# --------------------------------------------------------------------------- compare (synthetic)


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(-10, 10, n)
    b = rng.uniform(0, 5, n)
    target = 3 * a - 2 * b + rng.normal(0, 0.1, n)
    return pd.DataFrame({"A": a, "B": b, "T": target})


def test_compare_train_predict_roundtrip(monkeypatch):
    spec = compare.HazardSpec(
        name="Synthetic",
        csv_loader=_synthetic_frame,
        features=["A", "B"],
        target="T",
        input_fields=[compare.InputField("a", "A"), compare.InputField("b", "B")],
    )
    monkeypatch.setitem(compare.HAZARDS, "synthetic", spec)
    meta = compare.train("synthetic", max_rows=200)
    assert meta["n_rows"] == 200
    assert set(meta["metrics"]) == set(compare.MODEL_NAMES)
    assert registry.is_trained("compare_synthetic")
    results = compare.predict("synthetic", [2.0, 1.0])
    assert [r.name for r in results] == compare.MODEL_NAMES
    assert results[-1].name == compare.BASELINE_NAME and len(results) == 5
    linear = results[-2]
    assert abs(linear.prediction - (3 * 2.0 - 2 * 1.0)) < 0.5
    assert linear.accuracy_pct > 90 and linear.r2 > 0.95
    assert compare.metrics("synthetic") == meta["metrics"]
    assert compare.metrics("nope") is None
    with pytest.raises(KeyError):
        compare.predict("not-a-hazard", [1.0])


def test_compare_hazard_specs_match_contract():
    assert list(compare.HAZARDS) == ["earthquake", "tsunami", "flood", "hurricane"]
    assert [f.name for f in compare.HAZARDS["hurricane"].input_fields] == ["date", "lat", "lon"]
    assert compare.HAZARDS["hurricane"].input_fields[0].kind == "date"
    assert compare.HAZARDS["flood"].features == [
        "Current rainfall (mm)",
        "Cumulative rainfall (mm)",
    ]


def test_predict_without_artifact_raises_model_not_trained():
    from ndp.ml import earthquake, hurricane, tsunami

    with pytest.raises(registry.ModelNotTrained):
        earthquake.predict(0, 0, 10, dt.date(2020, 1, 1))
    with pytest.raises(registry.ModelNotTrained):
        tsunami.predict(0, 0, 1)
    with pytest.raises(registry.ModelNotTrained):
        hurricane.predict(dt.date(2020, 1, 1), 20, -60)
    assert hurricane.category_for(30) == "Tropical depression"
    assert hurricane.category_for(70) == "Category 1"
    assert hurricane.category_for(140) == "Category 5"
    assert earthquake.severity_for(3.9) == "minor" and earthquake.severity_for(6.0) == "strong"
