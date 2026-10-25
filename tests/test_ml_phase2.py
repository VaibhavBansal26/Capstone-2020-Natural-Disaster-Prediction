"""Phase 2 ("honest ML") tests: hazard maths, flood feature construction, baselines.

Everything here is synthetic or tiny so the file runs in a few seconds; the real-data checks
(``test_hazard_real_catalogue``) only read the 23k-row USGS catalogue.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from ndp.ml import compare, features, flood, hazard, hurricane, registry

# --------------------------------------------------------------------------- hazard maths


def _catalogue(lats, lons, mags, *, depth=10.0, start=1965.0, years=52.0, seed=0):
    n = len(mags)
    rng = np.random.default_rng(seed)
    year = start + rng.uniform(0, years, n)
    return {
        "lat": np.asarray(lats, dtype=float),
        "lon": np.asarray(lons, dtype=float),
        "depth": np.full(n, depth),
        "mag": np.asarray(mags, dtype=float),
        "epoch": (year - 1970.0) * 365.25 * 86400.0,
    }


def _gr_magnitudes(n, b, rng, mc=hazard.MC, step=0.1):
    """Continuous Gutenberg-Richter magnitudes, then rounded to the catalogue's 0.1 step."""
    beta = b * math.log(10)
    continuous = (mc - step / 2) + rng.exponential(1 / beta, n)
    return np.round(continuous / step) * step


@pytest.mark.parametrize("true_b", [0.8, 1.0, 1.2])
def test_aki_b_value_recovers_known_b(true_b):
    rng = np.random.default_rng(42)
    mags = _gr_magnitudes(20000, true_b, rng)
    mags = mags[mags >= hazard.MC - 1e-9]
    assert abs(hazard.aki_b_value(mags) - true_b) < 0.05


def test_aki_b_value_rejects_degenerate_input():
    with pytest.raises(ValueError):
        hazard.aki_b_value(np.array([5.5]))
    with pytest.raises(ValueError):
        hazard.aki_b_value(np.array([4.0, 4.5]))  # nothing at or above the cut-off


def test_synthetic_catalogue_recovers_rate_and_b():
    rng = np.random.default_rng(7)
    years, true_rate, true_b = 52.0, 12.0, 1.0
    n = rng.poisson(true_rate * years)
    mags = _gr_magnitudes(n, true_b, rng)
    mags = mags[mags >= hazard.MC - 1e-9]
    cat = _catalogue(
        35.0 + rng.normal(0, 0.3, mags.size), 139.0 + rng.normal(0, 0.3, mags.size), mags
    )
    est = hazard.estimate_from_catalogue(
        cat, years=years, b_global=0.9, lat=35.0, lon=139.0, radius_km=300
    )
    assert est.n_events == mags.size >= 50
    assert not est.b_is_global
    assert est.annual_rate_m55 == pytest.approx(true_rate, rel=0.1)
    assert est.b_value == pytest.approx(true_b, abs=0.1)
    assert est.annual_rate["6.0"] == pytest.approx(
        est.annual_rate_m55 * 10 ** (-est.b_value * 0.5), rel=1e-12
    )


def test_poisson_probability_matches_formula():
    cat = _catalogue([10.0] * 200, [20.0] * 200, [5.5] * 100 + [6.0] * 60 + [7.0] * 40)
    est = hazard.estimate_from_catalogue(cat, years=20.0, b_global=1.0, lat=10.0, lon=20.0)
    assert est.annual_rate_m55 == pytest.approx(200 / 20.0)
    for key in ("6.0", "7.0", "8.0"):
        rate = est.annual_rate[key]
        assert est.prob_10y[key] == pytest.approx(1 - math.exp(-10 * rate), rel=1e-12)
        assert est.prob_1y[key] == pytest.approx(1 - math.exp(-rate), rel=1e-12)
        assert est.prob_10y[key] >= est.prob_1y[key]
    assert est.annual_rate["7.0"] < est.annual_rate["6.0"]
    assert hazard.poisson_probability(0.0, 10.0) == 0.0
    assert hazard.poisson_probability(1e-12, 10.0) == pytest.approx(1e-11)  # no cancellation


def test_b_value_falls_back_to_global_below_50_events():
    rng = np.random.default_rng(3)
    mags = _gr_magnitudes(49, 1.0, rng)
    cat = _catalogue([0.0] * mags.size, [0.0] * mags.size, mags)
    few = hazard.estimate_from_catalogue(cat, years=10, b_global=0.77, lat=0, lon=0)
    assert few.b_is_global and few.b_value == 0.77
    mags = _gr_magnitudes(50, 1.0, rng)
    cat = _catalogue([0.0] * mags.size, [0.0] * mags.size, mags)
    enough = hazard.estimate_from_catalogue(cat, years=10, b_global=0.77, lat=0, lon=0)
    assert not enough.b_is_global and enough.b_value != 0.77


def test_radius_is_clamped_and_filters_events():
    # one event 1 km away, one 100 km away, one 500 km away (along a meridian)
    lats = [10.0 + d / 111.195 for d in (1, 100, 500)]
    cat = _catalogue(lats, [20.0] * 3, [6.0, 6.5, 7.0])
    tiny = hazard.estimate_from_catalogue(cat, years=10, b_global=1.0, lat=10, lon=20, radius_km=1)
    assert tiny.radius_km == hazard.MIN_RADIUS_KM == 25.0
    assert tiny.n_events == 1 and tiny.max_magnitude == 6.0
    mid = hazard.estimate_from_catalogue(cat, years=10, b_global=1.0, lat=10, lon=20, radius_km=300)
    assert mid.n_events == 2 and mid.max_magnitude == 6.5
    huge = hazard.estimate_from_catalogue(
        cat, years=10, b_global=1.0, lat=10, lon=20, radius_km=99999
    )
    assert huge.radius_km == hazard.MAX_RADIUS_KM == 2000.0
    assert huge.n_events == 3 and huge.max_magnitude == 7.0
    assert huge.nearest_event["distance_km"] == pytest.approx(1.0, abs=0.1)
    assert huge.nearest_event["magnitude"] == 6.0  # the nearest M >= 6 event


def test_open_ocean_has_zero_events_and_low_level():
    cat = _catalogue([35.0] * 30, [139.0] * 30, [6.0] * 30)
    est = hazard.estimate_from_catalogue(cat, years=52, b_global=1.0, lat=-40.0, lon=-30.0)
    assert est.n_events == 0
    assert est.annual_rate_m55 == 0.0
    assert all(v == 0.0 for v in est.annual_rate.values())
    assert all(v == 0.0 for v in est.prob_10y.values())
    assert est.level == "low" and est.elevated is False
    assert est.max_magnitude is None and est.max_date is None
    assert est.depth_median_km is None and est.nearest_event is None
    assert est.b_is_global and est.b_value == 1.0


def test_antimeridian_and_poles():
    # 179.5E and 179.5W are about 1 degree of longitude apart, not 359
    d = hazard.haversine_km(0.0, 179.5, np.array([0.0]), np.array([-179.5]))[0]
    assert d == pytest.approx(111.19, rel=0.01)
    # same on the other side, and for out-of-range query longitudes
    assert hazard.haversine_km(0.0, -179.5, np.array([0.0]), np.array([179.5]))[0] == pytest.approx(
        d
    )
    assert hazard.haversine_km(0.0, 539.5, np.array([0.0]), np.array([-179.5]))[0] == pytest.approx(
        d, rel=1e-6
    )
    # all meridians meet at the pole: 0.1 degrees from the pole, opposite sides = 22 km
    pole = hazard.haversine_km(89.9, 0.0, np.array([89.9]), np.array([180.0]))[0]
    assert pole == pytest.approx(2 * 0.1 * 111.195, rel=0.01)
    assert hazard.haversine_km(90.0, 0.0, np.array([90.0]), np.array([123.0]))[0] == pytest.approx(
        0
    )
    # Tokyo to Los Angeles is about 8,800 km
    tokyo_la = hazard.haversine_km(35.68, 139.69, np.array([34.05]), np.array([-118.24]))[0]
    assert tokyo_la == pytest.approx(8815, rel=0.01)

    cat = _catalogue([0.0, 0.0, 0.0], [-179.5, 179.9, 120.0], [6.0, 6.2, 6.4])
    est = hazard.estimate_from_catalogue(
        cat, years=10, b_global=1.0, lat=0, lon=179.5, radius_km=200
    )
    assert est.n_events == 2
    wrapped = hazard.estimate_from_catalogue(
        cat, years=10, b_global=1.0, lat=0, lon=-180.5, radius_km=200
    )
    assert wrapped.n_events == 2 and wrapped.lon == pytest.approx(179.5)
    polar = hazard.estimate_from_catalogue(cat, years=10, b_global=1.0, lat=90.0, lon=0)
    assert polar.n_events == 0 and polar.level == "low"


def test_invalid_coordinates_raise():
    cat = _catalogue([0.0], [0.0], [6.0])
    for bad_lat, bad_lon in ((91, 0), (-90.5, 0), (float("nan"), 0), (0, float("inf"))):
        with pytest.raises(ValueError):
            hazard.estimate_from_catalogue(cat, years=1, b_global=1.0, lat=bad_lat, lon=bad_lon)


@pytest.mark.parametrize(
    ("prob", "label"),
    [(0.0, "low"), (0.049, "low"), (0.05, "moderate"), (0.249, "moderate"), (0.25, "high"),
     (0.599, "high"), (0.60, "very high"), (1.0, "very high")],
)  # fmt: skip
def test_level_thresholds(prob, label):
    assert hazard.level_for(prob) == label


def test_elevated_follows_level():
    quiet = _catalogue([0.0] * 3, [0.0] * 3, [5.5, 5.6, 5.9])
    est = hazard.estimate_from_catalogue(quiet, years=52, b_global=1.0, lat=0, lon=0)
    assert est.level == "moderate" and not est.elevated  # 3 events / 52 y -> P(M>=6, 10 y) ~ 0.17
    busy_mags = _gr_magnitudes(520, 1.0, np.random.default_rng(9))
    busy = _catalogue([0.0] * 520, [0.0] * 520, busy_mags)
    est = hazard.estimate_from_catalogue(busy, years=52, b_global=1.0, lat=0, lon=0)
    assert est.level == "very high" and est.elevated


def test_hazard_real_catalogue(monkeypatch):
    monkeypatch.setattr(hazard, "MODEL", "earthquake_hazard_test")
    with pytest.raises(registry.ModelNotTrained):
        hazard.predict(35.68, 139.69)
    meta = hazard.train()
    assert meta["mc"] == 5.5 and meta["year_min"] == 1965 and meta["year_max"] == 2016
    assert meta["n_events"] > 20000 and 0.9 < meta["b_value_global"] < 1.1
    assert meta["backtest"]["lift"] > 1.5  # past activity beats a uniform map
    tokyo = hazard.predict(35.68, 139.69)
    ocean = hazard.predict(-45.0, -100.0, radius_km=300)  # empty South Pacific
    assert tokyo.n_events > 100 and tokyo.elevated and tokyo.max_magnitude >= 7.5
    assert tokyo.max_date == "2011-03-11"
    assert ocean.n_events <= 5 and ocean.prob_10y["6.0"] < tokyo.prob_10y["6.0"]
    assert hazard.predict(0.0, 0.0, radius_km=10).radius_km == 25.0
    with pytest.raises(ValueError):
        hazard.predict(120.0, 0.0)


# --------------------------------------------------------------------------- flood features


def _hourly(n=60, start="2020-01-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="h", name=flood.TIME_COL)
    return pd.DataFrame(
        {
            flood.LEVEL_COL: 1.0 + 0.01 * np.arange(n) ** 1.5,  # strictly increasing, irregular
            flood.CURRENT_COL: (np.arange(n) % 7).astype(float),
        },
        index=idx,
    )


def test_flood_features_match_definition_and_look_backwards_only():
    hourly = _hourly()
    out = flood.build_features(hourly)
    level, rain = hourly[flood.LEVEL_COL], hourly[flood.CURRENT_COL]
    t = out.index[0]
    i = hourly.index.get_loc(t)
    assert i == 24 - 1  # first row with a full 24 h rain window
    row = out.loc[t]
    assert row["level_now"] == level.iloc[i]
    assert row["level_1h_ago"] == level.iloc[i - 1]
    assert row["level_3h_ago"] == level.iloc[i - 3]
    assert row["rain_3h"] == rain.iloc[i - 2 : i + 1].sum()
    assert row["rain_6h"] == rain.iloc[i - 5 : i + 1].sum()
    assert row["rain_24h"] == rain.iloc[i - 23 : i + 1].sum()
    for h in flood.HORIZONS:
        assert row[f"y_{h}"] == level.iloc[i + h]
    # last usable row leaves room for the longest horizon
    assert out.index[-1] == hourly.index[-1 - max(flood.HORIZONS)]


def test_flood_features_do_not_depend_on_the_future():
    hourly = _hourly(80)
    base = flood.build_features(hourly)
    cut = hourly.index[50]
    changed = hourly.copy()
    changed.loc[changed.index > cut, flood.LEVEL_COL] += 5.0  # rewrite the future
    changed.loc[changed.index > cut, flood.CURRENT_COL] += 9.0
    other = flood.build_features(changed)
    input_cols = flood.FEATURES
    before = base.index[base.index <= cut]
    pd.testing.assert_frame_equal(base.loc[before, input_cols], other.loc[before, input_cols])
    # only the targets of rows whose horizon reaches past the cut may change
    safe = base.index[base.index <= cut - pd.Timedelta(hours=max(flood.HORIZONS))]
    pd.testing.assert_frame_equal(base.loc[safe], other.loc[safe])
    assert not base.loc[before[-1], "y_6"] == other.loc[before[-1], "y_6"]


def test_flood_features_drop_rows_touching_gaps():
    hourly = _hourly(100)
    gap = hourly.index[40]
    hourly.loc[gap, flood.LEVEL_COL] = np.nan
    hourly.loc[hourly.index[70], flood.CURRENT_COL] = np.nan
    out = flood.build_features(hourly)
    assert out.notna().all().all()
    pos = {ts: i for i, ts in enumerate(hourly.index)}
    for ts in out.index:
        i = pos[ts]
        used = {i, i - 1, i - 3, *(i + h for h in flood.HORIZONS)}
        assert 40 not in used, "a feature or target would read the missing level"
        assert not (i - 23 <= 70 <= i), "rain window spans the missing rainfall"


def test_flood_features_reject_irregular_index():
    hourly = _hourly(30).drop(_hourly(30).index[10])
    with pytest.raises(ValueError):
        flood.build_features(hourly)


def test_flood_chronological_split_has_no_overlap():
    out = flood.build_features(_hourly(400))
    train, test = flood.chronological_split(out)
    assert train.index.max() < test.index.min()
    gap = (test.index.min() - train.index.max()) / pd.Timedelta(hours=1)
    assert gap > max(flood.HORIZONS)  # no training target reaches into the test period
    assert len(test) == pytest.approx(0.2 * len(out), abs=1)


def test_load_hourly_merges_on_timestamp_and_masks_outages(monkeypatch):
    rain = pd.DataFrame(
        {
            "Date/Time": ["2020-01-01 03:00:00", "2020-01-01 02:00:00", "2020-01-01 01:00:00",
                          "2020-01-01 01:00:00", "2020-01-01 00:00:00"],
            flood.CURRENT_COL: [0.5, 0.0, np.nan, 1.0, 2.0],
            flood.RAIN_COL: [0.5, 0.0, np.nan, 1.0, 2.0],
        }
    )  # fmt: skip
    river = pd.DataFrame(
        {"Date/Time": ["2020-01-01 04:00:00", "2020-01-01 02:00:00", "2020-01-01 01:00:00"],
         flood.LEVEL_COL: [1.5, 0.0, 1.4]}
    )  # fmt: skip
    frames = {flood.RAINFALL_CSV: rain, flood.RIVER_CSV: river}
    monkeypatch.setattr(flood, "load_csv", lambda name: frames[name])
    hourly = flood.load_hourly()
    assert list(hourly.index) == list(
        pd.date_range("2020-01-01 00:00", "2020-01-01 04:00", freq="h")
    )
    assert hourly.loc["2020-01-01 01:00", flood.CURRENT_COL] == 1.0  # duplicate: first non-null
    assert np.isnan(hourly.loc["2020-01-01 02:00", flood.LEVEL_COL])  # 0.0 = logger outage
    assert np.isnan(hourly.loc["2020-01-01 00:00", flood.LEVEL_COL])  # absent from the river file
    assert np.isnan(hourly.loc["2020-01-01 04:00", flood.CURRENT_COL])  # absent from rainfall
    assert hourly.loc["2020-01-01 04:00", flood.LEVEL_COL] == 1.5
    flat = flood.load_merged()
    assert flood.TIME_COL in flat.columns and len(flat) == 5


def test_flood_train_on_synthetic_series_reports_persistence_comparison(monkeypatch):
    rng = np.random.default_rng(0)
    n = 3000
    rain = np.where(rng.random(n) < 0.05, rng.exponential(5, n), 0.0)
    level = np.empty(n)
    level[0] = 1.4
    for i in range(1, n):  # level follows lagged rain: genuinely predictable rises
        lagged = rain[i - 2] if i >= 2 else 0.0
        level[i] = 1.4 + 0.97 * (level[i - 1] - 1.4) + 0.02 * lagged
    idx = pd.date_range("2015-01-01", periods=n, freq="h", name=flood.TIME_COL)
    hourly = pd.DataFrame({flood.LEVEL_COL: level, flood.CURRENT_COL: rain}, index=idx)
    monkeypatch.setattr(flood, "load_hourly", lambda: hourly)
    monkeypatch.setattr(flood, "MODEL", "flood_level_test")
    meta = flood.train()
    assert set(meta["horizons"]) == {"1", "3", "6"}
    assert all(
        s["beats_persistence"] and s["skill_vs_persistence"] > 0 for s in meta["horizons"].values()
    )
    assert "chronological" in meta["split"]
    assert flood.predict(1.5, 1.49, 1.47, 5.0, 8.0, 12.0).horizons[0].hours == 1


# --------------------------------------------------------------------------- baselines


def test_baseline_metrics_and_skill():
    y_train = [1.0, 2.0, 3.0]  # mean 2
    y_test = [2.0, 2.0, 5.0]
    base = features.baseline_metrics(y_train, y_test)
    assert base["baseline_rmse"] == pytest.approx(math.sqrt((0 + 0 + 9) / 3))
    test_mean = sum(y_test) / 3
    ss_tot = sum((y - test_mean) ** 2 for y in y_test)
    assert base["baseline_r2"] == pytest.approx(1 - 9 / ss_tot)
    assert features.skill_score(1.0, 2.0) == pytest.approx(0.5)
    assert features.skill_score(3.0, 2.0) == pytest.approx(-0.5)  # worse than baseline
    assert features.skill_score(1.0, 0.0) == 0.0
    meta = features.honest_metadata(y_train, y_test, [2.0, 2.0, 5.0], "unit test split")
    assert meta["rmse"] == 0.0 and meta["skill"] == pytest.approx(1.0)
    assert meta["baseline_rmse"] == base["baseline_rmse"] and meta["split"] == "unit test split"
    assert features.cv_summary([0.2, 0.4])["cv_r2_mean"] == pytest.approx(0.3)


def test_registry_writes_atomically(tmp_path, monkeypatch):
    from ndp.config import get_settings

    registry.save("atomic_check", {"x": 1}, {"m": 1})
    models_dir = get_settings().models_dir
    assert not list(models_dir.glob("*.tmp")) and not list(models_dir.glob(".*.tmp"))
    assert registry.load("atomic_check") == {"x": 1}


# --------------------------------------------------------------------------- compare baseline


def _linear_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    a = rng.uniform(-5, 5, 300)
    return pd.DataFrame({"A": a, "B": rng.uniform(0, 1, 300), "T": 10 + 2 * a})


def test_compare_includes_baseline_row_last(monkeypatch):
    spec = compare.HazardSpec(
        name="Synthetic",
        csv_loader=_linear_frame,
        features=["A", "B"],
        target="T",
        input_fields=[compare.InputField("a", "A"), compare.InputField("b", "B")],
    )
    monkeypatch.setitem(compare.HAZARDS, "synthetic2", spec)
    assert compare.MODEL_NAMES[-1] == compare.BASELINE_NAME == "Baseline (predict the mean)"
    assert len(compare.MODEL_NAMES) == 5
    meta = compare.train("synthetic2", max_rows=0)
    assert set(meta["metrics"]) == set(compare.MODEL_NAMES)
    results = compare.predict("synthetic2", [3.0, 0.5])
    assert len(results) == 5 and results[-1].name == compare.BASELINE_NAME
    baseline, linear = results[-1], results[-2]
    # the baseline answers the same thing wherever you ask
    assert compare.predict("synthetic2", [-4.0, 0.9])[-1].prediction == pytest.approx(
        baseline.prediction
    )
    assert baseline.prediction == pytest.approx(10.0, abs=1.5)  # about the training mean
    assert baseline.r2 < 0.05 and linear.r2 > 0.99
    assert baseline.rmse > 5 * linear.rmse
    assert meta["metrics"][compare.BASELINE_NAME]["skill"] == pytest.approx(0.0, abs=1e-9)
    assert meta["metrics"]["Linear regression"]["skill"] > 0.9
    assert meta["baseline_rmse"] == pytest.approx(baseline.rmse)
    assert "split" in meta


# --------------------------------------------------------------------------- hurricane split


def _synthetic_hurricanes() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for year in range(1990, 2016):
        for _ in range(60):
            month, day = int(rng.integers(6, 11)), int(rng.integers(1, 28))
            lat = float(rng.uniform(10, 40))
            rows.append(
                {
                    "Date": year * 10000 + month * 100 + day,
                    "Latitude": lat,
                    "Longitude": float(rng.uniform(-100, -40)),
                    "Maximum Wind": float(120 - 2.0 * lat + rng.normal(0, 8)),
                }
            )
    return pd.DataFrame(rows)


def test_hurricane_split_is_chronological_at_2005(monkeypatch):
    frame = _synthetic_hurricanes()
    monkeypatch.setattr(hurricane, "load_frame", lambda: frame)
    monkeypatch.setattr(hurricane, "MODEL", "hurricane_wind_test")
    meta = hurricane.train()
    year = frame["Date"] // 10000
    assert meta["n_train"] == int((year < 2005).sum())
    assert meta["n_test"] == int((year >= 2005).sum())
    assert meta["split_year"] == 2005 and "chronological" in meta["split"]
    assert meta["features"] == ["lat", "lon", "day_of_year", "abs_lat"]  # no year feature
    for key in ("baseline_rmse", "skill", "r2", "rmse"):
        assert key in meta
    assert meta["skill"] > 0.3  # the synthetic latitude signal is learnable
    pred = hurricane.predict(dt.date(2010, 9, 10), 15.0, -60.0)
    assert pred.category == hurricane.category_for(pred.max_wind_kt)
    south = hurricane.predict(dt.date(2010, 9, 10), 38.0, -60.0)
    assert south.max_wind_kt < pred.max_wind_kt  # position reaches the model


def test_mask_stuck_sensor_runs():
    values = [1.5, 1.6] + [1.4] * 200 + [1.45, 1.46] + [1.3] * 10 + [np.nan, 1.3]
    masked = flood.mask_stuck(pd.Series(values, dtype=float))
    assert masked.iloc[:2].tolist() == [1.5, 1.6]
    assert masked.iloc[2:202].isna().all()  # 200 identical hours: stuck
    assert masked.iloc[202:204].tolist() == [1.45, 1.46]
    assert masked.iloc[204:214].tolist() == [1.3] * 10  # a short flat spell is real
