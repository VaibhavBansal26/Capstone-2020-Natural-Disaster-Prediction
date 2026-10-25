"""Phase 2 web layer: the earthquake hazard form, the flood forecast form, the JSON API, the
honest model card and the comparison baseline. No trained models are needed: the ML entry points
are monkeypatched to return the contract dataclasses."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ndp.main import create_app
from ndp.ml import compare, flood, hazard
from ndp.routers import earthquake as eq_router
from ndp.routers import flood as flood_router
from ndp.web import templates

HX = {"HX-Request": "true"}


# ----------------------------------------------------------------------------- builders


def make_hazard(**overrides) -> hazard.HazardEstimate:
    values = dict(
        lat=35.7,
        lon=139.7,
        radius_km=300.0,
        n_events=212,
        years=52.0,
        annual_rate_m55=4.08,
        b_value=0.91,
        b_is_global=False,
        annual_rate={"6.0": 1.27, "7.0": 0.156, "8.0": 0.0192},
        prob_1y={"6.0": 0.719, "7.0": 0.144, "8.0": 0.019},
        prob_10y={"6.0": 1.0, "7.0": 0.79, "8.0": 0.175},
        max_magnitude=9.1,
        max_date="2011-03-11",
        depth_median_km=33.0,
        nearest_event={"distance_km": 18.4, "magnitude": 6.1, "year": 1987},
        level="very high",
        elevated=True,
    )
    values.update(overrides)
    return hazard.HazardEstimate(**values)


def quiet_hazard(**overrides) -> hazard.HazardEstimate:
    return make_hazard(
        lat=0.0,
        lon=-30.0,
        n_events=0,
        annual_rate_m55=0.0,
        b_value=1.05,
        b_is_global=True,
        annual_rate={"6.0": 0.0, "7.0": 0.0, "8.0": 0.0},
        prob_1y={"6.0": 0.0, "7.0": 0.0, "8.0": 0.0},
        prob_10y={"6.0": 0.0, "7.0": 0.0, "8.0": 0.0},
        max_magnitude=None,
        max_date=None,
        depth_median_km=None,
        nearest_event=None,
        level="low",
        elevated=False,
        **overrides,
    )


def make_forecast(levels=(1.43, 1.46, 1.52), now=1.42) -> flood.FloodForecast:
    horizons = [
        flood.HorizonForecast(
            hours=h,
            level_m=lv,
            is_flood=lv > flood.FLOOD_THRESHOLD_M,
            threshold_m=flood.FLOOD_THRESHOLD_M,
            persistence_level_m=now,
        )
        for h, lv in zip((1, 3, 6), levels, strict=True)
    ]
    return flood.FloodForecast(
        horizons=horizons, any_flood=any(h.is_flood for h in horizons), peak_level_m=max(levels)
    )


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def hz(monkeypatch):
    calls: list[tuple] = []

    def fake(lat, lon, radius_km=300.0):
        calls.append((lat, lon, radius_km))
        return make_hazard(lat=lat, lon=lon, radius_km=radius_km)

    monkeypatch.setattr(eq_router.hazard, "predict", fake)
    eq_router._recent.clear()
    return calls


@pytest.fixture
def fl(monkeypatch):
    calls: list[tuple] = []

    def fake(*args):
        calls.append(args)
        return make_forecast()

    monkeypatch.setattr(flood_router.flood, "predict", fake)
    return calls


EQ_FORM = {"lat": "35.7", "lon": "139.7", "radius_km": "300"}
FLOOD_FORM = {
    "level_now": "1.42",
    "level_1h_ago": "1.41",
    "level_3h_ago": "1.40",
    "rain_3h": "2",
    "rain_6h": "3",
    "rain_24h": "5",
}


# ----------------------------------------------------------------------------- earthquake


def test_earthquake_page_has_the_hazard_form(client):
    html = client.get("/earthquake").text
    for name in ("lat", "lon", "radius_km"):
        assert f'name="{name}"' in html
    assert "Radius (km)" in html and "Latitude" in html and "Longitude" in html
    assert 'value="300"' in html and 'min="25"' in html and 'max="2000"' in html
    # the old magnitude estimate is gone
    assert 'name="depth"' not in html and 'name="date"' not in html
    assert "Estimate magnitude" not in html
    # the live panels stay
    assert (
        'id="eq-map"' in html and "earthquake_events" not in html and "/earthquake/events" in html
    )


def test_earthquake_valid_post_returns_partial_with_the_hazard_card(client, hz):
    resp = client.post("/earthquake/predict", data=EQ_FORM, headers=HX)
    assert resp.status_code == 200
    html = resp.text
    assert "<html" not in html
    assert hz == [(35.7, 139.7, 300.0)]
    assert "Very high" in html and "Seismic hazard" in html
    for label in ("M ≥ 6", "M ≥ 7", "M ≥ 8", "Chance in 1 year", "Chance in 10 years"):
        assert label in html
    assert "212" in html  # events in radius
    assert "M 9.1" in html and "in 2011" in html  # largest event and its year
    assert "33 km" in html or ">33<" in html  # median depth
    assert "M 6.1 in 1987, 18 km away" in html  # nearest large event
    assert "local, b = 0.91" in html
    assert "What it does not mean" in html and "not a prediction of when" in html
    assert 'data-speak="' in html and "long-term rate" in html


def test_earthquake_post_without_htmx_returns_the_full_page(client, hz):
    resp = client.post("/earthquake/predict", data=EQ_FORM)
    assert resp.status_code == 200
    assert "<html" in resp.text and "Seismic hazard" in resp.text and 'id="eq-map"' in resp.text


def test_earthquake_radius_defaults_to_300_when_omitted(client, hz):
    client.post("/earthquake/predict", data={"lat": "10", "lon": "20"}, headers=HX)
    assert hz == [(10.0, 20.0, 300.0)]


def test_elevated_hazard_is_spoken_with_alert_priority_and_a_red_button(client, hz):
    html = client.post("/earthquake/predict", data=EQ_FORM, headers=HX).text
    assert 'data-speak-priority="alert"' in html
    assert 'data-voice-command="find nearby help"' in html
    assert "bg-rose-600" in html


def test_low_hazard_has_no_alert_priority_and_explains_empty_results(client, monkeypatch):
    eq_router._recent.clear()
    monkeypatch.setattr(
        eq_router.hazard,
        "predict",
        lambda lat, lon, radius_km=300.0: quiet_hazard(radius_km=radius_km),
    )
    html = client.post(
        "/earthquake/predict", data={"lat": "0", "lon": "-30", "radius_km": "100"}, headers=HX
    ).text
    assert "data-speak-priority" not in html
    assert "Low" in html and "does not mean the area is safe" in html
    assert "global fallback" in html
    assert 'data-voice-command="find nearby help"' in html  # still offered


def test_nearby_button_posts_coordinates_and_radius_but_no_magnitude(client, hz):
    html = client.post("/earthquake/predict", data=EQ_FORM, headers=HX).text
    assert 'hx-post="http://testserver/alerts/nearby"' in html
    assert 'name="lat" value="35.7"' in html and 'name="lon" value="139.7"' in html
    # the facility search radius is capped at 50 km by /alerts/nearby; the hazard radius is not
    assert 'name="radius_km" value="50"' in html
    assert 'name="magnitude"' not in html


@pytest.mark.parametrize(
    "data,message",
    [
        ({"lat": "999", "lon": "0", "radius_km": "300"}, "at most 90"),
        ({"lat": "0", "lon": "-181", "radius_km": "300"}, "at least -180"),
        ({"lat": "abc", "lon": "0", "radius_km": "300"}, "must be a number"),
        ({"lat": "", "lon": "0", "radius_km": "300"}, "required"),
        ({"lat": "0", "lon": "0", "radius_km": "10"}, "at least 25"),
        ({"lat": "0", "lon": "0", "radius_km": "2001"}, "at most 2000"),
        ({"lat": "0", "lon": "0", "radius_km": "nan"}, "must be a number"),
    ],
)
def test_earthquake_invalid_input_is_reported_inline(client, hz, data, message):
    resp = client.post("/earthquake/predict", data=data, headers=HX)
    assert resp.status_code == 200 and message in resp.text
    assert hz == [] and "Seismic hazard" not in resp.text


def test_earthquake_recent_table_has_the_new_columns(client, hz):
    client.post("/earthquake/predict", data=EQ_FORM, headers=HX)
    html = client.get("/earthquake").text
    for header in ("Latitude", "Longitude", "Radius", "Level", "P(M ≥ 6, 10 y)"):
        assert header in html
    assert "300 km" in html and "Very high" in html and "100.0 %" in html
    assert '<th class="px-3 py-2">Magnitude</th>' not in html


# ----------------------------------------------------------------------------- flood


def test_flood_page_has_the_six_labelled_inputs_and_an_honest_scope_note(client):
    html = client.get("/flood").text
    for name, label in {
        "level_now": "Level now (m)",
        "level_1h_ago": "Level 1 hour ago (m)",
        "level_3h_ago": "Level 3 hours ago (m)",
        "rain_3h": "Rain last 3 hours (mm)",
        "rain_6h": "Rain last 6 hours (mm)",
        "rain_24h": "Rain last 24 hours (mm)",
    }.items():
        assert f'name="{name}"' in html and label in html
    assert 'name="cumulative_rainfall"' not in html
    assert "Hoppers Crossing" in html and "specific to the Hoppers Crossing gauge" in html
    assert 'value="1.42"' in html  # a realistic default for the gauge


def test_flood_valid_post_returns_the_forecast_table(client, fl):
    resp = client.post("/flood/predict", data=FLOOD_FORM, headers=HX)
    assert resp.status_code == 200
    html = resp.text
    assert "<html" not in html
    assert fl == [(1.42, 1.41, 1.40, 2.0, 3.0, 5.0)]
    for text in ("1 hour", "3 hours", "6 hours", "Model forecast", "Persistence"):
        assert text in html
    assert "1.43 m" in html and "1.52 m" in html and "1.42 m" in html
    # No flood / no-flood claims: the 1.44 m level is only a reference on this gauge.
    assert ">Flood<" not in html and ">No flood<" not in html
    assert "Above 1.44 m reference" in html and "Rising" in html
    assert "data-speak-priority" not in html
    assert "not a flood warning" in html and "mostly misses sudden rises" in html


def test_flood_below_reference_has_no_flood_badge_or_alert_priority(client, monkeypatch):
    monkeypatch.setattr(
        flood_router.flood, "predict", lambda *a: make_forecast((1.40, 1.41, 1.42), now=1.40)
    )
    html = client.post("/flood/predict", data=FLOOD_FORM, headers=HX).text
    assert "Below 1.44 m reference" in html and ">Flood<" not in html
    assert "data-speak-priority" not in html


def test_flood_post_without_htmx_returns_the_full_page(client, fl):
    html = client.post("/flood/predict", data=FLOOD_FORM).text
    assert "<html" in html and "River level forecast" in html and 'name="level_now"' in html


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("level_now", "15.5", "at most 15"),
        ("level_1h_ago", "-0.1", "at least 0"),
        ("level_3h_ago", "abc", "must be a number"),
        ("rain_3h", "-1", "at least 0"),
        ("rain_6h", "501", "at most 500"),
        ("rain_24h", "", "required"),
    ],
)
def test_flood_invalid_input_is_reported_inline(client, fl, field, value, message):
    data = {**FLOOD_FORM, field: value}
    resp = client.post("/flood/predict", data=data, headers=HX)
    assert resp.status_code == 200 and message in resp.text
    assert fl == [] and "River level forecast" not in resp.text


def test_flood_limits_are_inclusive(client, fl):
    data = {**FLOOD_FORM, "level_now": "15", "rain_24h": "500", "rain_3h": "0"}
    assert "River level forecast" in client.post("/flood/predict", data=data, headers=HX).text


# ----------------------------------------------------------------------------- JSON API


def test_hazard_api_returns_the_estimate(client, monkeypatch):
    monkeypatch.setattr(
        hazard,
        "predict",
        lambda lat, lon, radius_km=300.0: make_hazard(lat=lat, lon=lon, radius_km=radius_km),
    )
    resp = client.post("/api/v1/earthquake/hazard", json={"lat": 35.7, "lon": 139.7})
    assert resp.status_code == 200
    body = resp.json()
    assert body["radius_km"] == 300 and body["level"] == "very high" and body["elevated"] is True
    assert set(body["prob_10y"]) == {"6.0", "7.0", "8.0"}
    assert body["nearest_event"]["year"] == 1987
    resp = client.post("/api/v1/earthquake/hazard", json={"lat": 1, "lon": 2, "radius_km": 25})
    assert resp.json()["radius_km"] == 25


@pytest.mark.parametrize(
    "body",
    [
        {"lat": 91, "lon": 0},
        {"lat": 0, "lon": 181},
        {"lat": 0, "lon": 0, "radius_km": 24},
        {"lat": 0, "lon": 0, "radius_km": 2001},
        {"lat": 0},
        {"lat": "x", "lon": 0},
    ],
)
def test_hazard_api_rejects_bad_bodies(client, body):
    assert client.post("/api/v1/earthquake/hazard", json=body).status_code == 422


def test_flood_api_returns_the_forecast(client, monkeypatch):
    monkeypatch.setattr(flood, "predict", lambda *a: make_forecast())
    body = {
        "level_now_m": 1.42,
        "level_1h_ago_m": 1.41,
        "level_3h_ago_m": 1.4,
        "rain_3h_mm": 2,
        "rain_6h_mm": 3,
        "rain_24h_mm": 5,
    }
    resp = client.post("/api/v1/flood/forecast", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert [h["hours"] for h in data["horizons"]] == [1, 3, 6]
    assert data["any_flood"] is True and data["peak_level_m"] == 1.52
    assert data["horizons"][0]["persistence_level_m"] == 1.42


@pytest.mark.parametrize(
    "change",
    [
        {"level_now_m": 15.1},
        {"level_now_m": -1},
        {"rain_3h_mm": -0.1},
        {"rain_24h_mm": 500.1},
        {"rain_6h_mm": "wet"},
    ],
)
def test_flood_api_rejects_bad_bodies(client, change):
    body = {
        "level_now_m": 1.42,
        "level_1h_ago_m": 1.41,
        "level_3h_ago_m": 1.4,
        "rain_3h_mm": 2,
        "rain_6h_mm": 3,
        "rain_24h_mm": 5,
        **change,
    }
    assert client.post("/api/v1/flood/forecast", json=body).status_code == 422
    assert client.post("/api/v1/flood/forecast", json={}).status_code == 422


def test_old_earthquake_and_flood_api_routes_are_gone(client):
    assert client.post("/api/v1/earthquake/predict", json={}).status_code == 404
    assert client.post("/api/v1/flood/predict", json={}).status_code == 404


def test_other_api_routes_are_unchanged(client):
    routes = set(client.get("/openapi.json").json()["paths"])
    for path in ("tsunami", "hurricane", "storm"):
        assert f"/api/v1/{path}/predict" in routes


# ----------------------------------------------------------------------------- model card


def render_card(meta, trained=True, **extra):
    return templates.get_template("partials/_model_card.html").render(
        meta=meta, trained=trained, **extra
    )


@pytest.mark.parametrize(
    "skill,phrase",
    [
        (-0.3, "no better than predicting the average"),
        (0.0, "no better than predicting the average"),
        (0.049, "no better than predicting the average"),
        (0.05, "weak"),
        (0.249, "weak"),
        (0.25, "moderate"),
        (0.499, "moderate"),
        (0.5, "useful"),
        (0.9, "useful"),
    ],
)
def test_model_card_verdict_bands(skill, phrase):
    html = render_card({"name": "x", "rmse": 1.0, "baseline_rmse": 1.0, "skill": skill})
    assert f"<strong>{phrase}</strong>" in html
    if phrase != "no better than predicting the average":
        assert "no better than predicting the average" not in html


def test_model_card_shows_the_honest_keys_with_plain_labels():
    meta = {
        "name": "tsunami_magnitude",
        "trained_at": "2026-10-25T10:00:00+00:00",
        "r2": 0.31,
        "rmse": 0.62,
        "baseline_rmse": 0.8,
        "skill": 0.225,
        "split": "5-fold cross-validation",
        "cv_r2_mean": 0.3,
        "cv_r2_std": 0.04,
        "n_rows": 1500,
    }
    html = render_card(meta)
    assert "Skill vs. baseline" in html and "0.23" in html
    assert "Baseline RMSE" in html and "0.80" in html
    assert "How it was tested" in html and "5-fold cross-validation" in html
    assert "Cross-validated R²" in html and "± 0.04" in html
    assert "<strong>weak</strong>" in html
    assert "cv r2 std" not in html  # folded into the cross-validated tile
    assert "1,500" in html  # existing keys still render


def test_model_card_without_skill_keeps_working_for_old_artifacts():
    html = render_card({"name": "old", "r2": 0.5, "rmse": 1.2, "n_rows": 10})
    assert "R²" in html and "RMSE" in html and "Verdict" not in html


def test_model_card_classifier_compares_with_the_majority_class():
    good = render_card({"name": "storm", "accuracy": 0.8, "baseline_accuracy": 0.4})
    assert "Beats always guessing" in good and "40.0 points" in good
    bad = render_card({"name": "storm", "accuracy": 0.41, "baseline_accuracy": 0.4})
    assert "Only" in bad and "1.0 points" in bad


def test_model_card_shows_per_horizon_flood_metrics():
    horizons = {
        str(h): {
            "rmse": 0.05 * h,
            "persistence_rmse": 0.06 * h,
            "skill_vs_persistence": 0.1,
            "r2": 0.7,
            "flood_precision": 0.8,
            "flood_recall": 0.6,
        }
        for h in (1, 3, 6)
    }
    html = render_card(
        {
            "name": "flood_level",
            "skill": 0.12,
            "baseline_rmse": 0.2,
            "baseline": "persistence (level in h hours = level now)",
            "horizons": horizons,
            "notes": ["The 1.44 m threshold sits at the median."],
        }
    )
    assert "Held-out error by forecast horizon" in html
    assert "1 h" in html and "3 h" in html and "6 h" in html
    assert "Skill vs persistence" in html and "Flood recall" in html
    assert "no better than the baseline" not in html and "<strong>weak</strong>" in html
    assert "persistence (level in h hours = level now)" in html
    assert "sits at the median" in html


def test_model_card_shows_the_hazard_backtest_and_hides_bookkeeping_keys():
    html = render_card(
        {
            "name": "earthquake_hazard",
            "n_events": 23232,
            "years_span": 52.0,
            "year_min": 1965,
            "year_max": 2016,
            "n_train": 100,
            "backtest": {
                "train_period": "before 2000",
                "test_period": "2000 onwards",
                "share_of_globe_in_active_cells": 0.31,
                "share_of_test_events_in_active_cells": 0.99,
                "lift": 3.2,
            },
        }
    )
    assert "23,232" in html and ">1965<" in html.replace("\n", "").replace(" ", "")
    assert "lift 3.2" in html and "31%" in html and "99%" in html
    assert "n train" not in html and "Verdict" not in html


def test_model_card_untrained_state():
    html = render_card(None, trained=False)
    assert "needs training" in html and "ndp-train" in html


# ----------------------------------------------------------------------------- compare


def results(model_rmse: float, base_rmse: float = 1.0):
    names = ["Random forest", "Neural network", "SVM", "Linear regression"]
    rows = [
        compare.ModelResult(n, 5.5 + i * 0.1, 70.0 - i, 0.2 - i * 0.01, model_rmse + i * 0.01)
        for i, n in enumerate(names)
    ]
    rows.append(compare.ModelResult("Baseline (predict the mean)", 5.8, 66.0, 0.0, base_rmse))
    return rows


@pytest.fixture
def compare_with(monkeypatch):
    def _set(rows):
        from ndp.routers import pages

        monkeypatch.setattr(pages.compare, "predict", lambda hazard_key, values: rows)

    return _set


def test_compare_shows_the_baseline_row_and_the_no_model_beats_it_verdict(client, compare_with):
    compare_with(results(model_rmse=0.99))
    html = client.post(
        "/compare/earthquake", data={"lat": "1", "lon": "2", "depth": "3"}, headers=HX
    ).text
    assert html.count("Baseline (predict the mean)") >= 3  # table, bar, ...
    assert "data-baseline-row" in html
    assert "No model beats the baseline" in html
    assert "data-compare-verdict" in html
    # the baseline is never crowned best
    assert "Best: Baseline" not in html and "Best: Random forest" in html
    assert "Four models and a baseline" in html


def test_compare_verdict_when_a_model_clearly_beats_the_baseline(client, compare_with):
    compare_with(results(model_rmse=0.6))
    html = client.post(
        "/compare/earthquake", data={"lat": "1", "lon": "2", "depth": "3"}, headers=HX
    ).text
    assert "No model beats the baseline" not in html
    assert "Random forest has the lowest error, 40% below the baseline" in html


@pytest.mark.parametrize("rmse,beats", [(0.979, True), (0.98, False), (1.2, False)])
def test_compare_two_percent_boundary(client, compare_with, rmse, beats):
    compare_with(results(model_rmse=rmse))
    html = client.post(
        "/compare/earthquake", data={"lat": "1", "lon": "2", "depth": "3"}, headers=HX
    ).text
    assert ("No model beats the baseline" not in html) is beats


def test_compare_without_a_baseline_row_still_renders(client, compare_with):
    compare_with(results(model_rmse=0.5)[:4])
    html = client.post(
        "/compare/earthquake", data={"lat": "1", "lon": "2", "depth": "3"}, headers=HX
    ).text
    assert "Random forest" in html and "data-compare-verdict" not in html
