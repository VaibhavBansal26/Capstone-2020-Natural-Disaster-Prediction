"""Smoke tests for every page. They do not need trained models: untrained pages render,
prediction POSTs return a friendly 'not trained' page or a validation message."""

import pytest
from fastapi.testclient import TestClient

from ndp.main import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


PAGES = [
    "/",
    "/earthquake",
    "/earthquake/graphs",
    "/tsunami",
    "/flood",
    "/hurricane",
    "/storm",
    "/weather",
    "/alerts",
    "/sentiment",
    "/compare",
]


@pytest.mark.parametrize("path", PAGES)
def test_pages_render(client, path):
    resp = client.get(path)
    assert resp.status_code == 200, resp.text[:300]
    assert "<html" in resp.text


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_static_mounted(client):
    assert client.get("/static/js/app.js").status_code == 200


@pytest.mark.parametrize(
    "path,data",
    [
        ("/flood/predict", {"level_now": "not-a-number"}),
        ("/earthquake/predict", {"lat": "999", "lon": "0", "radius_km": "300"}),
        ("/tsunami/predict", {"lat": "1", "lon": "2", "max_height": ""}),
        ("/hurricane/predict", {"date": "garbage", "lat": "1", "lon": "2"}),
        (
            "/storm/predict",
            {
                "temperature": "70",
                "pressure": "1010",
                "humidity": "150",
                "wind_direction": "90",
                "wind_speed": "3",
            },
        ),
    ],
)
def test_invalid_input_is_reported_inline(client, path, data):
    resp = client.post(path, data=data, headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "must" in resp.text or "required" in resp.text


FLOOD_FORM = {
    "level_now": "1.42",
    "level_1h_ago": "1.41",
    "level_3h_ago": "1.40",
    "rain_3h": "2",
    "rain_6h": "3",
    "rain_24h": "5",
}


@pytest.fixture
def empty_models(tmp_path, monkeypatch):
    """Other tests train models into the shared temp folder; these need a folder with none."""
    from ndp.ml import registry

    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "no-models"))
    registry.load.cache_clear()
    yield
    registry.load.cache_clear()


@pytest.mark.parametrize(
    "path,data",
    [
        ("/flood/predict", FLOOD_FORM),
        ("/earthquake/predict", {"lat": "35.7", "lon": "139.7", "radius_km": "300"}),
    ],
)
def test_untrained_model_full_page_is_a_real_503(client, empty_models, path, data):
    resp = client.post(path, data=data)
    assert resp.status_code == 503
    assert "is not trained yet" in resp.text and "ndp-train" in resp.text


@pytest.mark.parametrize(
    "path,data",
    [
        ("/flood/predict", FLOOD_FORM),
        ("/earthquake/predict", {"lat": "35.7", "lon": "139.7", "radius_km": "300"}),
    ],
)
def test_untrained_model_htmx_gets_a_swappable_notice(client, empty_models, path, data):
    # HTMX does not swap 4xx/5xx by default, so the inline notice is a 200 partial.
    resp = client.post(path, data=data, headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "<html" not in resp.text and 'role="alert"' in resp.text
    assert "is not trained yet" in resp.text


def test_untrained_model_json_api_is_503(client, empty_models):
    resp = client.post(
        "/api/v1/flood/forecast",
        json={
            "level_now_m": 1.4,
            "level_1h_ago_m": 1.4,
            "level_3h_ago_m": 1.4,
            "rain_3h_mm": 0,
            "rain_6h_mm": 0,
            "rain_24h_mm": 0,
        },
    )
    assert resp.status_code == 503 and resp.json()["model"]


def test_sentiment_on_demand_score_without_data_is_a_notice_not_a_500(client, monkeypatch):
    from ndp.ml import sentiment
    from ndp.ml.features import DataMissing
    from ndp.routers import sentiment as sentiment_router

    sentiment_router._computed.clear()

    def missing(disaster):
        raise DataMissing(disaster.csv)

    monkeypatch.setattr(sentiment, "score_dataset", missing)
    resp = client.get("/sentiment/chennai-2015-flood", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "ndp-fetch-data" in resp.text or "missing" in resp.text


def test_sentiment_on_demand_score_is_computed_once(client, monkeypatch):
    from ndp.ml import sentiment
    from ndp.routers import sentiment as sentiment_router

    sentiment_router._computed.clear()
    calls = []

    def scored(disaster):
        calls.append(disaster.slug)
        return {"score": -3.5, "n_posts": 10, "n_sentences": 40}

    monkeypatch.setattr(sentiment, "score_dataset", scored)
    for _ in range(2):
        resp = client.get("/sentiment/chennai-2015-flood", headers={"HX-Request": "true"})
        assert "-3.50" in resp.text
    assert calls == ["chennai-2015-flood"]


def test_covid_geojson_features_carry_plain_properties_not_html(client, monkeypatch):
    from ndp.ml import covid

    monkeypatch.setattr(
        covid,
        "state_snapshot",
        lambda: [
            {
                "state": "Kerala",
                "lat": 10.0,
                "lon": 76.0,
                "active": 1,
                "confirmed": 2,
                "deaths": 0,
                "recovered": 1,
            }
        ],
    )
    feature = client.get("/covid/data/states").json()["features"][0]
    assert feature["properties"]["name"] == "Kerala"
    assert "popup" not in feature["properties"]


def test_api_validation_errors_are_422(client):
    resp = client.post("/api/v1/flood/forecast", json={"level_now_m": -5})
    assert resp.status_code == 422


def test_covid_page_is_on_by_default(client):
    assert client.get("/covid").status_code == 200
    assert client.get("/covid/maps").status_code == 200


def test_earthquake_page_prefills_from_query_string(client):
    # `depth` is still accepted in old links (it is no longer a model input) and is ignored.
    html = client.get("/earthquake?lat=32.38&lon=93.78&depth=18&radius_km=500").text
    assert 'value="32.38"' in html and 'value="93.78"' in html and 'value="500"' in html


def test_earthquake_page_ignores_bad_query_values(client):
    html = client.get("/earthquake?lat=999&lon=abc&radius_km=5").text
    assert 'value="28.6"' in html and 'value="77.2"' in html and 'value="300"' in html


def test_training_plots_are_served_from_the_configured_directory(tmp_path, monkeypatch):
    """With NDP_GENERATED_DIR on a volume, /static/generated/... must come from that folder."""
    from ndp.config import get_settings
    from ndp.main import create_app

    (tmp_path / "flood").mkdir()
    (tmp_path / "flood" / "plot.png").write_bytes(b"\x89PNG-test")
    monkeypatch.setenv("NDP_GENERATED_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        c = TestClient(create_app())
        assert c.get("/static/generated/flood/plot.png").content == b"\x89PNG-test"
        assert c.get("/static/js/app.js").status_code == 200  # the packaged assets still resolve
    finally:
        get_settings.cache_clear()
