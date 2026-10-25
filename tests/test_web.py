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
    assert client.get("/static/app.js").status_code == 200


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


@pytest.mark.parametrize(
    "path,data",
    [
        ("/flood/predict", FLOOD_FORM),
        ("/earthquake/predict", {"lat": "35.7", "lon": "139.7", "radius_km": "300"}),
    ],
)
def test_untrained_model_gives_503_page(client, path, data):
    resp = client.post(path, data=data, headers={"HX-Request": "true"})
    # Either a trained model answered (200) or the typed not-trained page (503); never a 500.
    assert resp.status_code in (200, 503)


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
