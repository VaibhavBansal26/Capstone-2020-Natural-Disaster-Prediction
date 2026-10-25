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
        ("/flood/predict", {"cumulative_rainfall": "not-a-number"}),
        ("/earthquake/predict", {"lat": "999", "lon": "0", "depth": "10", "date": "2026-01-01"}),
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


def test_untrained_model_gives_503_page(client):
    resp = client.post(
        "/flood/predict", data={"cumulative_rainfall": "12"}, headers={"HX-Request": "true"}
    )
    # Either a trained model answered (200) or the typed not-trained page (503); never a 500.
    assert resp.status_code in (200, 503)


def test_api_validation_errors_are_422(client):
    resp = client.post("/api/v1/flood/predict", json={"cumulative_rainfall_mm": -5})
    assert resp.status_code == 422


def test_covid_hidden_by_default(client):
    assert client.get("/covid").status_code == 404
