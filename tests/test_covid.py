import pytest
from fastapi.testclient import TestClient

from ndp.config import Settings, get_settings
from ndp.main import create_app
from ndp.ml import covid
from ndp.services import ServiceError, covidstats

HX = {"HX-Request": "true"}


def test_covid_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("NDP_ENABLE_COVID", raising=False)
    assert Settings(_env_file=None).enable_covid is True


def test_flag_off_hides_the_module(monkeypatch):
    monkeypatch.setenv("NDP_ENABLE_COVID", "false")
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        assert client.get("/covid").status_code == 404
        assert "COVID-19" not in client.get("/").text
    finally:
        get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(create_app())


def test_nearest_centres_are_sorted_by_distance(monkeypatch):
    monkeypatch.setattr(
        covid,
        "test_centres",
        lambda: [
            {"name": "far", "address": "", "city": "B", "lat": 19.0, "lon": 72.8},
            {"name": "near", "address": "", "city": "A", "lat": 12.9, "lon": 79.1},
        ],
    )
    ranked = covid.nearest_centres(12.97, 79.16, n=2)
    assert [c["name"] for c in ranked] == ["near", "far"]
    assert ranked[0]["distance_km"] < 20 < ranked[1]["distance_km"]


def test_centres_endpoint_by_coordinates(client, monkeypatch):
    monkeypatch.setattr(
        covid,
        "nearest_centres",
        lambda lat, lon, n=5: [
            {
                "name": "CMC Vellore",
                "address": "Ida Scudder Rd",
                "city": "Vellore",
                "lat": 12.92,
                "lon": 79.13,
                "distance_km": 5.2,
            }
        ],
    )
    html = client.post("/covid/centres", headers=HX, data={"lat": "12.97", "lon": "79.16"}).text
    assert "CMC Vellore" in html and "5.2 km" in html


def test_centres_endpoint_validation(client):
    html = client.post("/covid/centres", headers=HX, data={"lat": "abc", "lon": "1"}).text
    assert "must be a number" in html


def test_stats_endpoint_renders_and_degrades(client, monkeypatch):
    import datetime as dt

    def totals(label):
        return covidstats.Totals(
            label,
            1_000_000,
            10,
            20_000,
            1,
            900_000,
            80_000,
            5_000_000,
            dt.datetime(2026, 10, 25, tzinfo=dt.UTC),
        )

    monkeypatch.setattr(covidstats, "world", lambda: totals("World"))
    monkeypatch.setattr(covidstats, "india", lambda: totals("India"))
    html = client.get("/covid/stats").text
    assert "1,000,000" in html and "India" in html and "disease.sh" in html

    def down():
        raise ServiceError("Live COVID-19 statistics are unavailable right now (timeout).")

    monkeypatch.setattr(covidstats, "world", down)
    assert "unavailable right now" in client.get("/covid/stats").text


def test_covid_page_survives_missing_datasets(client, monkeypatch):
    from ndp.ml.features import DataMissing

    def missing():
        raise DataMissing("TestCentres_with_geolocation.csv")

    monkeypatch.setattr(covid, "test_centres", missing)
    assert client.get("/covid").status_code == 200
    assert client.get("/covid/data/centres").status_code == 503
