"""Earthquake -> nearby services -> alert chain, with every outbound service stubbed."""

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from ndp.config import get_settings
from ndp.main import create_app
from ndp.routers import earthquake as eq_router
from ndp.services import alerting, geo
from ndp.services.geo import NearbyPlace, Place

HX = {"HX-Request": "true"}
PLACES = [
    NearbyPlace("City Hospital", "Hospital", 28.61, 77.21, "Main Rd", "+91 11 2222"),
    NearbyPlace("Civil Lines Police Station", "Police", 28.62, 77.22, "", ""),
]


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setenv(f"NDP_{key.upper()}", str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(geo, "reverse_geocode", lambda lat, lon: Place("Delhi, India", lat, lon))
    monkeypatch.setattr(geo, "nearby_services", lambda lat, lon, radius, limit=40: PLACES)
    return TestClient(create_app())


def test_message_lists_facilities_and_disclaimer():
    body = alerting.build_earthquake_message(
        "Delhi", 28.6, 77.2, 6.43, 10.0, dt.date(2026, 10, 25), PLACES
    )
    assert "magnitude 6.4 Mw" in body and "depth 10 km" in body and "2026-10-25" in body
    assert "City Hospital (+91 11 2222)" in body
    assert "Civil Lines Police Station" in body
    assert "not an official warning" in body


def test_dispatch_reports_unconfigured_channels(env):
    env(alerts_dry_run="false")
    out = alerting.dispatch("Delhi", "body")
    assert out.channels == [] and out.would_send == []
    assert "email not configured" in out.skipped and "SMS not configured" in out.skipped


def test_dispatch_dry_run_sends_nothing(env, monkeypatch):
    env(
        alerts_dry_run="true",
        smtp_user="u@example.com",
        smtp_password="x",
        alert_to_email="to@example.com",
    )

    def boom(*a, **k):
        raise AssertionError("email must not be sent in dry-run mode")

    monkeypatch.setattr(alerting.notify, "send_email", boom)
    out = alerting.dispatch("Delhi", "body")
    assert out.dry_run and out.channels == []
    assert out.would_send == ["email to to@example.com"]


def test_nearby_returns_results_with_prefilled_alert(client):
    resp = client.post(
        "/alerts/nearby",
        headers=HX,
        data={
            "lat": "28.6",
            "lon": "77.2",
            "magnitude": "6.5",
            "depth": "12",
            "date": "2026-10-25",
        },
    )
    assert resp.status_code == 200
    html = resp.text
    assert "City Hospital" in html and "Civil Lines Police Station" in html
    assert "EARTHQUAKE ALERT: estimated magnitude 6.5" in html
    assert 'hx-confirm="Send this alert' in html
    assert 'data-voice-command="send alert"' in html and "data-voice-confirm" in html


def test_nearby_validates_coordinates(client):
    resp = client.post("/alerts/nearby", headers=HX, data={"lat": "123", "lon": "0"})
    assert resp.status_code == 200 and "at most 90" in resp.text


def test_send_in_dry_run_says_so(client, env):
    env(alerts_dry_run="true")
    resp = client.post(
        "/alerts/send", headers=HX, data={"place": "Delhi", "message": "Test alert body"}
    )
    assert resp.status_code == 200
    assert "Dry run, nothing was sent" in resp.text


def test_earthquake_router_has_no_model_based_auto_alert():
    """The 2020 behaviour (alert whenever a model estimates M >= 6) is gone: alerts are tied to
    real USGS events (the watcher) or to a person pressing the button."""
    from ndp.routers import earthquake as eq_router

    assert not hasattr(eq_router, "_auto_alert")
    assert not hasattr(eq_router, "BackgroundTasks")


def test_hazard_result_offers_nearby_help_without_sending(client, monkeypatch, env):
    from tests.factories import make_hazard

    env(auto_alert="true", alerts_dry_run="false")  # even if the old flag is set: nothing is sent
    monkeypatch.setattr(
        eq_router.hazard, "predict", lambda lat, lon, radius_km=300.0: make_hazard()
    )
    sent = []
    monkeypatch.setattr(alerting, "dispatch", lambda *a, **k: sent.append(a))
    html = client.post(
        "/earthquake/predict", headers=HX, data={"lat": "35.7", "lon": "139.7", "radius_km": "300"}
    ).text
    assert 'data-voice-command="find nearby help"' in html
    assert 'name="magnitude"' not in html and "Automatic alert" not in html
    assert sent == []


def test_nearby_help_from_a_hazard_result_works_without_magnitude(client):
    resp = client.post(
        "/alerts/nearby", headers=HX, data={"lat": "35.7", "lon": "139.7", "radius_km": "30"}
    )
    assert resp.status_code == 200
    assert "City Hospital" in resp.text and "EARTHQUAKE ALERT" not in resp.text


def test_live_events_panel_feeds_real_magnitudes_to_nearby_help(client, monkeypatch):
    from ndp.services import usgs

    quake = usgs.QuakeEvent(
        "us7000test",
        6.8,
        "55 km W of Somewhere",
        -3.2,
        100.4,
        24.0,
        dt.datetime(2026, 10, 24, 3, 15, tzinfo=dt.UTC),
        "https://earthquake.usgs.gov/x",
    )
    monkeypatch.setattr(usgs, "recent_events", lambda limit=8: [quake])
    html = client.get("/earthquake/events").text
    assert "55 km W of Somewhere" in html and "M 6.8" in html
    assert 'name="magnitude" value="6.8"' in html and 'name="lat" value="-3.2"' in html

    resp = client.post(
        "/alerts/nearby",
        headers=HX,
        data={
            "lat": "-3.2",
            "lon": "100.4",
            "depth": "24.0",
            "date": "2026-10-24",
            "magnitude": "6.8",
        },
    )
    assert "estimated magnitude 6.8 Mw" in resp.text


def test_live_events_panel_degrades_when_usgs_is_down(client, monkeypatch):
    from ndp.services import ServiceError, usgs

    def down(limit=8):
        raise ServiceError("The USGS feed is unavailable right now (timeout).")

    monkeypatch.setattr(usgs, "recent_events", down)
    assert "unavailable right now" in client.get("/earthquake/events").text


def test_failed_facility_lookup_still_offers_the_alert_form(client, monkeypatch):
    def down(*a, **k):
        raise geo.ServiceError("search services are busy")

    monkeypatch.setattr(geo, "nearby_services", down)
    html = client.post(
        "/alerts/nearby", headers=HX, data={"lat": "28", "lon": "139.9", "magnitude": "7.0"}
    ).text
    assert "search services are busy" in html
    assert "EARTHQUAKE ALERT: estimated magnitude 7.0" in html
    assert 'data-voice-command="send alert"' in html


def test_facilities_are_looked_up_only_once_per_request(client, monkeypatch):
    calls = []

    def counting(lat, lon, radius, limit=40):
        calls.append(1)
        return PLACES

    monkeypatch.setattr(geo, "nearby_services", counting)
    client.post("/alerts/nearby", headers=HX, data={"lat": "1", "lon": "2", "magnitude": "6.1"})
    assert len(calls) == 1
