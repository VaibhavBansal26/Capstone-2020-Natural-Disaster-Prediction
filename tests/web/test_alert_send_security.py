"""POST /alerts/send is reachable by anyone who can open the site: it must be rate limited,
can require a passcode, and must not let a web form pick the SMS recipient by default."""

import pytest
from fastapi.testclient import TestClient

from ndp.config import get_settings
from ndp.main import create_app
from ndp.routers import alerts as alerts_router
from ndp.services import alerting

HX = {"HX-Request": "true"}
FORM = {"place": "Kochi", "message": "Test alert body"}


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setenv(f"NDP_{key.upper()}", str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


@pytest.fixture
def sent(monkeypatch):
    """Capture what reaches the dispatcher instead of sending."""
    calls = []

    def fake(place_name, body, phone=None, sms_body=None):
        calls.append({"place": place_name, "phone": phone})
        return alerting.DispatchResult(channels=["email to ops@example.com"])

    monkeypatch.setattr(alerting, "dispatch", fake)
    return calls


@pytest.fixture
def client():
    return TestClient(create_app())


def test_sixth_send_from_one_address_is_refused(client, sent):
    for _ in range(alerts_router.SEND_LIMIT):
        assert "Too many alerts" not in client.post("/alerts/send", data=FORM, headers=HX).text
    refused = client.post("/alerts/send", data=FORM, headers=HX).text
    assert "Too many alerts from this address" in refused
    assert len(sent) == alerts_router.SEND_LIMIT  # the refused one never reached the dispatcher


def test_passcode_is_required_when_configured(client, sent, env):
    env(alerts_passcode="s3cret")
    assert "passcode is missing or wrong" in client.post("/alerts/send", data=FORM, headers=HX).text
    wrong = {**FORM, "passcode": "nope"}
    assert (
        "passcode is missing or wrong" in client.post("/alerts/send", data=wrong, headers=HX).text
    )
    assert sent == []
    right = {**FORM, "passcode": "s3cret"}
    assert "Alert sent" in client.post("/alerts/send", data=right, headers=HX).text
    assert len(sent) == 1


def test_no_passcode_field_when_not_configured(client):
    html = client.get("/alerts").text
    assert 'name="passcode"' not in html


def test_custom_phone_is_ignored_by_default_and_says_so(client, sent):
    html = client.post("/alerts/send", data={**FORM, "phone": "+15551234567"}, headers=HX).text
    assert sent[0]["phone"] is None
    assert "custom phone number ignored" in html


def test_custom_phone_is_passed_through_only_when_enabled_and_valid(client, sent, env):
    env(alerts_allow_custom_phone="true")
    client.post("/alerts/send", data={**FORM, "phone": "+91 98765 43210"}, headers=HX)
    assert sent[0]["phone"] == "+91 98765 43210"
    bad = client.post("/alerts/send", data={**FORM, "phone": "<script>1</script>"}, headers=HX).text
    assert "does not look valid" in bad
    assert len(sent) == 1


def test_send_form_shows_passcode_and_phone_fields_only_when_enabled(client, env, monkeypatch):
    from ndp.services import geo

    monkeypatch.setattr(geo, "reverse_geocode", lambda lat, lon: geo.Place("Kochi", lat, lon))
    monkeypatch.setattr(geo, "nearby_services", lambda *a, **k: [])
    data = {"lat": "9.9", "lon": "76.2"}
    plain = client.post("/alerts/nearby", data=data, headers=HX).text
    assert 'name="passcode"' not in plain and 'name="phone"' not in plain
    env(alerts_passcode="x", alerts_allow_custom_phone="true")
    full = client.post("/alerts/nearby", data=data, headers=HX).text
    assert 'name="passcode"' in full and 'name="phone"' in full
