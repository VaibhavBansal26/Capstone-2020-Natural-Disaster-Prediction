"""Endpoints that spend the owner's Claude budget are capped per client address."""

import pytest
from fastapi.testclient import TestClient

from ndp.main import create_app
from ndp.routers import alerts as alerts_router
from ndp.routers import sentiment as sentiment_router
from ndp.services import ServiceError, llm, translate, triage

HX = {"HX-Request": "true"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    return TestClient(create_app())


def test_single_post_triage_is_capped(client, monkeypatch):
    calls = []

    def fake(text):
        calls.append(text)
        raise ServiceError("stubbed")

    monkeypatch.setattr(triage, "triage_post", fake)
    limit = sentiment_router.triage_limiter.limit
    for _ in range(limit):
        assert (
            "Too many requests"
            not in client.post(
                "/sentiment/triage", data={"text": "help me please now"}, headers=HX
            ).text
        )
    assert (
        "Too many requests from this address"
        in client.post("/sentiment/triage", data={"text": "help me please now"}, headers=HX).text
    )
    assert len(calls) == limit


def test_sample_triage_is_capped_tighter_because_it_makes_five_calls(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        triage,
        "triage_sample",
        lambda slug, n: calls.append(slug) or (_ for _ in ()).throw(ServiceError("stubbed")),
    )
    limit = sentiment_router.sample_limiter.limit
    assert limit < sentiment_router.triage_limiter.limit
    for _ in range(limit):
        client.post("/sentiment/triage/sample", data={"dataset": "chennai-2015-flood"}, headers=HX)
    html = client.post(
        "/sentiment/triage/sample", data={"dataset": "chennai-2015-flood"}, headers=HX
    ).text
    assert "Too many requests" in html and len(calls) == limit


def test_translation_preview_is_capped(client, monkeypatch):
    calls = []

    def fake(message, languages):
        calls.append(languages)
        return {}

    monkeypatch.setattr(translate, "translate_many", fake)
    data = {"message": "Test alert", "languages": ["hi"]}
    for _ in range(alerts_router.translate_limiter.limit):
        client.post("/alerts/translate", data=data, headers=HX)
    html = client.post("/alerts/translate", data=data, headers=HX).text
    assert "Too many translation requests" in html
    assert len(calls) == alerts_router.translate_limiter.limit


def test_usgs_source_gives_reported_wording_through_the_real_form_fields(client, monkeypatch):
    from ndp.services import geo

    monkeypatch.setattr(
        geo, "reverse_geocode", lambda lat, lon: geo.Place("Nagqu, China", lat, lon)
    )
    monkeypatch.setattr(geo, "nearby_services", lambda *a, **k: [])
    base = {"lat": "32.38", "lon": "93.78", "depth": "18", "date": "2026-10-22", "magnitude": "7.0"}
    reported = client.post("/alerts/nearby", data={**base, "source": "usgs"}, headers=HX).text
    assert "USGS-reported magnitude 7.0 Mw" in reported
    estimated = client.post("/alerts/nearby", data=base, headers=HX).text
    assert "estimated magnitude 7.0 Mw" in estimated
