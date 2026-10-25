"""GDACS, Open-Meteo and the /live router, with every outbound request stubbed.

The fixtures under tests/fixtures are trimmed copies of real responses from the live services
(fetched 2026-10-25), so the parsers are tested against the real payload shape."""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ndp.config import get_settings
from ndp.main import create_app
from ndp.routers import live as live_router
from ndp.services import ServiceError, gdacs, geo, openmeteo, watcher

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HX = {"HX-Request": "true"}


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


ORANGE_RED = load("gdacs_orange_red.json")
GREEN = load("gdacs_green.json")
MUMBAI = load("openmeteo_mumbai.json")
# Fixture was fetched on 2026-10-25 for Mumbai (UTC+5:30); this is 05:30 local.
NOW = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)


_REAL_CLIENT = httpx.Client


def patch_http(monkeypatch, module, handler):
    """Route the module's httpx.Client through a MockTransport; returns the request log."""
    real = _REAL_CLIENT
    calls: list[httpx.Request] = []

    def logged(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kw: real(transport=httpx.MockTransport(logged), **kw),
    )
    return calls


def gdacs_handler(request: httpx.Request) -> httpx.Response:
    levels = request.url.params.get("alertlevel", "")
    return httpx.Response(200, json=ORANGE_RED if "Orange" in levels else GREEN)


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    gdacs.clear_cache()
    openmeteo.clear_cache()
    live_router._geocode_cache.clear()
    yield
    gdacs.clear_cache()
    openmeteo.clear_cache()
    live_router._geocode_cache.clear()


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setenv(f"NDP_{key.upper()}", str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


# ----------------------------------------------------------------------------------- GDACS


def test_recent_alerts_parses_real_payload_and_sorts(monkeypatch):
    patch_http(monkeypatch, gdacs, gdacs_handler)
    events = gdacs.recent_alerts(30)
    levels = [e.alert_level for e in events]
    assert levels == sorted(levels, key=["Red", "Orange", "Green"].index)
    assert levels.count("Red") == 2 and levels.count("Orange") == 5 and levels.count("Green") == 5
    # Within a level the most recent activity comes first.
    reds = [e for e in events if e.alert_level == "Red"]
    assert [e.kind for e in reds] == ["TC", "EQ"]
    oranges = [e.id for e in events if e.alert_level == "Orange"]
    assert oranges[0].startswith("DR-")  # 2026-10-23 is the newest Orange activity

    cyclone = events[0]
    assert cyclone.kind_name == "Tropical cyclone"
    assert cyclone.name == "Tropical Cyclone CHAMPI-26"
    assert (cyclone.lat, cyclone.lon) == (30.7, 126.3)
    assert cyclone.country.startswith("Philippines")
    assert cyclone.from_date == datetime(2026, 10, 14, 12, tzinfo=UTC)
    assert cyclone.url.startswith("https://www.gdacs.org/report.aspx")
    assert "296 km/h" in cyclone.summary


def test_flood_summary_skips_meaningless_magnitude_zero(monkeypatch):
    patch_http(monkeypatch, gdacs, gdacs_handler)
    flood = next(e for e in gdacs.recent_alerts(30) if e.kind == "FL")
    assert flood.summary == "Flood in India"


def test_limit_applies_after_sorting(monkeypatch):
    patch_http(monkeypatch, gdacs, gdacs_handler)
    top = gdacs.recent_alerts(3)
    assert [e.alert_level for e in top] == ["Red", "Red", "Orange"]
    assert gdacs.recent_alerts(0) == []


def test_requests_use_the_verified_endpoint_and_parameters(monkeypatch):
    calls = patch_http(monkeypatch, gdacs, gdacs_handler)
    gdacs.recent_alerts()
    assert len(calls) == 2  # Orange;Red and Green are separate, so Green cannot crowd them out
    first, second = (c.url for c in calls)
    assert str(first).startswith(gdacs.EVENT_LIST_URL)
    assert first.params["eventlist"] == "EQ;TC;FL;VO;DR;WF"
    assert first.params["alertlevel"] == "Orange;Red" and second.params["alertlevel"] == "Green"
    assert first.params["fromdate"] < first.params["todate"]
    assert first.params["pagenumber"] == "1"


def test_full_page_triggers_next_page(monkeypatch):
    feature = ORANGE_RED["features"][0]
    pages = {"1": {"features": [feature] * 100}, "2": {"features": [feature] * 4}}

    def handler(request):
        if "Orange" not in request.url.params["alertlevel"]:
            return httpx.Response(200, json=GREEN)
        return httpx.Response(200, json=pages[request.url.params["pagenumber"]])

    calls = patch_http(monkeypatch, gdacs, handler)
    events = gdacs.recent_alerts(1000)
    assert [c.url.params["pagenumber"] for c in calls[:2]] == ["1", "2"]
    assert len({e.id for e in events}) == len(events)  # duplicates collapse by id


def test_cache_for_ten_minutes(monkeypatch):
    calls = patch_http(monkeypatch, gdacs, gdacs_handler)
    now = [1000.0]
    monkeypatch.setattr(gdacs, "_clock", lambda: now[0])
    gdacs.recent_alerts()
    now[0] += 599
    gdacs.recent_alerts(5)
    assert len(calls) == 2
    now[0] += 2
    gdacs.recent_alerts()
    assert len(calls) == 4


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="<html>maintenance</html>"),
        httpx.Response(200, json={"unexpected": True}),
    ],
    ids=["http-500", "not-json", "no-features"],
)
def test_bad_upstream_becomes_service_error(monkeypatch, response):
    patch_http(monkeypatch, gdacs, lambda request: response)
    with pytest.raises(ServiceError, match="GDACS"):
        gdacs.recent_alerts()


def test_network_failure_becomes_service_error(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    patch_http(monkeypatch, gdacs, boom)
    with pytest.raises(ServiceError, match="unavailable"):
        gdacs.recent_alerts()


def test_green_failure_still_returns_orange_and_red_and_retries_soon(monkeypatch):
    def handler(request):
        if "Orange" in request.url.params["alertlevel"]:
            return httpx.Response(200, json=ORANGE_RED)
        return httpx.Response(503)

    calls = patch_http(monkeypatch, gdacs, handler)
    now = [0.0]
    monkeypatch.setattr(gdacs, "_clock", lambda: now[0])
    events = gdacs.recent_alerts()
    assert {e.alert_level for e in events} == {"Red", "Orange"}
    n = len(calls)
    now[0] += 61  # a partial answer is cached for a minute only
    gdacs.recent_alerts()
    assert len(calls) > n


def feature(geometry, **props):
    base = {"eventtype": "EQ", "eventid": 1, "alertlevel": "Green", "name": "Test"}
    return {"type": "Feature", "geometry": geometry, "properties": {**base, **props}}


def test_parse_defensively_handles_geometry_variants():
    point = gdacs.parse_event(feature({"type": "Point", "coordinates": [10.5, -20.25]}))
    assert (point.lat, point.lon) == (-20.25, 10.5)

    null = gdacs.parse_event(feature(None))
    assert null is not None and null.lat is None and null.lon is None

    poly = gdacs.parse_event(
        feature(
            {"type": "Polygon", "coordinates": [[[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]]]}
        )
    )
    assert (poly.lat, poly.lon) == (5.0, 15.0)

    multi = gdacs.parse_event(
        feature(
            {
                "type": "MultiPolygon",
                "coordinates": [[[[0, 0], [2, 0], [2, 2], [0, 0]]], [[[8, 8], [10, 8], [10, 10]]]],
            }
        )
    )
    assert (multi.lat, multi.lon) == (5.0, 5.0)

    collection = gdacs.parse_event(
        feature(
            {
                "type": "GeometryCollection",
                "geometries": [{"type": "Point", "coordinates": [30, 40]}, None],
            }
        )
    )
    assert (collection.lat, collection.lon) == (40.0, 30.0)

    bbox_only = feature(None)
    bbox_only["bbox"] = [0, 0, 10, 20]
    assert gdacs.representative_point(bbox_only) == (10.0, 5.0)

    out_of_range = gdacs.parse_event(feature({"type": "Point", "coordinates": [500, 500]}))
    assert out_of_range.lat is None


def test_parse_defensively_handles_missing_fields():
    assert gdacs.parse_event(None) is None
    assert gdacs.parse_event({"properties": None}) is None
    assert gdacs.parse_event(feature(None, eventtype="XX")) is None  # unknown hazard type
    assert gdacs.parse_event(feature(None, alertlevel="Purple")) is None
    assert gdacs.parse_event(feature(None, eventid=None)) is None

    bare = gdacs.parse_event(
        {"geometry": None, "properties": {"eventtype": "fl", "eventid": 7, "alertlevel": "orange"}}
    )
    assert bare.id == "FL-7" and bare.alert_level == "Orange"
    assert bare.name == "Flood" and bare.summary == "Flood"
    assert bare.from_date is None and bare.to_date is None and bare.url == ""

    odd = gdacs.parse_event(
        feature(
            None,
            fromdate="not a date",
            url={"report": "javascript:alert(1)"},
            country="",
            affectedcountries=[{"countryname": "Chile"}, "junk", {"countryname": "Peru"}],
            severitydata="nope",
        )
    )
    assert odd.from_date is None
    assert odd.url == ""  # only https report links are kept
    assert odd.country == "Chile, Peru"


def test_parse_events_skips_bad_features_but_rejects_bad_payloads():
    payload = {"features": [feature(None), "garbage", feature(None, eventid=2)]}
    assert len(gdacs.parse_events(payload)) == 2
    with pytest.raises(ValueError):
        gdacs.parse_events([])


# ------------------------------------------------------------------------------ Open-Meteo


def meteo_payload(precip, gusts=None, temps=None, start="2026-10-25T00:00", offset=19800):
    n = len(precip)
    t0 = datetime.fromisoformat(start)
    return {
        "latitude": 19.0,
        "longitude": 72.8,
        "utc_offset_seconds": offset,
        "timezone": "Asia/Kolkata",
        "hourly": {
            "time": [(t0 + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(n)],
            "precipitation": precip,
            "wind_gusts_10m": gusts if gusts is not None else [10.0] * n,
            "temperature_2m": temps if temps is not None else [28.0] * n,
        },
    }


def test_forecast_parses_real_payload_from_the_current_hour(monkeypatch):
    calls = patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(200, json=MUMBAI))
    fc = openmeteo.forecast(19.076, 72.8777, days=3, now=NOW)
    req = calls[0].url
    assert str(req).startswith(openmeteo.FORECAST_URL)
    assert req.params["hourly"] == "precipitation,wind_gusts_10m,temperature_2m"
    assert req.params["timezone"] == "auto" and req.params["forecast_days"] == "4"

    # 00:00 UTC is 05:30 in Mumbai, so the window starts at 05:00 local and the 5 earlier
    # hours of the day are dropped.
    assert fc.hours[0].time == datetime(2026, 10, 25, 5, 0)
    assert len(fc.hours) == 67
    assert fc.timezone == "Asia/Kolkata"
    assert fc.max_gust_kmh == max(MUMBAI["hourly"]["wind_gusts_10m"][5:])
    assert fc.precip_24h_max_mm == 0.0 and fc.precip_72h_total_mm == 0.0
    assert [a.level for a in fc.advisories] == ["info"]
    assert "not an official warning" in fc.disclaimer
    assert [d.day.day for d in fc.daily] == [25, 26, 27, 28][: len(fc.daily)]
    assert fc.temp_min_c <= fc.temp_max_c


def test_forecast_days_are_clamped(monkeypatch):
    calls = patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(200, json=MUMBAI))
    openmeteo.forecast(19.0, 72.8, days=99, now=NOW)
    assert calls[0].url.params["forecast_days"] == str(openmeteo.MAX_DAYS + 1)


def test_derived_stats_use_rolling_24h_window_and_ignore_nulls(monkeypatch):
    precip = [0.0] * 10 + [10.0] * 6 + [None] + [5.0] * 8 + [0.5] * 40  # 65 hours
    gusts = [20.0] * 64 + [None]
    payload = meteo_payload(precip, gusts)
    patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(200, json=payload))
    fc = openmeteo.forecast(19.0, 72.8, days=3, now=datetime(2026, 10, 24, 18, 30, tzinfo=UTC))
    # now = 00:00 local, the window is everything
    assert len(fc.hours) == 65
    # best day: the six 10 mm hours, the null hour (counts 0), eight 5 mm hours, nine 0.5 mm hours
    assert fc.precip_24h_max_mm == 104.5
    assert fc.precip_72h_total_mm == 120.0
    assert fc.max_gust_kmh == 20.0
    assert fc.hours[16].precip_mm is None


def test_rolling_max():
    assert openmeteo.rolling_max([]) == 0.0
    assert openmeteo.rolling_max([1.0, None, 2.0]) == 3.0  # shorter than a day: total
    series = [0.0] * 30 + [10.0] * 24 + [0.0] * 30
    assert openmeteo.rolling_max(series, 24) == 240.0
    assert openmeteo.rolling_max([1.0] * 100, 24) == 24.0


@pytest.mark.parametrize(
    ("rain", "text", "level"),
    [
        (64.4, "No heavy rainfall", "info"),
        (64.5, "Heavy rainfall class", "watch"),
        (115.5, "Heavy rainfall class", "watch"),
        (115.6, "Very heavy rainfall class", "warning"),
        (204.4, "Very heavy rainfall class", "warning"),
        (204.5, "Extremely heavy rainfall class", "warning"),
        (500.0, "Extremely heavy rainfall class", "warning"),
    ],
)
def test_rainfall_class_boundaries(rain, text, level):
    [advisory] = openmeteo.assess(rain, 10.0, 72)
    assert advisory.level == level and text in advisory.text


@pytest.mark.parametrize(
    ("gust", "alarmed"), [(62.9, False), (63.0, True), (120.0, True), (None, False)]
)
def test_gust_boundary(gust, alarmed):
    advisories = openmeteo.assess(0.0, gust, 72)
    has_gust = any("gusts" in a.text and a.level == "watch" for a in advisories)
    assert has_gust is alarmed
    if not alarmed:
        assert [a.level for a in advisories] == ["info"]


def test_rain_and_gust_advisories_combine():
    levels = [a.level for a in openmeteo.assess(130.0, 80.0, 72)]
    assert levels == ["warning", "watch"]


def test_rain_totals_are_rounded_to_the_gauge_resolution():
    # 24 x 0.7 is 16.799999999999997 in floating point; both code paths must give 16.8.
    assert openmeteo.rolling_max([0.7] * 24) == 16.8
    assert openmeteo.rolling_max([0.7] * 40) == 16.8


def test_forecast_errors(monkeypatch):
    patch_http(
        monkeypatch,
        openmeteo,
        lambda r: httpx.Response(400, json={"error": True, "reason": "Latitude must be in range"}),
    )
    with pytest.raises(ServiceError, match="Latitude must be in range"):
        openmeteo.forecast(19.0, 72.8, now=NOW)

    patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(502))
    with pytest.raises(ServiceError, match="unavailable"):
        openmeteo.forecast(19.0, 72.8, now=NOW)

    patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(200, json={"hourly": {}}))
    with pytest.raises(ServiceError, match="unavailable"):
        openmeteo.forecast(19.0, 72.8, now=NOW)

    # forecast hours all in the past of "now": nothing to show
    far = datetime(2030, 1, 1, tzinfo=UTC)
    patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(200, json=MUMBAI))
    with pytest.raises(ServiceError):
        openmeteo.forecast(19.1, 72.9, now=far)

    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    patch_http(monkeypatch, openmeteo, boom)
    with pytest.raises(ServiceError):
        openmeteo.forecast(20.0, 73.0, now=NOW)

    with pytest.raises(ServiceError, match="Latitude"):
        openmeteo.forecast(91.0, 0.0)


def test_forecast_cache_is_per_rounded_location_for_fifteen_minutes(monkeypatch):
    calls = patch_http(monkeypatch, openmeteo, lambda r: httpx.Response(200, json=MUMBAI))
    now = [0.0]
    monkeypatch.setattr(openmeteo, "_clock", lambda: now[0])
    openmeteo.forecast(19.0761, 72.8771, now=NOW)
    openmeteo.forecast(19.0764, 72.8774, now=NOW)  # same 0.01 degree cell
    assert len(calls) == 1
    openmeteo.forecast(19.5, 72.8771, now=NOW)  # different place
    openmeteo.forecast(19.0761, 72.8771, days=2, now=NOW)  # different window
    assert len(calls) == 3
    now[0] = 15 * 60 + 1
    openmeteo.forecast(19.0761, 72.8771, now=NOW)
    assert len(calls) == 4


# --------------------------------------------------------------------------------- router


@pytest.fixture
def client():
    return TestClient(create_app())


def stub_events(monkeypatch):
    patch_http(monkeypatch, gdacs, gdacs_handler)


def stub_forecast(monkeypatch, name="Mumbai, Maharashtra, India"):
    monkeypatch.setattr(geo, "geocode", lambda q: geo.Place(name, 19.076, 72.8777))
    fc = openmeteo.parse_forecast(MUMBAI, 19.076, 72.8777, 3, NOW)
    monkeypatch.setattr(openmeteo, "forecast", lambda lat, lon, days=3, now=None: fc)


def test_live_page_is_a_shell_with_lazy_partials(client):
    r = client.get("/live")
    assert r.status_code == 200
    html = r.text
    assert "Global alerts (GDACS)" in html and "Weather outlook for a place" in html
    assert "Real-event alert watcher" in html
    assert 'hx-get="http://testserver/live/gdacs"' in html or 'hx-get="/live/gdacs"' in html
    assert html.count('hx-trigger="load') == 3  # gdacs, forecast form, watcher
    assert "Mumbai, India" in html


def test_gdacs_partial(client, monkeypatch):
    stub_events(monkeypatch)
    r = client.get("/live/gdacs", headers=HX)
    assert r.status_code == 200
    html = r.text
    assert "<html" not in html
    assert "Tropical Cyclone CHAMPI-26" in html and "Earthquake in China" in html
    assert 'data-speak="Red alert. Tropical Cyclone CHAMPI-26 in Philippines, Guam' in html
    assert "data-ndp-map" in html and 'data-markers-src="#gdacs-markers"' in html
    markers = json.loads(html.split('id="gdacs-markers">')[1].split("</script>")[0])
    assert len(markers) == 12  # every fixture event is a point
    assert {m["color"] for m in markers} == {"#be123c", "#ea580c", "#16a34a"}
    assert "GDACS report" in html and "GDACS's assessments" in html
    # a long country list is shortened for display
    assert re.search(r"and \d+ more", html)


def test_gdacs_partial_marks_events_without_position(client, monkeypatch):
    nowhere = gdacs.parse_event(feature(None, alertlevel="Red", name="Mystery"))
    monkeypatch.setattr(gdacs, "recent_alerts", lambda limit=30: [nowhere])
    html = client.get("/live/gdacs", headers=HX).text
    assert "Mystery" in html and "1 alert has no map position" in html
    assert json.loads(html.split('id="gdacs-markers">')[1].split("</script>")[0]) == []


def test_gdacs_partial_error_and_empty(client, monkeypatch):
    def fail(limit=30):
        raise ServiceError("The GDACS alert service is unavailable right now (timeout).")

    monkeypatch.setattr(gdacs, "recent_alerts", fail)
    r = client.get("/live/gdacs", headers=HX)
    assert r.status_code == 200
    assert 'role="alert"' in r.text and "Could not load GDACS alerts." in r.text
    assert "unavailable right now" in r.text and "Try again" in r.text

    monkeypatch.setattr(gdacs, "recent_alerts", lambda limit=30: [])
    assert "no events in the last 30 days" in client.get("/live/gdacs", headers=HX).text


def test_forecast_partial(client, monkeypatch):
    stub_forecast(monkeypatch)
    r = client.post("/live/forecast", data={"place": "Mumbai"}, headers=HX)
    assert r.status_code == 200
    html = r.text
    assert "<html" not in html
    assert "Outlook for Mumbai, Maharashtra, India" in html
    assert "Wettest 24 h" in html and "Peak gust" in html
    assert "not an official warning" in html and "Open-Meteo.com" in html
    assert "No heavy rainfall" in html
    assert 'data-speak="Weather outlook for Mumbai' in html
    assert "The model forecasts no rain in this period." in html  # the real fixture is dry
    assert html.count("title=") >= 67 * 2  # one tooltip per hourly bar in both charts


def test_forecast_partial_shows_advisories(client, monkeypatch):
    monkeypatch.setattr(geo, "geocode", lambda q: geo.Place("Cherrapunji, India", 25.3, 91.7))
    rainy = meteo_payload([4.0] * 30 + [0.0] * 40, [70.0] + [10.0] * 69)
    fc = openmeteo.parse_forecast(rainy, 25.3, 91.7, 3, datetime(2026, 10, 24, 18, 30, tzinfo=UTC))
    monkeypatch.setattr(openmeteo, "forecast", lambda lat, lon, days=3, now=None: fc)
    html = client.post("/live/forecast", data={"place": "Cherrapunji"}, headers=HX).text
    assert "Heavy rainfall class" in html and "Gale-strength gusts" in html
    assert "bg-amber-500" in html  # the over-threshold gust bar is highlighted


def test_forecast_errors_are_partials_not_500s(client, monkeypatch):
    r = client.post("/live/forecast", data={"place": "   "}, headers=HX)
    assert r.status_code == 200 and "Enter a place name" in r.text

    r = client.post("/live/forecast", data={"place": "x" * 201}, headers=HX)
    assert "limited to 200" in r.text

    def not_found(q):
        raise ServiceError("No location found for 'Atlantis'.")

    monkeypatch.setattr(geo, "geocode", not_found)
    r = client.post("/live/forecast", data={"place": "Atlantis"}, headers=HX)
    assert r.status_code == 200 and "No location found" in r.text and 'role="alert"' in r.text

    monkeypatch.setattr(geo, "geocode", lambda q: geo.Place("Mumbai", 19.0, 72.8))

    def down(lat, lon, days=3, now=None):
        raise ServiceError("The Open-Meteo forecast is unavailable right now (timeout).")

    monkeypatch.setattr(openmeteo, "forecast", down)
    r = client.post("/live/forecast", data={"place": "Mumbai"}, headers=HX)
    assert "Could not get the outlook." in r.text and "Open-Meteo forecast is unavailable" in r.text


def test_forecast_without_htmx_renders_the_full_page(client, monkeypatch):
    stub_forecast(monkeypatch)
    r = client.post("/live/forecast", data={"place": "Mumbai"})
    assert r.status_code == 200
    assert "<html" in r.text and "Global alerts (GDACS)" in r.text
    assert "Outlook for Mumbai, Maharashtra, India" in r.text


def test_forecast_escapes_place_text(client, monkeypatch):
    stub_forecast(monkeypatch, name="<script>alert(1)</script>")
    html = client.post("/live/forecast", data={"place": "<b>x"}, headers=HX).text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_geocoding_is_cached_between_requests(client, monkeypatch):
    stub_forecast(monkeypatch)
    count = []
    monkeypatch.setattr(
        geo, "geocode", lambda q: count.append(q) or geo.Place("Mumbai", 19.0, 72.8)
    )
    for _ in range(3):
        client.post("/live/forecast", data={"place": "Mumbai"}, headers=HX)
    client.post("/live/forecast", data={"place": "  mumbai "}, headers=HX)
    assert len(count) == 1


def test_watcher_partial_off(client, monkeypatch, tmp_path, env):
    monkeypatch.setattr(watcher, "STATE_PATH", tmp_path / "w.json")
    watcher.reset()
    env(watch_quakes="false")
    html = client.get("/live/watcher", headers=HX).text
    assert "NDP_WATCH_QUAKES=true" in html and "Off" in html
    assert "Nothing handled yet" in html
    assert "The real event alert watcher is off." in html


def test_watcher_partial_enabled_with_history(client, monkeypatch, tmp_path, env):
    path = tmp_path / "w.json"
    path.write_text(
        json.dumps(
            {
                "baselined": True,
                "seen": ["us1"],
                "handled": [
                    {
                        "id": "us1",
                        "magnitude": 6.7,
                        "place": "10 km S of <Town>",
                        "at": "2026-10-25T12:00:00+00:00",
                        "outcome": "dry run, would send: email to ops@example.com",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(watcher, "STATE_PATH", path)
    watcher.reset()
    env(
        watch_quakes="true",
        alerts_dry_run="true",
        watch_min_magnitude="5.5",
        watch_interval_seconds="120",
    )
    html = client.get("/live/watcher", headers=HX).text
    assert "Dry run: nothing is sent" in html and "Magnitude 5.5 or more" in html
    assert "Every 2 minutes" in html
    assert "Enabled, not running" in html  # lifespan not entered: no task
    assert "M 6.7" in html and "10 km S of &lt;Town&gt;" in html
    assert "dry run, would send" in html
    assert "NDP_WATCH_QUAKES=true" not in html
