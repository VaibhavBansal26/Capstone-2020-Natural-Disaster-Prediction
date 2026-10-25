"""Assistant: tool loop (scripted fake model), tools, limits and the router. No network, no key."""

import dataclasses
import inspect
import json
import re
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ndp.config import get_settings
from ndp.main import app
from ndp.services import ServiceError, assistant, llm

# ---- a scripted fake for llm.create_message ---------------------------------------------


def text(t):
    return SimpleNamespace(type="text", text=t)


def thinking(t="hmm", sig="sig-1"):
    return SimpleNamespace(type="thinking", thinking=t, signature=sig)


def tool_use(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def reply(*blocks, stop="end_turn", usage=(10, 5, 0, 0)):
    i, o, cr, cw = usage
    u = SimpleNamespace(
        input_tokens=i, output_tokens=o, cache_read_input_tokens=cr, cache_creation_input_tokens=cw
    )
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=u)


class FakeModel:
    """Returns scripted responses in order and records a snapshot of every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, **kwargs):
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs["messages"])  # the list is mutated afterwards
        self.requests.append(snapshot)
        if not self.responses:
            raise AssertionError("fake model ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _clean_state():
    assistant.sessions.clear()
    assistant.rate_limiter.reset()
    yield
    assistant.sessions.clear()
    assistant.rate_limiter.reset()
    get_settings.cache_clear()


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("NDP_ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("NDP_ASSISTANT_MAX_TOOL_ROUNDS", "3")
    get_settings.cache_clear()


@pytest.fixture
def unconfigured(monkeypatch):
    monkeypatch.delenv("NDP_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()


@pytest.fixture
def model(monkeypatch, configured):
    def install(*responses):
        fake = FakeModel(*responses)
        monkeypatch.setattr(llm, "create_message", fake)
        return fake

    return install


@pytest.fixture
def stub_tool(monkeypatch):
    """Replace a tool's implementation; returns the list of recorded inputs."""

    def install(name, fn):
        original = assistant.TOOLS_BY_NAME[name]
        monkeypatch.setitem(assistant.TOOLS_BY_NAME, name, dataclasses.replace(original, run=fn))

    return install


def sid():
    return assistant.new_session_id()


# ---- the loop ---------------------------------------------------------------------------


def test_plain_answer_and_request_shape(model):
    fake = model(reply(text("Hello there."), usage=(100, 20, 50, 10)))
    turn = assistant.ask(sid(), "  hi  ")
    assert turn.answer == "Hello there." and turn.tools_used == [] and not turn.refused
    assert (
        turn.rounds == 0 and turn.usage.input_tokens == 100 and turn.usage.cache_read_tokens == 50
    )
    (request,) = fake.requests
    assert request["max_tokens"] == 4096
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["text"] == assistant.SYSTEM_PROMPT
    for banned in ("temperature", "top_p", "top_k", "tool_choice", "model"):
        assert banned not in request
    assert [t["name"] for t in request["tools"]] == sorted(t["name"] for t in request["tools"])
    assert all(t["strict"] is True for t in request["tools"])


def test_one_tool_round_then_answer(model, stub_tool):
    seen = []

    def geocode(args):
        seen.append(args)
        return assistant.ToolResult(
            {"name": "Chennai", "lat": 13.08},
            "Chennai -> 13.08",
            (assistant.Source("OSM", "https://osm.org/c"),),
        )

    stub_tool("geocode", geocode)
    first = reply(text("Let me look."), tool_use("t1", "geocode", query="Chennai"), stop="tool_use")
    fake = model(first, reply(text("Chennai is at 13.08 N.")))
    turn = assistant.ask(sid(), "Where is Chennai?")
    assert seen == [{"query": "Chennai"}]
    assert turn.answer == "Chennai is at 13.08 N."
    assert [(c.name, c.summary) for c in turn.tools_used] == [("geocode", "Chennai -> 13.08")]
    assert [s.url for s in turn.sources] == ["https://osm.org/c"]
    assert turn.rounds == 1 and turn.usage.api_calls == 2
    second = fake.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"] is first.content  # appended unchanged
    (result,) = second[2]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert "is_error" not in result and json.loads(result["content"])["lat"] == 13.08


def test_parallel_tool_calls_come_back_in_one_user_message(model, stub_tool):
    stub_tool("geocode", lambda a: assistant.ToolResult({"q": a["query"]}, "g"))
    stub_tool("gdacs_alerts", lambda a: assistant.ToolResult({"n": 2}, "a"))
    calls = reply(
        tool_use("a1", "geocode", query="Lima"),
        tool_use("b2", "gdacs_alerts"),
        tool_use("c3", "geocode", query="Quito"),
        stop="tool_use",
    )
    fake = model(calls, reply(text("done")))
    turn = assistant.ask(sid(), "compare")
    user_msg = fake.requests[1]["messages"][-1]
    assert user_msg["role"] == "user"
    assert [r["tool_use_id"] for r in user_msg["content"]] == ["a1", "b2", "c3"]
    assert all(r["type"] == "tool_result" for r in user_msg["content"])
    assert json.loads(user_msg["content"][2]["content"]) == {"q": "Quito"}
    assert [c.name for c in turn.tools_used] == ["geocode", "gdacs_alerts", "geocode"]
    assert turn.rounds == 1  # one round, three calls
    assert len(fake.requests) == 2


def test_failing_tool_becomes_an_error_result(model, stub_tool):
    def boom(args):
        raise RuntimeError("secret internal detail /etc/passwd")

    def down(args):
        raise ServiceError("USGS is down")

    stub_tool("recent_earthquakes", boom)
    stub_tool("geocode", down)
    fake = model(
        reply(
            tool_use("x", "recent_earthquakes"),
            tool_use("y", "geocode", query="q"),
            tool_use("z", "no_such_tool"),
            stop="tool_use",
        ),
        reply(text("I could not check that.")),
    )
    turn = assistant.ask(sid(), "what happened?")
    results = fake.requests[1]["messages"][-1]["content"]
    assert [r.get("is_error") for r in results] == [True, True, True]
    assert "secret internal detail" not in results[0]["content"]  # unexpected errors are not leaked
    assert results[1]["content"] == "Error: USGS is down"
    assert "unknown tool" in results[2]["content"]
    assert turn.answer == "I could not check that."
    assert all(c.summary.startswith(("failed", "unknown")) for c in turn.tools_used)


def test_tool_input_validation_errors_are_reported(model):
    fake = model(
        reply(tool_use("x", "earthquake_hazard", lat=123, lon=0), stop="tool_use"),
        reply(text("ok")),
    )
    assistant.ask(sid(), "hazard")
    (result,) = fake.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True and "'lat' must be between -90 and 90" in result["content"]


def test_round_cap_gives_a_graceful_answer(model, stub_tool):
    stub_tool("gdacs_alerts", lambda a: assistant.ToolResult({}, "ok"))
    again = lambda i: reply(tool_use(f"t{i}", "gdacs_alerts"), stop="tool_use")  # noqa: E731
    # cap is 3: three executed rounds, then a wrap-up request that is refused, then a last call
    fake = model(again(1), again(2), again(3), again(4), again(5))
    session = sid()
    turn = assistant.ask(session, "loop forever")
    assert turn.answer == assistant.GIVE_UP_TEXT and turn.rounds == 3
    assert len(fake.requests) == 5
    wrap_up = fake.requests[4]["messages"][-1]["content"][0]
    assert wrap_up["is_error"] is True and "limit reached" in wrap_up["content"]
    assert assistant.sessions.get(session).user_turns == 0  # half-finished exchange not stored


def test_round_cap_wrap_up_call_can_still_answer(model, stub_tool):
    stub_tool("gdacs_alerts", lambda a: assistant.ToolResult({}, "ok"))
    again = lambda i: reply(tool_use(f"t{i}", "gdacs_alerts"), stop="tool_use")  # noqa: E731
    model(again(1), again(2), again(3), again(4), reply(text("Here is what I found.")))
    session = sid()
    turn = assistant.ask(session, "q")
    assert turn.answer == "Here is what I found." and turn.rounds == 3
    assert assistant.sessions.get(session).user_turns == 1


def test_zero_round_cap_still_terminates(model, stub_tool, monkeypatch):
    monkeypatch.setenv("NDP_ASSISTANT_MAX_TOOL_ROUNDS", "0")
    get_settings.cache_clear()
    model(reply(tool_use("a", "gdacs_alerts"), stop="tool_use"), reply(text("No tools used.")))
    assert assistant.ask(sid(), "q").answer == "No tools used."


def test_refusal_stop_reason(model):
    session = sid()
    model(reply(text("partial"), stop="refusal"))
    turn = assistant.ask(session, "something the classifier dislikes")
    assert turn.refused and turn.answer == assistant.REFUSAL_TEXT
    assert (
        assistant.sessions.get(session).user_turns == 0
        and assistant.sessions.get(session).messages == []
    )


def test_max_tokens_with_text_is_marked_truncated_and_kept(model):
    session = sid()
    model(reply(text("A long answer that stops"), stop="max_tokens"))
    turn = assistant.ask(session, "q")
    assert (
        turn.answer.startswith("A long answer that stops")
        and assistant.TRUNCATED_NOTE in turn.answer
    )
    assert assistant.sessions.get(session).user_turns == 1


def test_max_tokens_cutting_a_tool_call_is_not_executed_or_stored(model):
    session = sid()
    fake = model(reply(tool_use("t", "geocode", query="Lim"), stop="max_tokens"))
    turn = assistant.ask(session, "q")
    assert turn.answer == assistant.GIVE_UP_TEXT and turn.tools_used == []
    assert len(fake.requests) == 1 and assistant.sessions.get(session).user_turns == 0


def test_thinking_blocks_are_preserved_unchanged_across_rounds_and_turns(model, stub_tool):
    stub_tool("geocode", lambda a: assistant.ToolResult({"ok": 1}, "g"))
    think = thinking("secret reasoning", "sig-abc")
    first = reply(think, tool_use("t1", "geocode", query="x"), stop="tool_use")
    final = reply(thinking("more", "sig-def"), text("answer one"))
    fake = model(first, final, reply(text("answer two")))
    session = sid()
    assistant.ask(session, "one")
    assert fake.requests[1]["messages"][1]["content"][0] is think
    assistant.ask(session, "two")
    history = fake.requests[2]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant", "user"]
    assert history[1]["content"] is first.content and history[3]["content"] is final.content
    assert history[1]["content"][0].signature == "sig-abc"
    assert history[4] == {"role": "user", "content": "two"}


def test_api_failure_does_not_poison_the_session(model):
    session = sid()
    model(ServiceError("Claude is rate limited right now; please retry in a minute."))
    with pytest.raises(ServiceError, match="rate limited"):
        assistant.ask(session, "q")
    assert assistant.sessions.get(session).user_turns == 0


def test_empty_answer_is_not_stored(model):
    session = sid()
    model(reply(thinking()))
    assert assistant.ask(session, "q").answer == assistant.EMPTY_TEXT
    assert assistant.sessions.get(session).messages == []


def test_pause_turn_is_resumed(model):
    fake = model(reply(text("..."), stop="pause_turn"), reply(text("finished")))
    assert assistant.ask(sid(), "q").answer == "finished"
    assert fake.requests[1]["messages"][-1]["role"] == "assistant"


def test_not_configured_and_bad_session(unconfigured):
    with pytest.raises(llm.LLMNotConfigured):
        assistant.ask(sid(), "hello")


def test_invalid_session_id_is_rejected(model):
    model(reply(text("x")))
    with pytest.raises(assistant.AssistantError, match="session"):
        assistant.ask("short", "hello")


def test_session_turn_limit_and_busy_session(model):
    session_id = sid()
    session = assistant.sessions.get(session_id)
    session.log.extend([("q", assistant.AssistantTurn("a"))] * assistant.MAX_USER_TURNS_PER_SESSION)
    model(reply(text("never used")))
    with pytest.raises(assistant.AssistantError, match="Reset"):
        assistant.ask(session_id, "one more")

    other = sid()
    busy = assistant.sessions.get(other)
    assert busy.lock.acquire(blocking=False)
    try:
        with pytest.raises(assistant.AssistantError, match="still working"):
            assistant.ask(other, "hi")
    finally:
        busy.lock.release()


# ---- input cleaning ----------------------------------------------------------------------


def test_clean_message():
    assert assistant.clean_message("  Is it \x00safe?​‮\r\n\r\n\r\n\r\nyes  ") == "Is it safe?\n\nyes"
    with pytest.raises(assistant.AssistantError, match="type a question"):
        assistant.clean_message("  ​ \n")
    with pytest.raises(assistant.AssistantError, match="too long"):
        assistant.clean_message("x" * (assistant.MAX_MESSAGE_CHARS + 1))
    assert len(assistant.clean_message("x" * assistant.MAX_MESSAGE_CHARS)) == 1500
    assert assistant.clean_message("¿Qué debo hacer? 東京 भूकंप") == "¿Qué debo hacer? 東京 भूकंप"


# ---- registry: schemas, order, and no way to send anything ----------------------------------

ALLOWED_SCHEMA_KEYS = {
    "type",
    "description",
    "enum",
    "properties",
    "required",
    "additionalProperties",
    "items",
}


def test_registry_has_the_nine_tools_in_deterministic_order():
    names = [t.name for t in assistant.TOOLS]
    assert names == sorted(names) and len(names) == 9
    assert set(names) == {
        "recent_earthquakes",
        "earthquake_hazard",
        "geocode",
        "nearby_services",
        "weather_forecast",
        "gdacs_alerts",
        "flood_forecast",
        "preparedness_guidance",
        "draft_alert",
    }
    assert assistant.TOOL_DEFINITIONS == [t.definition() for t in assistant.TOOLS]
    assert json.dumps(assistant.TOOL_DEFINITIONS) == json.dumps(
        [t.definition() for t in assistant.TOOLS]
    )


@pytest.mark.parametrize("tool", assistant.TOOLS, ids=lambda t: t.name)
def test_tool_schemas_are_strict_and_plain(tool):
    d = tool.definition()
    schema = d["input_schema"]
    assert (
        d["strict"] is True
        and schema["additionalProperties"] is False
        and schema["type"] == "object"
    )
    assert set(schema["required"]) <= set(schema["properties"])
    assert len(d["description"]) > 120  # says when to use it
    for name, prop in schema["properties"].items():
        assert set(prop) <= ALLOWED_SCHEMA_KEYS, (name, set(prop) - ALLOWED_SCHEMA_KEYS)
        assert prop["type"] in {"string", "number", "integer", "boolean"} and prop["description"]


def _all_names(fn):
    names = set(fn.__code__.co_names)
    for const in fn.__code__.co_consts:
        if hasattr(const, "co_names"):
            names |= set(const.co_names)
    return names


def test_no_tool_and_no_module_code_can_send_anything():
    forbidden = re.compile(r"notify|dispatch|send_email|send_sms|smtplib|twilio|sendmail", re.I)
    source = inspect.getsource(assistant)
    assert not forbidden.search(source)
    for tool in assistant.TOOLS:
        assert not forbidden.search(tool.name + tool.description.replace("sends nothing", ""))
        assert not (_all_names(tool.run) & {"notify", "dispatch", "send_email", "send_sms"})
        assert not forbidden.search(inspect.getsource(tool.run))
    assert not [n for n in vars(assistant) if forbidden.search(n)]
    assert assistant.TOOLS_BY_NAME["draft_alert"].description.count("sends nothing") == 1


def test_draft_alert_only_composes_text(monkeypatch):
    monkeypatch.setattr(
        assistant.geo,
        "nearby_services",
        lambda lat, lon, r: [
            assistant.geo.NearbyPlace(
                "City Hospital", "Hospital", lat, lon, "1 Main St", "+977-1-000"
            )
        ],
    )
    content, is_error, call, _ = assistant.run_tool(
        "draft_alert",
        {"place_name": "Kathmandu", "lat": 27.7, "lon": 85.3, "magnitude": 6.5, "depth_km": 12},
    )
    data = json.loads(content)
    assert not is_error and data["status"] == "DRAFT_NOT_SENT"
    assert "EARTHQUAKE ALERT" in data["draft"] and "City Hospital" in data["draft"]
    assert (
        "Nothing has been sent" in data["next_step"] and call.summary == "draft composed (not sent)"
    )


def test_draft_alert_tolerates_a_failing_facility_lookup(monkeypatch):
    def fail(*a):
        raise ServiceError("Overpass busy")

    monkeypatch.setattr(assistant.geo, "nearby_services", fail)
    data = json.loads(assistant.run_tool("draft_alert", {"place_name": "X", "lat": 1, "lon": 2})[0])
    assert data["status"] == "DRAFT_NOT_SENT" and "could not be added" in data["facilities_note"]


# ---- individual tools with faked services -------------------------------------------------


def _quake(i, mag, lat, lon, minute):
    return SimpleNamespace(
        id=f"e{i}",
        magnitude=mag,
        place=f"Place {i}",
        lat=lat,
        lon=lon,
        depth_km=10.0,
        time=datetime(2026, 10, 25, 12, minute, tzinfo=UTC),
        url=f"https://usgs.example/e{i}",
    )


def test_recent_earthquakes_filters_sorts_and_limits(monkeypatch):
    events = [
        _quake(1, 6.1, 35.0, 139.0, 1),
        _quake(2, 5.0, 36.0, 140.0, 30),
        _quake(3, 4.6, -33.0, -70.0, 15),
    ]
    monkeypatch.setattr(assistant.usgs, "recent_events", lambda limit: events)
    run = lambda args: json.loads(assistant.run_tool("recent_earthquakes", args)[0])  # noqa: E731
    assert [e["magnitude"] for e in run({})["events"]] == [6.1, 5.0, 4.6]
    assert [e["magnitude"] for e in run({"sort_by": "time"})["events"]] == [5.0, 4.6, 6.1]
    assert [e["magnitude"] for e in run({"min_magnitude": 5.0, "limit": 1})["events"]] == [6.1]
    near = run({"lat": 35.5, "lon": 139.5, "radius_km": 300})
    assert [e["magnitude"] for e in near["events"]] == [6.1, 5.0] and "distance_km" in near[
        "events"
    ][0]
    assert near["events"][0]["time_utc"] == "2026-10-25 12:01"
    content, is_error, _, _ = assistant.run_tool("recent_earthquakes", {"lat": 35.0})
    assert is_error and "'lon' is required" in content


def test_earthquake_hazard_and_flood_tools_report_model_not_trained(monkeypatch):
    from ndp.ml import flood, hazard, registry

    def not_trained(*a, **k):
        raise registry.ModelNotTrained("earthquake_hazard")

    monkeypatch.setattr(hazard, "predict", not_trained)
    monkeypatch.setattr(flood, "predict", not_trained)
    content, is_error, call, _ = assistant.run_tool("earthquake_hazard", {"lat": 1, "lon": 2})
    assert is_error and "not trained" in content and call.summary.startswith("failed")
    args = dict(
        level_now_m=1, level_1h_ago_m=1, level_3h_ago_m=1, rain_3h_mm=0, rain_6h_mm=0, rain_24h_mm=0
    )
    assert assistant.run_tool("flood_forecast", args)[1] is True


def test_earthquake_hazard_result_carries_caveats(monkeypatch):
    from ndp.ml import hazard

    est = hazard.HazardEstimate(
        lat=35.0,
        lon=139.0,
        radius_km=300.0,
        n_events=7,
        years=52.0,
        annual_rate_m55=0.13,
        b_value=1.0,
        b_is_global=True,
        annual_rate={"6.0": 0.04},
        prob_1y={"6.0": 0.04},
        prob_10y={"6.0": 0.33},
        max_magnitude=7.9,
        max_date="2011-03-11",
        depth_median_km=30.0,
        nearest_event={"distance_km": 50, "magnitude": 6.0, "year": 2001},
        level="high",
        elevated=True,
    )
    monkeypatch.setattr(hazard, "predict", lambda lat, lon, r: est)
    content, is_error, _, sources = assistant.run_tool("earthquake_hazard", {"lat": 35, "lon": 139})
    data = json.loads(content)
    assert not is_error and data["n_events"] == 7 and data["b_is_global"] is True
    joined = " ".join(data["caveats"])
    assert "aftershocks" in joined and "fewer than about 10" in joined and "this circle" in joined
    assert "1965-2016" in joined and "not a prediction" in joined
    assert "33%" in data["plain_language"] and sources


def test_flood_forecast_reports_levels_and_rise_not_a_verdict(monkeypatch):
    from ndp.ml import flood

    hz = lambda h, lvl: flood.HorizonForecast(h, lvl, lvl > 1.44, 1.44, 1.5)  # noqa: E731
    fc = flood.FloodForecast([hz(1, 1.55), hz(3, 1.7), hz(6, 1.6)], True, 1.7)
    monkeypatch.setattr(flood, "predict", lambda *a: fc)
    args = dict(
        level_now_m=1.5,
        level_1h_ago_m=1.5,
        level_3h_ago_m=1.4,
        rain_3h_mm=5,
        rain_6h_mm=9,
        rain_24h_mm=20,
    )
    content, is_error, _, _ = assistant.run_tool("flood_forecast", args)
    data = json.loads(content)
    assert not is_error and "Hoppers Crossing" in data["scope"]
    assert [h["rise_vs_current_m"] for h in data["horizons"]] == [0.05, 0.2, 0.1]
    blob = content.lower()
    assert "any_flood" not in blob and "above_flood" not in blob and "flood_threshold" not in blob
    assert "7% recall" in " ".join(data["caveats"]) and "persistence" in " ".join(data["caveats"])
    assert assistant.run_tool("flood_forecast", {**args, "level_now_m": 99})[1] is True


def test_weather_gdacs_geocode_nearby_and_guidance_tools(monkeypatch, tmp_path):
    from ndp.services import gdacs, openmeteo

    day = SimpleNamespace(
        day=date(2026, 10, 26), precip_mm=3.2, max_gust_kmh=40.0, temp_min_c=24.0, temp_max_c=31.0
    )
    fc = SimpleNamespace(
        lat=13.08,
        lon=80.27,
        timezone="Asia/Kolkata",
        days=3,
        precip_24h_max_mm=12.0,
        precip_72h_total_mm=20.0,
        max_gust_kmh=45.0,
        temp_min_c=24.0,
        temp_max_c=33.0,
        advisories=[SimpleNamespace(level="info", text="Nothing heavy")],
        daily=[day],
        disclaimer="d",
    )
    monkeypatch.setattr(openmeteo, "forecast", lambda lat, lon, days: fc)
    w = json.loads(assistant.run_tool("weather_forecast", {"lat": 13.08, "lon": 80.27})[0])
    assert w["advisories"][0]["level"] == "info" and w["daily"][0]["date"] == "2026-10-26"

    event = SimpleNamespace(
        kind="TC",
        kind_name="Tropical cyclone",
        name="NAME",
        alert_level="Orange",
        country="India",
        lat=15.0,
        lon=80.0,
        from_date=datetime(2026, 10, 20, tzinfo=UTC),
        to_date=None,
        url="https://gdacs.example/1",
        summary="Wind 120 km/h " * 40,
    )
    other = SimpleNamespace(**{**vars(event), "kind": "EQ", "name": "Q"})
    monkeypatch.setattr(gdacs, "recent_alerts", lambda limit: [event, other])
    g = json.loads(assistant.run_tool("gdacs_alerts", {"kind": "TC"})[0])
    assert g["returned"] == 1 and len(g["alerts"][0]["summary"]) <= 200
    assert assistant.run_tool("gdacs_alerts", {"kind": "XX"})[1] is True

    monkeypatch.setattr(
        assistant.geo, "geocode", lambda q: assistant.geo.Place("Chennai, India", 13.0827, 80.2707)
    )
    geo_result = json.loads(assistant.run_tool("geocode", {"query": "Chennai"})[0])
    assert geo_result["lat"] == 13.0827 and "ambiguous" in geo_result["note"]
    assert assistant.run_tool("geocode", {"query": "  "})[1] is True

    P = assistant.geo.NearbyPlace
    monkeypatch.setattr(
        assistant.geo,
        "nearby_services",
        lambda lat, lon, r: [
            P("Far Police", "Police", lat + 0.2, lon, "", ""),
            P("Near Hospital", "Hospital", lat + 0.01, lon, "1 Road", "+1 555"),
        ],
    )
    n = json.loads(assistant.run_tool("nearby_services", {"lat": 10, "lon": 20})[0])
    assert [p["name"] for p in n["places"]] == ["Near Hospital", "Far Police"]
    only = json.loads(
        assistant.run_tool("nearby_services", {"lat": 10, "lon": 20, "kind": "police"})[0]
    )
    assert [p["name"] for p in only["places"]] == ["Far Police"]

    content, is_error, _, sources = assistant.run_tool(
        "preparedness_guidance",
        {"query": "what to do during an earthquake", "hazard": "earthquake"},
    )
    data = json.loads(content)
    assert (
        not is_error
        and data["excerpts"]
        and all(e["url"].startswith("https://") for e in data["excerpts"])
    )
    assert sources and all(s.url.startswith("https://") for s in sources)
    assert (
        assistant.run_tool("preparedness_guidance", {"query": "x", "hazard": "zombies"})[1] is True
    )


def test_oversized_tool_results_are_capped_but_stay_valid_json(stub_tool):
    stub_tool("geocode", lambda a: assistant.ToolResult({"blob": "x" * 50_000}, "big"))
    content, is_error, _, _ = assistant.run_tool("geocode", {"query": "q"})
    assert not is_error and len(content) <= assistant.MAX_TOOL_RESULT_CHARS + 100
    assert json.loads(content)["truncated"] is True


# ---- rate limiter, sessions, rendering -------------------------------------------------------


def test_rate_limiter_sliding_window():
    now = [100.0]
    limiter = assistant.RateLimiter(limit=3, window=60, clock=lambda: now[0])
    assert [limiter.hit("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = limiter.hit("a")
    assert not allowed and retry == pytest.approx(60.0)
    assert limiter.hit("b")[0]  # other clients unaffected
    now[0] += 59
    assert not limiter.hit("a")[0]
    now[0] += 1.5
    assert limiter.hit("a")[0]


def test_rate_limiter_bounds_memory():
    limiter = assistant.RateLimiter(limit=1, window=60, max_keys=5, clock=lambda: 0.0)
    for i in range(50):
        limiter.hit(f"client-{i}")
    assert len(limiter._hits) == 5


def test_session_store_is_lru():
    store = assistant.SessionStore(max_sessions=3)
    for key in "abc":
        store.get(key)
    store.get("a")  # touch: b is now the oldest
    store.get("d")
    assert "b" not in store and all(k in store for k in "acd") and len(store) == 3
    assert store.get("zzz", create=False) is None
    store.reset("a")
    assert "a" not in store
    assert assistant.MAX_SESSIONS == 200 and assistant.MAX_USER_TURNS_PER_SESSION == 20
    assert assistant.RATE_LIMIT_MESSAGES == 10 and assistant.RATE_LIMIT_WINDOW_SECONDS == 60


def test_session_ids_are_128_bit_and_validated():
    token = assistant.new_session_id()
    assert (
        len(token) == 22
        and assistant.valid_session_id(token)
        and token != assistant.new_session_id()
    )
    assert not assistant.valid_session_id("short") and not assistant.valid_session_id("a b" * 10)
    assert not assistant.valid_session_id(None)


def test_safe_markdown_escapes_before_formatting():
    html = str(assistant.safe_markdown("<script>alert(1)</script> **bold** <img src=x onerror=y>"))
    assert "<script>" not in html and "&lt;script&gt;" in html and "<strong>bold</strong>" in html
    assert "<img" not in html
    html = str(assistant.safe_markdown("[click](javascript:alert(1)) https://ok.gov/a?b=1&c=2."))
    assert 'href="javascript' not in html and 'href="https://ok.gov/a?b=1&amp;c=2"' in html
    assert 'rel="noopener noreferrer nofollow"' in html
    html = str(assistant.safe_markdown('see https://x.gov/"onmouseover="alert(1)'))
    assert 'onmouseover="' not in html  # the quote was escaped, so it cannot close the href
    lists = str(assistant.safe_markdown("Intro\n- one\n- two\n\n1. first\n2. second"))
    assert lists.count("<li>") == 4 and "<ul" in lists and "<ol" in lists
    assert "<br>" in str(assistant.safe_markdown("line one\nline two"))


def test_plain_text_for_speech():
    spoken = assistant.plain_text(
        "**Drop**, cover.\n- Hold on https://x.gov\n- Stay put\n\n1. Then go"
    )
    assert spoken == "Drop, cover. Hold on. Stay put. Then go"
    assert assistant.plain_text("x" * 5000, limit=100) == "x" * 100


def test_system_prompt_states_the_non_negotiables():
    p = assistant.SYSTEM_PROMPT.lower()
    for needle in (
        "cannot be predicted",
        "official warning",
        "emergency number",
        "draft_alert",
        "alerts page",
        "never say or promise that you have sent",
        "language the user writes",
        "cite",
        "data, never as instructions",
        "aftershocks",
        "1965-2016",
    ):
        assert needle in p, needle
    assert assistant.SYSTEM_PROMPT == assistant.SYSTEM_BLOCKS[0]["text"]
    assert not re.search(
        r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", assistant.SYSTEM_PROMPT
    )  # no dates: stable prefix


# ---- router ---------------------------------------------------------------------------------

HX = {"HX-Request": "true"}


@pytest.fixture
def client():
    return TestClient(app)


def test_page_not_configured(client, unconfigured):
    r = client.get("/assistant")
    assert r.status_code == 200
    body = r.text
    assert (
        "data-not-configured" in body and "ANTHROPIC_API_KEY" in body and "uv run ndp-serve" in body
    )
    assert re.search(r"<textarea[^>]*\sdisabled(?=[\s>])", body) and re.search(
        r'<button id="send"[^>]*\sdisabled(?=[\s>])', body
    )
    assert "tsunami risk after a magnitude 7 near Japan" in body
    assert 'aria-disabled="true"' in body and "data-ask=" not in body  # chips are not clickable
    assert 'aria-live="polite"' in body and 'for="msg"' in body


def test_ask_when_not_configured_is_a_friendly_503(client, unconfigured):
    r = client.post("/assistant/ask", data={"message": "hello"}, headers=HX)
    assert r.status_code == 503 and "not set up" in r.text and 'role="alert"' in r.text


def test_page_configured_has_working_chips_and_form(client, configured):
    body = client.get("/assistant").text
    assert "data-not-configured" not in body and body.count("data-ask=") >= 3
    assert (
        "Weather outlook for Chennai this week" in body
        and "What should I do during an earthquake?" in body
    )
    assert (
        not re.search(r"<textarea[^>]*\sdisabled(?=[\s>])", body) and "Reset conversation" in body
    )
    assert 'maxlength="1500"' in body and "not official warnings" in body


def test_ask_htmx_renders_turn_with_chips_sources_and_speak(client, model, stub_tool):
    stub_tool(
        "preparedness_guidance",
        lambda a: assistant.ToolResult(
            {"excerpts": []},
            "1 excerpt",
            (assistant.Source("CDC: During an Earthquake", "https://www.cdc.gov/x"),),
        ),
    )
    stub_tool("recent_earthquakes", lambda a: assistant.ToolResult({}, "2 events"))
    model(
        reply(
            tool_use("a", "preparedness_guidance", query="q"),
            tool_use("b", "recent_earthquakes"),
            stop="tool_use",
        ),
        reply(
            text(
                "**Drop, cover, hold on.**\n- Stay away from windows <b>now</b>\n\nSee https://www.cdc.gov/x"
            )
        ),
    )
    r = client.post(
        "/assistant/ask", data={"message": "What should I do during an earthquake?"}, headers=HX
    )
    assert r.status_code == 200 and "<html" not in r.text  # a partial
    assert "What should I do during an earthquake?" in r.text
    assert "<strong>Drop, cover, hold on.</strong>" in r.text and "&lt;b&gt;now&lt;/b&gt;" in r.text
    assert "<b>now</b>" not in r.text
    assert 'data-speak="Drop, cover, hold on. Stay away from windows now. See"' in r.text
    assert (
        "Used:" in r.text and "preparedness_guidance" in r.text and "recent_earthquakes" in r.text
    )
    assert 'href="https://www.cdc.gov/x"' in r.text and "CDC: During an Earthquake" in r.text


def test_ask_without_htmx_returns_the_full_page_with_history(client, model):
    model(reply(text("Plain answer.")))
    r = client.post("/assistant/ask", data={"message": "Hello?"})
    assert r.status_code == 200 and "<html" in r.text
    assert "Hello?" in r.text and "Plain answer." in r.text
    again = client.get("/assistant")  # reload keeps the conversation
    assert "Plain answer." in again.text and "data-welcome" not in again.text


def test_reset_clears_the_conversation(client, model):
    model(reply(text("First answer.")))
    client.post("/assistant/ask", data={"message": "q"}, headers=HX)
    r = client.post("/assistant/reset", headers=HX)
    assert r.status_code == 200 and "data-welcome" in r.text
    assert "First answer." not in client.get("/assistant").text
    plain = client.post("/assistant/reset", follow_redirects=False)
    assert plain.status_code == 303 and plain.headers["location"].endswith("/assistant")


def test_cookie_flags_and_scope(client, configured):
    r = client.get("/assistant")
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("ndp_assistant=") and "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower() and "Path=/assistant" in cookie
    token = cookie.split(";")[0].split("=", 1)[1]
    assert assistant.valid_session_id(token) and len(token) == 22
    assert "set-cookie" not in client.get("/assistant").headers  # only set when missing
    assert "ndp_assistant" not in client.get("/healthz").headers.get("set-cookie", "")
    fresh = TestClient(app)
    assert "ndp_assistant" not in fresh.get("/").headers.get("set-cookie", "")
    forged = TestClient(app, cookies={"ndp_assistant": "x y"})
    assert "ndp_assistant=" in forged.get("/assistant").headers["set-cookie"]  # invalid -> replaced


def test_rate_limited_response_is_a_friendly_429(client, model, monkeypatch):
    monkeypatch.setattr(assistant, "rate_limiter", assistant.RateLimiter(limit=2, window=60))
    model(reply(text("one")), reply(text("two")))
    for _ in range(2):
        assert client.post("/assistant/ask", data={"message": "hi"}, headers=HX).status_code == 200
    r = client.post("/assistant/ask", data={"message": "again"}, headers=HX)
    assert r.status_code == 429 and "try again in" in r.text and 'role="alert"' in r.text
    assert int(r.headers["retry-after"]) >= 1 and "again" in r.text


def test_errors_are_friendly_never_500(client, model, monkeypatch):
    model(ServiceError("Claude is rate limited right now; please retry in a minute."))
    r = client.post("/assistant/ask", data={"message": "hi"}, headers=HX)
    assert r.status_code == 503 and "rate limited" in r.text

    r = client.post("/assistant/ask", data={"message": "   "}, headers=HX)
    assert r.status_code == 400 and "type a question" in r.text
    r = client.post("/assistant/ask", data={"message": "x" * 1600}, headers=HX)
    assert r.status_code == 400 and "too long" in r.text

    assert client.post("/assistant/ask", headers=HX).status_code == 400  # missing field entirely

    def explode(session_id, message):
        raise RuntimeError("kaboom with /secret/path")

    monkeypatch.setattr(assistant, "ask", explode)
    r = client.post("/assistant/ask", data={"message": "hi"}, headers=HX)
    assert r.status_code == 503 and "kaboom" not in r.text and "went wrong" in r.text


def test_sessions_are_isolated_between_browsers(model):
    model(reply(text("for alice")), reply(text("for bob")))
    alice, bob = TestClient(app), TestClient(app)
    alice.post("/assistant/ask", data={"message": "alice question"}, headers=HX)
    bob.post("/assistant/ask", data={"message": "bob question"}, headers=HX)
    assert (
        "alice question" in alice.get("/assistant").text
        and "bob question" not in alice.get("/assistant").text
    )
