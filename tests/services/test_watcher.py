"""The real-event watcher: polling, dedupe, state file, failure handling and lifecycle.
Nothing touches the network, the real state directory or a real mail server."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from ndp.config import get_settings
from ndp.main import create_app
from ndp.services import ServiceError, alerting, geo, notify, usgs, watcher

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def quake(event_id, magnitude, minutes_ago=30, place="10 km SW of Somewhere"):
    return usgs.QuakeEvent(
        id=event_id,
        magnitude=magnitude,
        place=place,
        lat=35.0,
        lon=139.0,
        depth_km=12.0,
        time=NOW - timedelta(minutes=minutes_ago),
        url=f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}",
    )


class Feed:
    """A mutable stand-in for usgs.recent_events."""

    def __init__(self, *events):
        self.events = list(events)

    def __call__(self, limit):
        return list(self.events)


class Recorder:
    """A stand-in for watcher._deliver that records what would have been alerted."""

    def __init__(self, outcome=("sent: email", False)):
        self.outcome = outcome
        self.events = []

    def __call__(self, event):
        self.events.append(event.id)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "state" / "watcher.json"  # the directory does not exist yet
    monkeypatch.setattr(watcher, "STATE_PATH", path)
    watcher.reset()
    yield path
    watcher.reset()


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setenv(f"NDP_{key.upper()}", str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


def saved(path):
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------ polling


def test_first_poll_marks_existing_events_without_alerting(state_file, env):
    env(watch_min_magnitude="6.0")
    feed = Feed(quake("a", 7.1), quake("b", 4.6), quake("c", 6.2))
    deliver = Recorder()
    assert watcher.poll_once(feed, deliver, NOW) == 0
    assert deliver.events == []
    data = saved(state_file)
    assert data["baselined"] is True and set(data["seen"]) == {"a", "b", "c"}
    assert watcher.status().handled == []


def test_new_event_above_threshold_alerts_exactly_once(state_file, env):
    env(watch_min_magnitude="6.0")
    feed, deliver = Feed(quake("old", 6.5, minutes_ago=600)), Recorder()
    watcher.poll_once(feed, deliver, NOW)
    feed.events.append(quake("new", 6.4, minutes_ago=5, place="Off the coast of Honshu"))

    assert watcher.poll_once(feed, deliver, NOW) == 1
    assert watcher.poll_once(feed, deliver, NOW) == 0
    assert watcher.poll_once(feed, deliver, NOW) == 0
    assert deliver.events == ["new"]

    [entry] = watcher.status().handled
    assert (entry.id, entry.magnitude, entry.place) == ("new", 6.4, "Off the coast of Honshu")
    assert entry.outcome == "sent: email" and entry.at == NOW


def test_threshold_is_inclusive_and_lower_events_are_ignored(state_file, env):
    env(watch_min_magnitude="6.0")
    feed, deliver = Feed(), Recorder()
    watcher.poll_once(feed, deliver, NOW)
    feed.events += [quake("low", 5.9), quake("exact", 6.0)]
    watcher.poll_once(feed, deliver, NOW)
    assert deliver.events == ["exact"]
    assert [h.id for h in watcher.status().handled] == ["exact"]
    # The low one is remembered as seen, so raising the feed's quake count later never re-checks it.
    assert "low" in saved(state_file)["seen"]


def test_burst_is_alerted_oldest_first(state_file, env):
    env(watch_min_magnitude="6.0")
    feed, deliver = Feed(), Recorder()
    watcher.poll_once(feed, deliver, NOW)
    feed.events += [quake("later", 6.3, minutes_ago=5), quake("earlier", 6.8, minutes_ago=50)]
    watcher.poll_once(feed, deliver, NOW)
    assert deliver.events == ["earlier", "later"]
    assert [h.id for h in watcher.status().handled] == ["later", "earlier"]  # newest first


def test_events_older_than_a_day_are_not_alerted(state_file, env):
    env(watch_min_magnitude="6.0")
    feed, deliver = Feed(), Recorder()
    watcher.poll_once(feed, deliver, NOW)
    feed.events.append(quake("stale", 7.0, minutes_ago=25 * 60))
    assert watcher.poll_once(feed, deliver, NOW) == 0
    assert deliver.events == [] and "stale" in saved(state_file)["seen"]


def test_dedupe_survives_a_restart_and_alerts_for_quakes_missed_while_down(state_file, env):
    env(watch_min_magnitude="6.0")
    feed, deliver = Feed(quake("seen-before", 6.5)), Recorder()
    watcher.poll_once(feed, deliver, NOW)  # baseline
    feed.events.append(quake("first", 6.6, minutes_ago=20))
    watcher.poll_once(feed, deliver, NOW)
    assert deliver.events == ["first"]

    watcher.reset()  # the process restarts; only the state file remains
    feed.events.append(quake("while-down", 6.9, minutes_ago=10))
    assert watcher.poll_once(feed, deliver, NOW) == 1
    assert deliver.events == ["first", "while-down"]  # "first" and "seen-before" not repeated
    assert [h.id for h in watcher.status().handled] == ["while-down", "first"]


def test_missing_or_corrupt_state_is_a_fresh_start(state_file, env):
    env(watch_min_magnitude="6.0")
    state_file.parent.mkdir(parents=True)
    state_file.write_text("{ this is not json")
    deliver = Recorder()
    assert watcher.poll_once(Feed(quake("x", 7.0)), deliver, NOW) == 0  # baseline, no alert blast
    assert deliver.events == [] and saved(state_file)["seen"] == ["x"]

    state_file.write_text(json.dumps({"seen": "wrong type"}))
    watcher.reset()
    assert watcher.poll_once(Feed(quake("y", 7.0)), deliver, NOW) == 0


def test_state_file_is_written_atomically_and_bounded(state_file, env, monkeypatch):
    env(watch_min_magnitude="6.0")
    monkeypatch.setattr(watcher, "MAX_SEEN", 5)
    watcher.poll_once(Feed(*[quake(f"e{i}", 4.6) for i in range(10)]), Recorder(), NOW)
    assert len(saved(state_file)["seen"]) == 5
    assert [p.name for p in state_file.parent.iterdir()] == ["watcher.json"]  # no temp leftovers


def test_failed_write_leaves_the_previous_state_intact(state_file, monkeypatch):
    watcher.save_state(watcher._State(seen=["a"], baselined=True))
    before = state_file.read_text()

    def explode(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(watcher.os, "replace", explode)
    with pytest.raises(OSError):
        watcher.save_state(watcher._State(seen=["a", "b"], baselined=True))
    assert state_file.read_text() == before
    assert [p.name for p in state_file.parent.iterdir()] == ["watcher.json"]


def test_feed_failure_propagates_so_the_loop_can_record_it(state_file):
    def down(limit):
        raise ServiceError("The USGS feed is unavailable right now (timeout).")

    with pytest.raises(ServiceError):
        watcher.poll_once(down, Recorder(), NOW)
    assert not state_file.exists()  # nothing was baselined from a failed read


# ------------------------------------------------------------------- delivery and failures


def test_delivery_exception_is_recorded_retried_then_given_up(state_file, env):
    env(watch_min_magnitude="6.0")
    feed = Feed()
    watcher.poll_once(feed, Recorder(), NOW)
    feed.events.append(quake("boom", 6.5))
    deliver = Recorder(outcome=RuntimeError("smtp exploded"))

    for _ in range(5):  # extra polls after giving up must not retry again
        watcher.poll_once(feed, deliver, NOW)
    assert deliver.events == ["boom"] * watcher.MAX_ATTEMPTS
    outcomes = [h.outcome for h in watcher.status().handled]
    assert len(outcomes) == watcher.MAX_ATTEMPTS
    assert all("failed: RuntimeError: smtp exploded" in o for o in outcomes)
    assert "will retry" in outcomes[-1] and "will retry" not in outcomes[0]  # newest first
    assert "boom" in saved(state_file)["seen"] and "boom" not in saved(state_file)["attempts"]


def test_one_failing_event_does_not_block_the_next(state_file, env):
    env(watch_min_magnitude="6.0")
    feed = Feed()
    watcher.poll_once(feed, Recorder(), NOW)
    feed.events += [quake("bad", 6.5, minutes_ago=40), quake("good", 6.5, minutes_ago=10)]

    def deliver(event):
        if event.id == "bad":
            raise RuntimeError("nope")
        return "sent: email", False

    assert watcher.poll_once(feed, deliver, NOW) == 2
    by_id = {h.id: h.outcome for h in watcher.status().handled}
    assert by_id["good"] == "sent: email" and by_id["bad"].startswith("failed: RuntimeError")


def test_retry_state_is_persisted_across_restarts(state_file, env):
    env(watch_min_magnitude="6.0")
    feed = Feed()
    watcher.poll_once(feed, Recorder(), NOW)
    feed.events.append(quake("flaky", 6.5))
    watcher.poll_once(feed, Recorder(outcome=("not sent: smtp down", True)), NOW)
    assert saved(state_file)["attempts"] == {"flaky": 1}
    watcher.reset()
    ok = Recorder()
    watcher.poll_once(feed, ok, NOW)
    assert ok.events == ["flaky"] and saved(state_file)["attempts"] == {}


@pytest.fixture
def quiet_geo(monkeypatch):
    monkeypatch.setattr(
        geo, "reverse_geocode", lambda lat, lon: geo.Place("Tokyo, Japan", lat, lon)
    )
    monkeypatch.setattr(
        geo,
        "nearby_services",
        lambda lat, lon, radius_km, limit=40: [
            geo.NearbyPlace("City Hospital", "hospital", lat, lon, "Main Rd", "+81 3 1111")
        ],
    )


def test_deliver_builds_a_reported_message_and_honours_dry_run(quiet_geo, env, monkeypatch):
    env(
        alerts_dry_run="true",
        smtp_user="demo@example.com",
        smtp_password="unused",
        alert_to_email="ops@example.com",
        alert_radius_km="20",
    )
    sent = []
    monkeypatch.setattr(notify, "send_email", lambda *a, **k: sent.append(a))
    seen_radius = []
    monkeypatch.setattr(
        geo,
        "nearby_services",
        lambda lat, lon, radius_km, limit=40: seen_radius.append(radius_km) or [],
    )
    dispatched = []
    real_dispatch = alerting.dispatch
    monkeypatch.setattr(
        alerting,
        "dispatch",
        lambda name, body, *a: dispatched.append((name, body)) or real_dispatch(name, body, *a),
    )

    outcome, retry = watcher._deliver(quake("us1", 6.8))
    assert outcome == "dry run, would send: email to ops@example.com" and retry is False
    assert sent == []  # dry run sends nothing
    assert seen_radius == [20.0]
    [(name, body)] = dispatched
    assert name == "Tokyo, Japan"
    assert "USGS-reported magnitude 6.8 Mw" in body and "not an official warning" in body
    assert "https://earthquake.usgs.gov/earthquakes/eventpage/us1" in body


def test_deliver_tolerates_geocoding_and_facility_lookup_failures(env, monkeypatch):
    env(
        alerts_dry_run="true",
        smtp_user="u@example.com",
        smtp_password="p",
        alert_to_email="o@example.com",
    )

    def fail(*args, **kwargs):
        raise ServiceError("Overpass busy")

    monkeypatch.setattr(geo, "reverse_geocode", fail)
    monkeypatch.setattr(geo, "nearby_services", fail)
    bodies = []
    real_dispatch = alerting.dispatch
    monkeypatch.setattr(
        alerting,
        "dispatch",
        lambda name, body, *a: bodies.append((name, body)) or real_dispatch(name, body, *a),
    )
    outcome, _ = watcher._deliver(quake("us2", 6.1, place="Off the coast of Honshu"))
    assert outcome.startswith("dry run")
    assert bodies[0][0] == "Off the coast of Honshu"  # falls back to the USGS place name


def test_deliver_real_send_failure_is_retryable_but_unconfigured_is_final(
    quiet_geo, env, monkeypatch
):
    env(
        alerts_dry_run="false",
        smtp_user="demo@example.com",
        smtp_password="unused",
        alert_to_email="ops@example.com",
    )

    def refuse(*args, **kwargs):
        raise ServiceError("SMTP authentication failed")

    monkeypatch.setattr(notify, "send_email", refuse)
    outcome, retry = watcher._deliver(quake("us3", 6.5))
    assert "SMTP authentication failed" in outcome and retry is True

    monkeypatch.setattr(notify, "send_email", lambda subject, body: "email to ops@example.com")
    outcome, retry = watcher._deliver(quake("us3", 6.5))
    assert outcome == "sent: email to ops@example.com" and retry is False

    env(smtp_user="", smtp_password="", alert_to_email="")
    outcome, retry = watcher._deliver(quake("us3", 6.5))
    assert outcome.startswith("not sent: email not configured") and retry is False


# --------------------------------------------------------------------------------- the loop


def run_loop_with(polls, interval=300.0, stop_after=None):
    """Run watcher.run_loop with scripted poll outcomes; returns the sleeps requested."""
    delays, errors = [], []
    script = list(polls)

    def poll():
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def fake_sleep(delay):
        delays.append(delay)
        errors.append(watcher.status().last_error)
        if len(delays) >= (stop_after or len(polls)):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(watcher.run_loop(poll, interval=interval, sleep=fake_sleep))
    return delays, errors


def test_loop_survives_errors_backs_off_and_recovers(state_file):
    delays, errors = run_loop_with(
        [ServiceError("USGS down"), RuntimeError("weird bug"), None, None], interval=300.0
    )
    assert delays == [600.0, 1200.0, 300.0, 300.0]  # doubles per failure, resets on success
    assert errors[0] == "ServiceError: USGS down" and errors[1] == "RuntimeError: weird bug"
    assert errors[2] is None  # a good poll clears last_error
    assert watcher.status().last_poll is not None


def test_backoff_is_capped():
    assert watcher.next_delay(300, 0) == 300
    assert watcher.next_delay(300, 1) == 600
    assert watcher.next_delay(300, 20) == watcher.MAX_BACKOFF_SECONDS
    assert watcher.next_delay(7200, 3) == 7200  # never shorter than the normal interval


def test_loop_runs_blocking_work_off_the_event_loop(state_file):
    import threading

    main = threading.get_ident()
    threads = []

    def poll():
        threads.append(threading.get_ident())

    async def fake_sleep(delay):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(watcher.run_loop(poll, interval=1.0, sleep=fake_sleep))
    assert threads and threads[0] != main


# ------------------------------------------------------------------------------ lifecycle


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_lifespan_starts_and_stops_the_watcher_when_enabled(state_file, env, monkeypatch):
    env(watch_quakes="true", watch_interval_seconds="3600")
    polls = []
    monkeypatch.setattr(watcher, "poll_once", lambda *a, **k: polls.append(1))

    with TestClient(create_app()) as client:
        assert wait_for(lambda: polls)
        status = watcher.status()
        assert status.enabled and status.running
        assert status.last_error is None
        task = client.app.state.watcher_task
        assert not task.done()
        assert wait_for(lambda: watcher.status().last_poll is not None)
    assert task.done() and task.cancelled()
    assert watcher.status().running is False
    assert len(polls) == 1  # one poll at start, the long interval prevented another


def test_lifespan_does_nothing_when_disabled(state_file, env, monkeypatch):
    env(watch_quakes="false")
    polls = []
    monkeypatch.setattr(watcher, "poll_once", lambda *a, **k: polls.append(1))
    with TestClient(create_app()) as client:
        time.sleep(0.1)
        status = watcher.status()
        assert not status.enabled and not status.running
        assert not hasattr(client.app.state, "watcher_task")
    assert polls == [] and not state_file.exists()


def test_a_crashing_poll_does_not_stop_the_app_or_the_task(state_file, env, monkeypatch):
    env(watch_quakes="true", watch_interval_seconds="3600")

    def crash(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(watcher, "poll_once", crash)
    with TestClient(create_app()) as client:
        assert wait_for(lambda: watcher.status().last_error)
        assert "kaboom" in watcher.status().last_error
        assert watcher.status().running
        assert client.get("/healthz").json()["status"] == "ok"
    assert watcher.status().running is False


def test_stop_is_safe_to_call_twice_and_without_start():
    asyncio.run(watcher.stop(None))
    asyncio.run(watcher.stop(None))


def test_status_reflects_settings(state_file, env):
    env(
        watch_quakes="true",
        alerts_dry_run="true",
        watch_min_magnitude="5.5",
        watch_interval_seconds="60",
    )
    status = watcher.status()
    assert status.enabled and status.dry_run
    assert (status.min_magnitude, status.interval_s) == (5.5, 60)
    assert status.running is False and status.last_poll is None and status.handled == []


def test_status_lists_at_most_twenty_newest_first(state_file, env):
    env(watch_min_magnitude="6.0")
    feed = Feed()
    watcher.poll_once(feed, Recorder(), NOW)
    for i in range(25):
        feed.events.append(quake(f"q{i}", 6.5, minutes_ago=600 - i))
    watcher.poll_once(feed, Recorder(), NOW)
    handled = watcher.status().handled
    assert len(handled) == 20 and handled[0].id == "q24"


def test_state_file_is_the_source_of_truth_between_polls(state_file, env):
    env(watch_min_magnitude="6.0")
    feed, deliver = Feed(quake("a", 6.5), quake("b", 6.6)), Recorder()
    watcher.poll_once(feed, deliver, NOW)  # baseline
    data = saved(state_file)
    data["seen"].remove("b")  # an operator edits the file while the server runs
    state_file.write_text(json.dumps(data))
    assert watcher.poll_once(feed, deliver, NOW) == 1
    assert watcher.poll_once(feed, deliver, NOW) == 0
    assert deliver.events == ["b"]

    state_file.unlink()  # deleting it is a fresh start: re-baseline, no alert blast
    assert watcher.poll_once(feed, deliver, NOW) == 0
    assert deliver.events == ["b"] and set(saved(state_file)["seen"]) == {"a", "b"}
