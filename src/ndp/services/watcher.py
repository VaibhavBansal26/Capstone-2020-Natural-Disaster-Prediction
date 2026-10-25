"""Background watcher for new strong earthquakes (opt-in: NDP_WATCH_QUAKES=true).

Every `watch_interval_seconds` it reads the USGS feed. A reported event at or above
`watch_min_magnitude` that has not been seen before is turned into an alert through the same
`alerting` functions the manual flow uses (so dry run, channel configuration and the
"not an official warning" wording all apply), sent to the contacts configured on the server.

Design notes
- One asyncio task. All blocking work (HTTP, SMTP, file writes) happens in `poll_once`, which
  runs in a worker thread, so the event loop is never blocked.
- The loop survives any exception: it logs, records `last_error`, backs off and tries again.
  Only cancellation (`stop`) ends it.
- Dedupe state lives in `state/watcher.json` (atomic writes). On a fresh start with no state
  file the first poll only *marks* the events already in the feed as seen, so enabling the
  watcher never sends a burst of alerts for the past week. After that, a restart resumes where
  it left off, and quakes that happened while the app was down are alerted unless they are older
  than `MAX_ALERT_AGE`.
- At most one alert per event, except that a delivery that failed outright (a channel was
  configured but nothing could be sent) is retried on later polls up to `MAX_ATTEMPTS` times.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ndp.config import PROJECT_DIR, get_settings
from ndp.services import ServiceError, alerting, geo, usgs

log = logging.getLogger("ndp.watcher")

STATE_PATH = PROJECT_DIR / "state" / "watcher.json"
MAX_SEEN = 2000  # ids remembered; the USGS week feed holds far fewer
MAX_HANDLED = 20
MAX_ATTEMPTS = 3
MAX_ALERT_AGE = timedelta(hours=24)
MAX_BACKOFF_SECONDS = 1800
FEED_LIMIT = 100_000  # usgs.recent_events slices after sorting; we want the whole feed


@dataclass(frozen=True)
class HandledEvent:
    id: str
    magnitude: float
    place: str
    at: datetime
    outcome: str


@dataclass(frozen=True)
class WatcherStatus:
    enabled: bool
    running: bool
    dry_run: bool
    min_magnitude: float
    interval_s: int
    last_poll: datetime | None
    last_error: str | None
    handled: list[HandledEvent] = field(default_factory=list)


# --------------------------------------------------------------------------- persistent state


@dataclass
class _State:
    seen: list[str] = field(default_factory=list)  # insertion ordered
    handled: list[dict] = field(default_factory=list)  # newest first
    attempts: dict[str, int] = field(default_factory=dict)
    baselined: bool = False  # True once a first poll has recorded the events present at start

    def to_json(self) -> dict:
        return {
            "version": 1,
            "baselined": self.baselined,
            "seen": self.seen[-MAX_SEEN:],
            "attempts": self.attempts,
            "handled": self.handled[:MAX_HANDLED],
        }


def load_state(path: Path | None = None) -> _State:
    """Read the state file; a missing or corrupt file is a fresh start."""
    path = path or STATE_PATH
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return _State()
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable watcher state %s: %s", path, exc)
        return _State()
    if not isinstance(raw, dict) or not isinstance(raw.get("seen"), list):
        return _State()
    attempts = raw.get("attempts") if isinstance(raw.get("attempts"), dict) else {}
    handled = raw.get("handled") if isinstance(raw.get("handled"), list) else []
    return _State(
        seen=[str(i) for i in raw["seen"]],
        handled=[h for h in handled if isinstance(h, dict)][:MAX_HANDLED],
        attempts={str(k): int(v) for k, v in attempts.items() if isinstance(v, int)},
        baselined=bool(raw.get("baselined", True)),
    )


def save_state(state: _State, path: Path | None = None) -> None:
    """Atomic write: temp file in the same directory, fsync, rename."""
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".watcher-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state.to_json(), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- runtime


class _Runtime:
    """Process-wide watcher state. `lock` guards what the worker thread and status() share."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.task: asyncio.Task | None = None
        self.state: _State | None = None
        self.last_poll: datetime | None = None
        self.last_error: str | None = None
        self.failures = 0


_rt = _Runtime()


def reset() -> None:
    """Forget in-memory state; the next poll reloads the state file (a simulated restart)."""
    with _rt.lock:
        _rt.state = None
        _rt.last_poll = _rt.last_error = None
        _rt.failures = 0


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _handled_from_dict(d: dict) -> HandledEvent | None:
    try:
        at = datetime.fromisoformat(str(d["at"]))
        return HandledEvent(
            id=str(d["id"]),
            magnitude=float(d["magnitude"]),
            place=str(d.get("place", "")),
            at=at if at.tzinfo else at.replace(tzinfo=UTC),
            outcome=str(d.get("outcome", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def status() -> WatcherStatus:
    s = get_settings()
    with _rt.lock:
        state = _rt.state
        last_poll, last_error = _rt.last_poll, _rt.last_error
        running = _rt.task is not None and not _rt.task.done()
        handled_raw = list(state.handled) if state else None
    if handled_raw is None:  # never started in this process: show history from the state file
        handled_raw = load_state().handled
    handled = [h for h in (_handled_from_dict(d) for d in handled_raw) if h][:MAX_HANDLED]
    return WatcherStatus(
        enabled=s.watch_quakes,
        running=running,
        dry_run=s.alerts_dry_run,
        min_magnitude=s.watch_min_magnitude,
        interval_s=s.watch_interval_seconds,
        last_poll=last_poll,
        last_error=last_error,
        handled=handled,
    )


# --------------------------------------------------------------------------- handling one event


def _deliver(event: usgs.QuakeEvent) -> tuple[str, bool]:
    """Build and dispatch the alert for one real event. Returns (outcome text, retryable)."""
    s = get_settings()
    place_name = event.place
    try:
        place_name = geo.reverse_geocode(event.lat, event.lon).name or event.place
    except ServiceError as exc:
        log.info("reverse geocoding failed for %s, using the USGS place name: %s", event.id, exc)
    try:
        places = geo.nearby_services(event.lat, event.lon, s.alert_radius_km)
    except ServiceError as exc:
        log.info("nearby services unavailable for %s: %s", event.id, exc)
        places = []
    body = alerting.build_earthquake_message(
        place_name,
        event.lat,
        event.lon,
        event.magnitude,
        event.depth_km,
        event.time.date(),
        places,
        reported=True,
    )
    if event.url:
        body += f"\nUSGS event page: {event.url}"
    result = alerting.dispatch(place_name, body)
    if result.would_send:
        return "dry run, would send: " + "; ".join(result.would_send), False
    if result.channels:
        return "sent: " + "; ".join(result.channels), False
    reasons = "; ".join(result.skipped) or "no channel available"
    configured = not all("not configured" in r for r in result.skipped)
    return f"not sent: {reasons}", configured


def _record(state: _State, event: usgs.QuakeEvent, outcome: str, now: datetime) -> None:
    state.handled.insert(
        0,
        {
            "id": event.id,
            "magnitude": event.magnitude,
            "place": event.place,
            "at": now.isoformat(timespec="seconds"),
            "outcome": outcome,
        },
    )
    del state.handled[MAX_HANDLED:]


def poll_once(
    fetch: Callable[[int], list[usgs.QuakeEvent]] | None = None,
    deliver: Callable[[usgs.QuakeEvent], tuple[str, bool]] | None = None,
    now: datetime | None = None,
) -> int:
    """One blocking poll. Returns the number of alerts attempted. Raises ServiceError if the feed
    cannot be read (the loop records it). `fetch`, `deliver` and `now` are injection points for
    tests (the state file location is the module constant `STATE_PATH`). Runs in a worker thread
    when called from the loop."""
    s = get_settings()
    fetch = fetch or usgs.recent_events
    deliver = deliver or _deliver
    now = now or _utcnow()

    events = fetch(FEED_LIMIT)

    # The file is the source of truth, re-read on every poll: an operator can clear or edit it
    # while the server runs (a deleted file means "fresh start", so no alert blast either).
    state = load_state()
    with _rt.lock:
        _rt.state = state
    if not state.baselined:
        # Fresh start: remember what is already there, alert on nothing.
        state.seen = [e.id for e in events][-MAX_SEEN:]
        state.baselined = True
        with _rt.lock:
            save_state(state)
        log.info("watcher baseline: %d existing events marked as seen", len(events))
        return 0

    seen = set(state.seen)
    # Oldest first, so a burst is alerted in the order it happened.
    fresh = sorted((e for e in events if e.id not in seen), key=lambda e: e.time)
    alerted = 0
    dirty = False
    for event in fresh:
        if event.magnitude < s.watch_min_magnitude:
            _mark_seen(state, event.id)
            dirty = True
            continue
        if now - event.time > MAX_ALERT_AGE:
            log.info("not alerting for %s: older than %s", event.id, MAX_ALERT_AGE)
            _mark_seen(state, event.id)
            dirty = True
            continue
        try:
            outcome, retryable = deliver(event)
        except Exception as exc:
            log.exception("alert for %s failed", event.id)
            outcome, retryable = f"failed: {type(exc).__name__}: {exc}", True
        alerted += 1
        attempts = state.attempts.get(event.id, 0) + 1
        if retryable and attempts < MAX_ATTEMPTS:
            state.attempts[event.id] = attempts
            outcome += f" (attempt {attempts} of {MAX_ATTEMPTS}, will retry)"
        else:
            state.attempts.pop(event.id, None)
            _mark_seen(state, event.id)
        with _rt.lock:
            _record(state, event, outcome, now)
            save_state(state)  # persist before moving on: a crash must not re-alert
    if dirty:
        with _rt.lock:
            save_state(state)
    return alerted


def _mark_seen(state: _State, event_id: str) -> None:
    state.seen.append(event_id)
    del state.seen[:-MAX_SEEN]


# --------------------------------------------------------------------------- the loop


def next_delay(interval: float, failures: int) -> float:
    """Normal interval after a good poll; exponential backoff (capped) after failures."""
    if failures <= 0:
        return interval
    return min(interval * 2 ** min(failures, 5), max(interval, MAX_BACKOFF_SECONDS))


async def run_loop(
    poll: Callable[[], object] | None = None,
    interval: float | None = None,
    sleep: Callable[[float], object] = asyncio.sleep,
) -> None:
    """Poll forever. Never raises except CancelledError."""
    poll = poll or poll_once
    while True:
        try:
            await asyncio.to_thread(poll)
            with _rt.lock:
                _rt.last_poll, _rt.last_error, _rt.failures = _utcnow(), None, 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with _rt.lock:
                _rt.last_poll = _utcnow()
                _rt.last_error = f"{type(exc).__name__}: {exc}"
                _rt.failures += 1
            log.warning("watcher poll failed (%d in a row): %s", _rt.failures, exc)
        base = interval if interval is not None else get_settings().watch_interval_seconds
        await sleep(next_delay(base, _rt.failures))


async def start(app) -> None:
    s = get_settings()
    if not s.watch_quakes:
        log.info("real-event watcher is off (set NDP_WATCH_QUAKES=true to enable)")
        return
    if _rt.task is not None and not _rt.task.done():
        return
    reset()
    _rt.task = asyncio.create_task(run_loop(), name="ndp-quake-watcher")
    app.state.watcher_task = _rt.task
    log.info(
        "real-event watcher started: magnitude >= %s every %ss (dry run: %s)",
        s.watch_min_magnitude,
        s.watch_interval_seconds,
        s.alerts_dry_run,
    )


async def stop(app) -> None:
    task, _rt.task = _rt.task, None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    log.info("real-event watcher stopped")
