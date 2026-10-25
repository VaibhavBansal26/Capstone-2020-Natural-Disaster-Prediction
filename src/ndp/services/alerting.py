"""Compose and dispatch emergency alerts.

Used by the manual flow (alerts page, earthquake result button) and the optional automatic
flow (NDP_AUTO_ALERT). Messages go only to the contacts configured on the server; the
facilities found near the epicentre are listed *in* the message so the recipient can act on
them. Nothing here contacts hospitals or police stations directly.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from ndp.config import get_settings
from ndp.services import ServiceError, notify
from ndp.services.geo import NearbyPlace

log = logging.getLogger("ndp.alerts")

DISCLAIMER = (
    "This is an automated statistical estimate, not an official warning. "
    "Follow guidance from your local disaster authority."
)


REPORTED_NOTE = (
    "Magnitude and location as reported by the USGS; this message is automated and is not an "
    "official warning. Follow guidance from your local disaster authority."
)


@dataclass
class DispatchResult:
    channels: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    would_send: list[str] = field(default_factory=list)
    dry_run: bool = False


def _facility_lines(places: list[NearbyPlace] | list[dict]) -> list[str]:
    def get(p, key):
        return p[key] if isinstance(p, dict) else getattr(p, key)

    lines: list[str] = []
    for kind, title, limit in (("Hospital", "Nearest hospitals", 3), ("Police", "Police", 2)):
        found = [p for p in places if str(get(p, "kind")).lower() == kind.lower()][:limit]
        if found:
            items = "; ".join(
                f"{get(p, 'name')}" + (f" ({get(p, 'phone')})" if get(p, "phone") else "")
                for p in found
            )
            lines.append(f"{title}: {items}.")
    return lines


def build_earthquake_message(
    place_name: str,
    lat: float,
    lon: float,
    magnitude: float,
    depth_km: float | None,
    on: dt.date | str | None,
    places: list,
    reported: bool = False,
) -> str:
    """`reported=True` for a real event taken from the USGS feed, False for a model estimate."""
    depth = f", depth {depth_km:g} km" if depth_km is not None else ""
    when = f", {on}" if on else ""
    kind = "USGS-reported" if reported else "estimated"
    head = (
        f"EARTHQUAKE ALERT: {kind} magnitude {magnitude:.1f} Mw{depth}{when}, "
        f"near {place_name} ({lat:.3f}, {lon:.3f})."
    )
    note = REPORTED_NOTE if reported else DISCLAIMER
    return "\n".join([head, *_facility_lines(places), note])


def build_generic_message(place_name: str, lat: float, lon: float, places: list) -> str:
    head = f"Emergency alert for {place_name} ({lat:.3f}, {lon:.3f})."
    return "\n".join([head, *_facility_lines(places), "Please follow official guidance."])


def dispatch(place_name: str, body: str, phone: str | None = None) -> DispatchResult:
    s = get_settings()
    result = DispatchResult(dry_run=s.alerts_dry_run)
    phone = (phone or "").strip() or None

    if s.email_configured:
        if s.alerts_dry_run:
            result.would_send.append(f"email to {s.alert_to_email}")
        else:
            try:
                result.channels.append(notify.send_email(f"Disaster alert: {place_name}", body))
            except ServiceError as exc:
                result.skipped.append(str(exc))
    else:
        result.skipped.append("email not configured")

    sms_ready = s.sms_configured or bool(phone and s.twilio_account_sid and s.twilio_auth_token)
    if sms_ready:
        if s.alerts_dry_run:
            result.would_send.append(f"sms to {phone or s.alert_to_number}")
        else:
            try:
                result.channels.append(notify.send_sms(body[:1500], phone))
            except ServiceError as exc:
                result.skipped.append(str(exc))
    else:
        result.skipped.append("SMS not configured")

    log.info(
        "alert for %s: sent=%s would_send=%s skipped=%s",
        place_name,
        result.channels,
        result.would_send,
        result.skipped,
    )
    return result
