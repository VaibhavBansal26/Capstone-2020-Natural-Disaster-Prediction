from __future__ import annotations

import re
import secrets
from dataclasses import asdict

from fastapi import APIRouter, Depends, Form, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_float, parse_text
from ndp.services import ServiceError, alerting, geo, llm, translate
from ndp.services.ratelimit import RateLimiter
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/alerts")

# notify.send_sms is given body[:1500]; keep the SMS inside that limit on purpose.
SMS_LIMIT = 1500
SMS_TRANSLATIONS_NOTE = "[Translations are in the email version.]"
MAX_TRANSLATION_CHARS = 4000

# Sending is unauthenticated unless NDP_ALERTS_PASSCODE is set, so cap it per client address.
SEND_LIMIT = 5
SEND_WINDOW_SECONDS = 600.0
send_limiter = RateLimiter(SEND_LIMIT, SEND_WINDOW_SECONDS)
translate_limiter = RateLimiter(6, 600.0)  # each preview makes up to twelve Claude calls
PHONE_PATTERN = re.compile(r"^\+?[0-9][0-9 ()-]{6,18}[0-9]$")


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _translation_ui() -> dict:
    """Context for the language picker: hidden (replaced by a note) without a Claude key."""
    return {
        "llm_enabled": llm.is_configured(),
        "languages": [{"code": c, "name": n} for c, n in translate.LANGUAGES.items()],
    }


def _page_context() -> dict:
    s = get_settings()
    return {
        "email_configured": s.email_configured,
        "sms_configured": s.sms_configured,
        "dry_run": s.alerts_dry_run,
        "watcher_on": s.watch_quakes,
        "passcode_required": bool(s.alerts_passcode),
        "custom_phone_allowed": s.alerts_allow_custom_phone,
        "meta": None,
        "trained": True,
        **_translation_ui(),
    }


@router.get("", name="alerts_page")
def alerts_page(request: Request):
    return render(request, "alerts/page.html", **_page_context())


def _results_response(request: Request, inputs: dict, found, radius: float, make_message):
    """Look up facilities once and always render the alert form. A failed lookup becomes a
    warning rather than an error, so an alert can still be written and sent without a list."""
    warning = None
    try:
        places = geo.nearby_services(found.lat, found.lon, radius)
    except ServiceError as exc:
        places, warning = [], str(exc)
    return page_or_partial(
        request,
        "alerts/page.html",
        "alerts/_results.html",
        _page_context,
        inputs=inputs,
        place=asdict(found),
        places=[asdict(p) for p in places],
        radius_km=radius,
        warning=warning,
        message=make_message(found, places),
        passcode_required=bool(get_settings().alerts_passcode),
        custom_phone_allowed=get_settings().alerts_allow_custom_phone,
        **_translation_ui(),
    )


def _error_response(request: Request, inputs: dict, exc: Exception):
    return page_or_partial(
        request,
        "alerts/page.html",
        "alerts/_results.html",
        _page_context,
        inputs=inputs,
        error=str(exc),
    )


@router.post("/search", name="alerts_search")
def alerts_search(request: Request, place: str = Form(""), radius_km: str = Form("15")):
    inputs = {"place": place, "radius_km": radius_km}
    try:
        query = parse_text(place, "Place", 200)
        radius = parse_float(radius_km, "Radius (km)", 0.5, 50)
        found = geo.geocode(query)
        return _results_response(
            request,
            inputs,
            found,
            radius,
            lambda f, places: alerting.build_generic_message(f.name, f.lat, f.lon, places),
        )
    except (FormError, ServiceError) as exc:
        return _error_response(request, inputs, exc)


@router.post("/nearby", name="alerts_nearby")
def alerts_nearby(
    request: Request,
    lat: str = Form(""),
    lon: str = Form(""),
    radius_km: str = Form(""),
    magnitude: str = Form(""),
    depth: str = Form(""),
    date_: str = Form("", alias="date"),
    source: str = Form(""),
):
    """Hospitals and police around a coordinate, with an earthquake alert message prefilled
    when a magnitude is supplied. This is the target of the earthquake result's button."""
    inputs = {"lat": lat, "lon": lon, "radius_km": radius_km, "magnitude": magnitude}
    try:
        f_lat = parse_float(lat, "Latitude", -90, 90)
        f_lon = parse_float(lon, "Longitude", -180, 180)
        radius = parse_float(
            radius_km or str(get_settings().alert_radius_km), "Radius (km)", 0.5, 50
        )
        found = geo.reverse_geocode(f_lat, f_lon)
        if magnitude.strip():
            f_mag = parse_float(magnitude, "Magnitude", 0, 12)
            f_depth = parse_float(depth, "Depth (km)", 0, 800) if depth.strip() else None
            when = date_.strip() or None

            def make_message(f, places):
                return alerting.build_earthquake_message(
                    f.name, f_lat, f_lon, f_mag, f_depth, when, places, reported=source == "usgs"
                )
        else:

            def make_message(f, places):
                return alerting.build_generic_message(f.name, f_lat, f_lon, places)

        return _results_response(request, inputs, found, radius, make_message)
    except (FormError, ServiceError) as exc:
        return _error_response(request, inputs, exc)


_LANGUAGES_FIELD = Form(None)


@router.post("/translate", name="alerts_translate")
def alerts_translate(
    request: Request, message: str = Form(""), languages: list[str] | None = _LANGUAGES_FIELD
):
    """Machine-translate the message into the ticked languages and return one editable textarea
    per language (``translation_<code>``). The partial is swapped *inside* the send form, so a
    person reads and edits every translation before pressing Send. Always the bare partial: this
    control exists only with HTMX."""
    if not llm.is_configured():
        return render(
            request,
            "alerts/_translations.html",
            error="Translation needs Claude, which is not configured on this server.",
        )
    allowed, retry_after = translate_limiter.hit(_client_key(request))
    if not allowed:
        return render(
            request,
            "alerts/_translations.html",
            error=f"Too many translation requests. Try again in {int(retry_after) + 1} seconds.",
        )
    try:
        text = parse_text(message, "Message", 1200)
        languages = languages or []
        unknown = [c for c in languages if c not in translate.LANGUAGES]
        if unknown:
            raise FormError("Unsupported language: " + ", ".join(unknown))
        picked = [c for c in translate.LANGUAGES if c in languages]
        if not picked:
            raise FormError("Choose at least one language.")
        results = translate.translate_many(text, picked)
    except (FormError, ServiceError) as exc:
        return render(request, "alerts/_translations.html", error=str(exc))
    return render(request, "alerts/_translations.html", translations=list(results.values()))


async def _translation_fields(request: Request) -> dict[str, str]:
    """Every non-empty ``translation_<code>`` field of the send form, in the picker's order.
    Read from the (cached) form body so the endpoint itself stays a plain ``def``."""
    form = await request.form()
    found: dict[str, str] = {}
    for code in translate.LANGUAGES:
        value = form.get(f"translation_{code}")
        if isinstance(value, str) and value.strip():
            found[code] = value.strip()
    return found


_TRANSLATIONS = Depends(_translation_fields)


def compose_bodies(text: str, translations: dict[str, str]) -> tuple[str, str, bool]:
    """``(email_body, sms_body, sms_is_english_only)``.

    The English text always comes first; each translation follows under a language heading.
    SMS is limited to 1500 characters (and Indic/Arabic/CJK text costs many SMS segments), so:
    if the whole message fits, SMS carries all of it; if not, SMS carries only the English
    (translation-free) text plus a note that translations are in the email."""
    if not translations:
        return text, text, False
    parts = [text]
    for code, body in translations.items():
        parts.append(f"=== {translate.LANGUAGES[code]} ===\n{body}")
    email_body = "\n\n".join(parts)
    if len(email_body) <= SMS_LIMIT:
        return email_body, email_body, False
    sms_body = f"{text[: SMS_LIMIT - len(SMS_TRANSLATIONS_NOTE) - 1]}\n{SMS_TRANSLATIONS_NOTE}"
    return email_body, sms_body, True


def _dispatch(place_name: str, email_body: str, sms_body: str, phone: str | None):
    """Send (or dry-run). SMS gets its own, shorter body only when the combined text is too long."""
    separate = sms_body != email_body
    outcome = alerting.dispatch(
        place_name, email_body, phone, sms_body=sms_body if separate else None
    )
    return outcome, separate


@router.post("/send", name="alerts_send")
def alerts_send(
    request: Request,
    place: str = Form(""),
    message: str = Form(""),
    phone: str = Form(""),
    passcode: str = Form(""),
    translations: dict[str, str] = _TRANSLATIONS,
):
    inputs = {"place": place, "message": message, "phone": phone}
    settings = get_settings()
    try:
        if settings.alerts_passcode and not secrets.compare_digest(
            passcode.encode(), settings.alerts_passcode.encode()
        ):
            raise FormError("The alert passcode is missing or wrong.")
        place_name = parse_text(place, "Place", 300)
        text = parse_text(message, "Message", 1200)
        for code, body in translations.items():
            if len(body) > MAX_TRANSLATION_CHARS:
                raise FormError(
                    f"The {translate.LANGUAGES[code]} translation is too long "
                    f"(max {MAX_TRANSLATION_CHARS} characters)."
                )
    except FormError as exc:
        return page_or_partial(
            request,
            "alerts/page.html",
            "alerts/_sent.html",
            _page_context,
            inputs=inputs,
            sent=True,
            error=str(exc),
        )
    allowed, retry_after = send_limiter.hit(_client_key(request))
    if not allowed:
        return page_or_partial(
            request,
            "alerts/page.html",
            "alerts/_sent.html",
            _page_context,
            inputs=inputs,
            sent=True,
            error=(
                f"Too many alerts from this address. Try again in {int(retry_after) + 1} seconds."
            ),
        )
    custom_phone = phone.strip()
    ignored_phone = False
    if custom_phone and not settings.alerts_allow_custom_phone:
        custom_phone, ignored_phone = "", True  # never text an arbitrary number from a web form
    elif custom_phone and not PHONE_PATTERN.match(custom_phone):
        return page_or_partial(
            request,
            "alerts/page.html",
            "alerts/_sent.html",
            _page_context,
            inputs=inputs,
            sent=True,
            error="That phone number does not look valid (digits, spaces, + ( ) - only).",
        )
    email_body, sms_body, sms_english_only = compose_bodies(text, translations)
    outcome, sms_separate = _dispatch(place_name, email_body, sms_body, custom_phone or None)
    if ignored_phone:
        outcome.skipped.append("custom phone number ignored (disabled on this server)")
    return page_or_partial(
        request,
        "alerts/page.html",
        "alerts/_sent.html",
        _page_context,
        inputs=inputs,
        sent=True,
        channels=outcome.channels,
        skipped=outcome.skipped,
        would_send=outcome.would_send,
        dry_run=outcome.dry_run,
        place=place_name,
        message=text,
        included_languages=[
            {
                "code": c,
                "name": translate.LANGUAGES[c],
                "text": t,
                "rtl": c in translate.RTL_LANGUAGES,
            }
            for c, t in translations.items()
        ],
        sms_english_only=sms_english_only and sms_separate,
    )
