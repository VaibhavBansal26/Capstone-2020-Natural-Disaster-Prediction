from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Form, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_float, parse_text
from ndp.services import ServiceError, geo, notify
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/alerts")


def _page_context() -> dict:
    s = get_settings()
    return {
        "email_configured": s.email_configured,
        "sms_configured": s.sms_configured,
        "meta": None,
        "trained": True,
    }


@router.get("", name="alerts_page")
def alerts_page(request: Request):
    return render(request, "alerts/page.html", **_page_context())


@router.post("/search", name="alerts_search")
def alerts_search(request: Request, place: str = Form(""), radius_km: str = Form("15")):
    inputs = {"place": place, "radius_km": radius_km}
    try:
        query = parse_text(place, "Place", 200)
        radius = parse_float(radius_km, "Radius (km)", 0.5, 50)
        found = geo.geocode(query)
        places = geo.nearby_services(found.lat, found.lon, radius)
    except (FormError, ServiceError) as exc:
        return page_or_partial(
            request,
            "alerts/page.html",
            "alerts/_results.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "alerts/page.html",
        "alerts/_results.html",
        _page_context,
        inputs=inputs,
        place=asdict(found),
        places=[asdict(p) for p in places],
        radius_km=radius,
    )


@router.post("/send", name="alerts_send")
def alerts_send(
    request: Request, place: str = Form(""), message: str = Form(""), phone: str = Form("")
):
    inputs = {"place": place, "message": message, "phone": phone}
    channels: list[str] = []
    skipped: list[str] = []
    try:
        place_name = parse_text(place, "Place", 300)
        text = parse_text(message, "Message", 1200)
    except FormError as exc:
        return page_or_partial(
            request,
            "alerts/page.html",
            "alerts/_sent.html",
            _page_context,
            inputs=inputs,
            error=str(exc),
        )
    body = f"Disaster alert for {place_name}\n\n{text}"
    s = get_settings()
    if s.email_configured:
        try:
            channels.append(notify.send_email(f"Disaster alert: {place_name}", body))
        except ServiceError as exc:
            skipped.append(str(exc))
    else:
        skipped.append("email not configured")
    if s.sms_configured or (phone.strip() and s.twilio_account_sid):
        try:
            channels.append(notify.send_sms(body[:1500], phone.strip() or None))
        except ServiceError as exc:
            skipped.append(str(exc))
    else:
        skipped.append("SMS not configured")
    return page_or_partial(
        request,
        "alerts/page.html",
        "alerts/_sent.html",
        _page_context,
        inputs=inputs,
        channels=channels,
        skipped=skipped,
        place=place_name,
        message=text,
    )
