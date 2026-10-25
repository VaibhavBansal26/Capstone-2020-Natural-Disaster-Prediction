from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Form, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_float, parse_text
from ndp.services import ServiceError, alerting, geo
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/alerts")


def _page_context() -> dict:
    s = get_settings()
    return {
        "email_configured": s.email_configured,
        "sms_configured": s.sms_configured,
        "dry_run": s.alerts_dry_run,
        "auto_alert": s.auto_alert,
        "meta": None,
        "trained": True,
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


@router.post("/send", name="alerts_send")
def alerts_send(
    request: Request, place: str = Form(""), message: str = Form(""), phone: str = Form("")
):
    inputs = {"place": place, "message": message, "phone": phone}
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
    outcome = alerting.dispatch(place_name, text, phone)
    return page_or_partial(
        request,
        "alerts/page.html",
        "alerts/_sent.html",
        _page_context,
        inputs=inputs,
        channels=outcome.channels,
        skipped=outcome.skipped,
        would_send=outcome.would_send,
        dry_run=outcome.dry_run,
        place=place_name,
        message=text,
    )
