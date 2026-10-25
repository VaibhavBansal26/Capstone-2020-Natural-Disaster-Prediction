from __future__ import annotations

from fastapi import APIRouter, Form, Request

from ndp.forms import FormError, parse_float
from ndp.ml import flood, registry
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/flood")
PLOTS = [
    ("River level history", "generated/flood/f1_level_history.png"),
    ("3-hour forecast vs actual (held-out month)", "generated/flood/f2_forecast_vs_actual.png"),
    ("Error by forecast horizon: model vs persistence", "generated/flood/f3_skill_by_horizon.png"),
]
LEVEL_MAX_M = 15
RAIN_MAX_MM = 500
# (form field, label used in error messages, kind). Levels in metres, rain in millimetres.
FIELDS = [
    ("level_now", "Level now (m)", "level"),
    ("level_1h_ago", "Level 1 hour ago (m)", "level"),
    ("level_3h_ago", "Level 3 hours ago (m)", "level"),
    ("rain_3h", "Rain last 3 hours (mm)", "rain"),
    ("rain_6h", "Rain last 6 hours (mm)", "rain"),
    ("rain_24h", "Rain last 24 hours (mm)", "rain"),
]
# A quiet, slightly damp day for this gauge: the median level in the record is 1.45 m,
# and 95 % of 24-hour totals are under about 6 mm.
DEFAULTS = {
    "level_now": "1.42",
    "level_1h_ago": "1.41",
    "level_3h_ago": "1.40",
    "rain_3h": "2",
    "rain_6h": "3",
    "rain_24h": "5",
}


def _page_context() -> dict:
    return {
        "meta": registry.metadata(flood.MODEL),
        "trained": registry.is_trained(flood.MODEL),
        "plots": PLOTS,
        "threshold_m": flood.FLOOD_THRESHOLD_M,
        "defaults": DEFAULTS,
        "level_max_m": LEVEL_MAX_M,
        "rain_max_mm": RAIN_MAX_MM,
    }


@router.get("", name="flood_page")
def flood_page(request: Request):
    return render(request, "flood/page.html", **_page_context())


@router.post("/predict", name="flood_predict")
def flood_predict(
    request: Request,
    level_now: str = Form(""),
    level_1h_ago: str = Form(""),
    level_3h_ago: str = Form(""),
    rain_3h: str = Form(""),
    rain_6h: str = Form(""),
    rain_24h: str = Form(""),
):
    raw = {
        "level_now": level_now,
        "level_1h_ago": level_1h_ago,
        "level_3h_ago": level_3h_ago,
        "rain_3h": rain_3h,
        "rain_6h": rain_6h,
        "rain_24h": rain_24h,
    }
    try:
        values = {
            name: parse_float(raw[name], label, 0, LEVEL_MAX_M if kind == "level" else RAIN_MAX_MM)
            for name, label, kind in FIELDS
        }
    except FormError as exc:
        return page_or_partial(
            request,
            "flood/page.html",
            "flood/_result.html",
            _page_context,
            inputs=raw,
            error=str(exc),
        )
    result = flood.predict(
        values["level_now"],
        values["level_1h_ago"],
        values["level_3h_ago"],
        values["rain_3h"],
        values["rain_6h"],
        values["rain_24h"],
    )
    return page_or_partial(
        request,
        "flood/page.html",
        "flood/_result.html",
        _page_context,
        inputs=values,
        result=result,
    )
