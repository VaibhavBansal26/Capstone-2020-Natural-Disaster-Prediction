"""Home page and the model-comparison page."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_date, parse_float
from ndp.ml import compare, covid, flood, hazard, hurricane, registry, sentiment, storm, tsunami
from ndp.web import page_or_partial, render

router = APIRouter()

UNITS = {"earthquake": "magnitude", "tsunami": "magnitude", "flood": "m", "hurricane": "kt"}
TITLES = {
    "earthquake": "Earthquake magnitude",
    "tsunami": "Tsunami source magnitude",
    "flood": "River level",
    "hurricane": "Maximum wind",
}


def _status(model_name: str) -> str:
    return "ready" if registry.is_trained(model_name) else "needs training"


@router.get("/", name="index")
def index(request: Request):
    settings = get_settings()
    modules = [
        {
            "name": "Earthquake",
            "blurb": "Historical earthquake hazard by place, plus the live USGS feed.",
            "endpoint": "earthquake_page",
            "status": _status(hazard.MODEL),
        },
        {
            "name": "Tsunami",
            "blurb": "Source magnitude from location and maximum wave height (NOAA records).",
            "endpoint": "tsunami_page",
            "status": _status(tsunami.MODEL),
        },
        {
            "name": "Flood",
            "blurb": "River-level forecast for the Hoppers Crossing gauge, 1 to 6 hours ahead.",
            "endpoint": "flood_page",
            "status": _status(flood.MODEL),
        },
        {
            "name": "Hurricane",
            "blurb": "Maximum sustained wind from date and position, trained on HURDAT2.",
            "endpoint": "hurricane_page",
            "status": _status(hurricane.MODEL),
        },
        {
            "name": "Storm",
            "blurb": "Classify weather conditions from temperature, pressure, humidity and wind.",
            "endpoint": "storm_page",
            "status": _status(storm.MODEL),
        },
        {
            "name": "Weather",
            "blurb": "Live nearest-neighbour check of a city's weather against nearby stations.",
            "endpoint": "weather_page",
            "status": "ready" if settings.openweather_api_key else "not configured",
        },
        {
            "name": "Alerts",
            "blurb": "Find hospitals and police near a place and send an alert by email or SMS.",
            "endpoint": "alerts_page",
            "status": "ready"
            if (settings.email_configured or settings.sms_configured)
            else "not configured",
        },
        {
            "name": "Live hazards",
            "blurb": "Global alerts, a weather outlook for any place, and the alert watcher.",
            "endpoint": "live_page",
            "status": "ready",
        },
        {
            "name": "Assistant",
            "blurb": "Ask in plain language; Claude uses live tools and cites guidance.",
            "endpoint": "assistant_page",
            "status": "ready" if settings.llm_configured else "not configured",
        },
        {
            "name": "Sentiment",
            "blurb": "Sentiment of public posts about seven disasters vs their monetary impact.",
            "endpoint": "sentiment_page",
            "status": _status(sentiment.SCORES),
        },
        {
            "name": "Compare models",
            "blurb": "Random forest, neural network, SVM and linear regression per hazard.",
            "endpoint": "compare_page",
            "status": "ready"
            if all(registry.is_trained(f"compare_{h}") for h in compare.HAZARDS)
            else "needs training",
        },
    ]
    if settings.enable_covid:
        modules.append(
            {
                "name": "COVID-19",
                "blurb": "Live totals, the 2020 India case curve, and nearest ICMR test centres.",
                "endpoint": "covid_page",
                "status": _status(covid.MODEL),
            }
        )
    return render(request, "index.html", modules=modules)


def _compare_context() -> dict:
    hazards = []
    for key, spec in compare.HAZARDS.items():
        meta = registry.metadata(f"compare_{key}")
        plot = f"generated/compare/{key}_accuracy.png"
        if not (get_settings().generated_dir / "compare" / f"{key}_accuracy.png").exists():
            plot = None
        hazards.append(
            {
                "key": key,
                "title": TITLES[key],
                "input_fields": spec.input_fields,
                "metrics": (meta or {}).get("metrics"),
                "plot": plot,
                "unit": UNITS[key],
            }
        )
    return {"hazards": hazards, "trained": all(h["metrics"] for h in hazards)}


@router.get("/compare", name="compare_page")
def compare_page(request: Request):
    return render(request, "compare/page.html", **_compare_context())


@router.post("/compare/{hazard}", name="compare_predict")
async def compare_predict(request: Request, hazard: str):
    ctx = _compare_context()
    hazard_ctx = next((h for h in ctx["hazards"] if h["key"] == hazard), None)
    if hazard_ctx is None:
        return page_or_partial(
            request,
            "compare/page.html",
            "compare/_result.html",
            _compare_context,
            hazard=None,
            inputs={},
            error=f"Unknown hazard {hazard!r}.",
        )
    form = await request.form()
    inputs: dict[str, str] = {}
    values: list[float] = []
    try:
        for field in hazard_ctx["input_fields"]:
            raw = form.get(field.name)
            inputs[field.name] = str(raw or "")
            if field.kind == "date":
                d = parse_date(raw, field.label)
                values.append(float(f"{d.year:04d}{d.month:02d}{d.day:02d}"))
            else:
                values.append(parse_float(raw, field.label))
        results = compare.predict(hazard, values)
    except FormError as exc:
        return page_or_partial(
            request,
            "compare/page.html",
            "compare/_result.html",
            _compare_context,
            hazard=hazard_ctx,
            inputs=inputs,
            error=str(exc),
        )
    return page_or_partial(
        request,
        "compare/page.html",
        "compare/_result.html",
        _compare_context,
        hazard=hazard_ctx,
        inputs=inputs,
        results=results,
        unit=UNITS[hazard],
    )
