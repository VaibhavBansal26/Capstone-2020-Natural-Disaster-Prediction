# Architecture

A FastAPI application that renders server-side Jinja2 pages, enhanced with HTMX. Models are
trained once from the command line and loaded from disk at request time. Nothing trains, downloads
or writes files inside a web request.

## Repository map

```
src/ndp/                    the application package
  main.py                   app factory, security headers, lifespan (warm-up, watcher), ndp-serve
  config.py                 pydantic-settings; every NDP_* variable lives here
  data.py                   dataset manifest and `ndp-fetch-data`
  guidance.py               preparedness-guidance corpus: fetch (`ndp-fetch-guidance`) and search
  forms.py, web.py          form parsing; Jinja rendering, HTMX partial vs full page
  routers/                  one module per page, plus api.py (JSON)
  services/                 clients and logic around the outside world (see below)
  ml/                       training and prediction, one module per model (see models.md)
  evals/                    `ndp-eval`: assistant and triage eval runners
  templates/                Jinja2: <page>/page.html, <page>/_result.html partials, partials/
  static/                   css/, js/, img/disasters/, generated/ (training plots, git-ignored)
frontend/                   Tailwind config, input CSS and build.sh (compiles static/css/tailwind.css)
tests/                      ml/, services/, web/, js/ plus factories.py and fixtures/
data/                       raw/ (fetched, git-ignored), guidance/ (committed), evals/ (committed)
deploy/                     Dockerfile, docker-compose.yml, entrypoint.sh
docs/                       these documents
render.yaml                 Render blueprint (must stay at the root)
```

## Request flow

1. A page route (`GET /flood`) renders `flood/page.html` with the form and a model card.
2. The form posts to `POST /flood/predict`. With HTMX (`HX-Request: true`) the handler returns only
   `flood/_result.html`, which is swapped into `#result`. Without JavaScript the same handler
   returns the full page with the result included (`web.page_or_partial`).
3. Handlers validate input with `forms.py` (`FormError` becomes an inline message, never a 422),
   call a prediction function that loads its artifact through `ml/registry.py`, and render.
4. A missing artifact raises `ModelNotTrained`, which becomes a 503 page telling the user to run
   `ndp-train` (or a JSON error under `/api/`).

## Layers and rules

- **ml/** knows nothing about HTTP. `train()` writes artifacts (joblib plus a JSON sidecar with
  metrics) and plots; `predict()` loads them. Every model records a baseline and a skill score.
- **services/** wraps external systems and degrades with a `ServiceError`: USGS, GDACS, Open-Meteo,
  OpenStreetMap (Nominatim, Overpass), OpenWeatherMap, disease.sh, SMTP, Twilio and Claude. All
  Claude calls go through `services/llm.py`, which fixes the model, refusal fallbacks and error
  mapping in one place.
- **routers/** are thin: parse, call, render. They do not contain model code.
- **Alerts never send themselves.** `services/alerting.py` composes and dispatches; the manual
  flow needs a click and a confirm; the optional watcher (`services/watcher.py`) alerts only for new
  real USGS events, once each, and honours dry run. The assistant has no tool that can send.

## Frontend conventions

- Tailwind is compiled (`frontend/build.sh`) to `static/css/tailwind.css`. Rebuild after adding
  utility classes to a template or script.
- `data-speak="text"` on a result root is read aloud by `static/js/voice.js`;
  `data-speak-priority="alert"` is read even when voice is off (after the user opted in once).
  `data-voice-command="phrase"` makes an element clickable by voice and `data-voice-confirm`
  requires a spoken "confirm".
- Maps use Leaflet through the declarative helpers in `static/js/app.js` (`data-ndp-map`,
  `data-markers-src`, `data-usgs-url`, `data-pick-lat`/`data-pick-lon`).
- Pages must work at 375 px wide with no horizontal scroll.

## Endpoints

HTML pages and HTMX partials (all return HTML) and the JSON API under `/api/v1`. Interactive
API docs are at `/docs`.

| Area | Endpoints |
|---|---|
| Home | `GET /` |
| Earthquake | `GET /earthquake`, `POST /earthquake/predict`, `GET /earthquake/events`, `GET /earthquake/graphs` |
| Flood | `GET /flood`, `POST /flood/predict` |
| Tsunami, hurricane, storm | `GET /<hazard>`, `POST /<hazard>/predict` |
| Weather | `GET /weather`, `POST /weather/predict` |
| Live | `GET /live`, `GET /live/gdacs`, `POST /live/forecast`, `GET /live/watcher` |
| Assistant | `GET /assistant`, `POST /assistant/ask`, `POST /assistant/reset` |
| Alerts | `GET /alerts`, `POST /alerts/search`, `POST /alerts/nearby`, `POST /alerts/translate`, `POST /alerts/send` |
| Sentiment | `GET /sentiment`, `GET /sentiment/{slug}`, `POST /sentiment/score`, `POST /sentiment/triage`, `POST /sentiment/triage/sample` |
| Compare | `GET /compare`, `POST /compare/{hazard}` |
| COVID-19 | `GET /covid`, `POST /covid/predict`, `GET /covid/stats`, `POST /covid/centres`, `GET /covid/maps`, `GET /covid/data/{layer}` |
| JSON API | `POST /api/v1/earthquake/hazard`, `/flood/forecast`, `/tsunami/predict`, `/hurricane/predict`, `/storm/predict` |
| Ops | `GET /healthz` |

## Tests

`uv run pytest -q` runs everything. `tests/ml` covers training and prediction on small data,
`tests/services` the external clients (against recorded fixtures and fakes), `tests/web` every page
and form, and `tests/js` runs the Node tests for the voice interpreter. No test needs the network,
trained models or an API key. `tests/factories.py` builds model-result objects for the web tests.
