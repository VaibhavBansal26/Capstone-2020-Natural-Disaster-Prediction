# Phase 2 and 3 contract

Addendum to `docs/CONTRACT.md`. Names here are binding for everyone working in parallel.
Principles from the first contract still hold: nothing trains in a request, no secrets in code,
user input must reach every model, estimates are never presented as forecasts or warnings.

Shared files that are **not** to be edited by feature agents (the coordinator edits them):
`pyproject.toml`, `uv.lock`, `src/ndp/config.py` (all settings already exist), `src/ndp/main.py`,
`templates/base.html`, `routers/pages.py` (module cards), `README.md`, `.env.example`.
Need a setting or dependency? Say so in your final report instead of editing.

---

## Part A. Phase 2: honest ML (`src/ndp/ml/`)

### A1. Honest metrics for every regressor
- `features.baseline_metrics(y_train, y_test) -> dict(baseline_rmse, baseline_r2)`:
  `DummyRegressor(strategy="mean")` fit on the train target.
- Every `train()` metadata gains: `baseline_rmse`, `skill` (= `1 - rmse / baseline_rmse`, may be
  negative), `split` (e.g. `"chronological, last 20% held out"` or `"5-fold cross-validation"`),
  and for cross-validated models `cv_r2_mean`, `cv_r2_std`.
- Classifiers (storm) record `baseline_accuracy` (majority class) and `accuracy`.
- Splits must be honest: time-ordered data is split chronologically, never shuffled.

### A2. `ndp.ml.hazard` (replaces the earthquake magnitude regression on the earthquake page)
- `MODEL = "earthquake_hazard"`.
- `train()`: load `database.csv` (rows with `Type == "Earthquake"`), keep lat, lon, depth,
  magnitude, year-fraction; save compact numpy arrays plus metadata `{n_events, years_span,
  mc: 5.5, b_value_global, year_min, year_max}`. Plots under `generated/earthquake/`:
  `hz1_seismicity.png` (epicentre scatter, lon vs lat, coloured by magnitude) and
  `hz2_gutenberg_richter.png` (log10 cumulative count vs magnitude with the fitted line).
- `@dataclass(frozen=True) HazardEstimate`: `lat, lon, radius_km, n_events, years,
  annual_rate_m55, b_value, b_is_global (bool), annual_rate: dict[str, float]`,
  `prob_1y: dict[str, float]`, `prob_10y: dict[str, float]` (keys `"6.0"`, `"7.0"`, `"8.0"`),
  `max_magnitude: float | None`, `max_date: str | None`, `depth_median_km: float | None`,
  `nearest_event: dict | None` (`distance_km, magnitude, year`), `level: str`, `elevated: bool`.
- `predict(lat: float, lon: float, radius_km: float = 300.0) -> HazardEstimate`: events within the
  great-circle radius (vectorised haversine); Aki maximum-likelihood b-value for events >= Mc when
  `n_events >= 50` else fall back to the global b with `b_is_global=True`; Gutenberg-Richter rate
  for M >= m = `annual_rate_m55 * 10 ** (-b * (m - 5.5))`; Poisson probability `1 - exp(-rate * T)`.
  `level` by `prob_10y["6.0"]`: `< 0.05` "low", `< 0.25` "moderate", `< 0.60` "high", else
  "very high"; `elevated = level in ("high", "very high")`. No events in radius gives rate 0 and
  level "low" (never raise for open ocean).
- Validation: radius clamped to 25..2000 km.

### A3. `ndp.ml.flood` (time-series forecast, replaces the cumulative-rainfall regression)
- `MODEL = "flood_level"`, `FLOOD_THRESHOLD_M = 1.44`, `HORIZONS = (1, 3, 6)` hours.
- `train()`: merge the hourly rainfall and river-level CSVs on timestamp. Features at time t:
  `level_now`, `level_1h_ago`, `level_3h_ago`, `rain_3h`, `rain_6h`, `rain_24h` (rolling sums of
  `Current rainfall (mm)`). Target: level at t + h for each horizon. One
  `HistGradientBoostingRegressor` per horizon. Chronological split (last 20% held out).
  Persistence baseline = predict `level_now`. Metadata per horizon `{h: {rmse, r2,
  persistence_rmse, skill_vs_persistence, flood_precision, flood_recall}}` plus top-level
  `baseline_rmse`, `skill` for the 3-hour horizon. Plots under `generated/flood/`:
  `f1_level_history.png`, `f2_forecast_vs_actual.png` (3 h horizon, one month of the test period,
  actual vs model vs persistence), `f3_skill_by_horizon.png` (RMSE bars, model vs persistence).
- `@dataclass(frozen=True) HorizonForecast`: `hours, level_m, is_flood, threshold_m,
  persistence_level_m`. `@dataclass(frozen=True) FloodForecast`: `horizons: list[HorizonForecast],
  any_flood: bool, peak_level_m: float`.
- `predict(level_now_m, level_1h_ago_m, level_3h_ago_m, rain_3h_mm, rain_6h_mm, rain_24h_mm)
  -> FloodForecast`. Inputs are validated by the caller (levels 0..15 m, rain >= 0).
- The old `predict(cumulative_rainfall_mm)` is removed. `compare` keeps its own flood spec.

### A4. Other models (same public API as before, better internals, honest metadata)
- `hurricane`: `HistGradientBoostingRegressor` on `lat, lon, day_of_year, abs(lat)` (+ year is NOT
  used); chronological split by year (train before 2005, test 2005 and later); `predict(on, lat,
  lon) -> HurricanePrediction` unchanged.
- `tsunami`: `RandomForestRegressor` with 5-fold CV metadata; `predict(lat, lon, max_height_m)`
  unchanged.
- `storm`: keep the decision tree, add a `RandomForestClassifier` comparison; keep whichever has
  higher cross-validated accuracy; metadata `accuracy`, `baseline_accuracy`, `cv_accuracy_mean`,
  `classes`; `predict(...)` unchanged.
- `earthquake` (the old magnitude regression) stays only because `compare` and the tests use it;
  its metadata states `skill` (about 0) and the module docstring says "illustrative, not used by
  the earthquake page".

### A5. `ndp.ml.compare`
- Adds a fifth entry `"Baseline (predict the mean)"` (always last in `MODEL_NAMES`) so users see
  whether any model beats guessing. `predict` returns five `ModelResult`s; `metrics()` includes it
  (accuracy_pct, r2, rmse). Hazard keys unchanged.

### A6. `ndp.ml.train`
- Trainers (in order): flood, earthquake (old regression + stats), hazard, tsunami, hurricane,
  storm, compare, sentiment, covid. `--quick` unchanged.

---

## Part B. Phase 2: routers, templates, API for the new models

Endpoint names stay; form fields change as below.

| Endpoint | Fields | Result context |
|---|---|---|
| `earthquake_predict` POST /earthquake/predict | `lat`, `lon`, `radius_km` (default 300) | `result: HazardEstimate` |
| `flood_predict` POST /flood/predict | `level_now`, `level_1h_ago`, `level_3h_ago`, `rain_3h`, `rain_6h`, `rain_24h` | `result: FloodForecast` |

- Earthquake page: hazard card (level badge, probability table for M >= 6 / 7 / 8 over 1 and 10
  years, events found in the radius, largest event with year, median depth, nearest large event),
  plain-language "what this means / does not mean" (a historical rate, not a prediction of when),
  the live USGS map and live-events panel stay. The "Find nearby hospitals and police" button posts
  `lat`, `lon`, `radius_km` (no magnitude) to `alerts_nearby`. The model-based automatic alert is
  removed from the earthquake router (the real-event watcher replaces it). Recent table columns:
  lat, lon, radius, level, P(M >= 6, 10 y).
- Flood page: six-input form with sensible defaults and units, a forecast table for 1, 3 and 6 h
  (model vs persistence, flood badge when above 1.44 m), the three plots.
- Model card (`partials/_model_card.html`): show `skill`, `baseline_rmse`, `split`,
  `cv_r2_mean/std` with labels, and a verdict line: skill < 0.05 "no better than predicting the
  average", < 0.25 "weak", < 0.5 "moderate", else "useful".
- Compare result shows the baseline row and a one-line verdict ("No model beats the baseline" when
  the best model's rmse is within 2% of the baseline's).
- JSON API (`routers/api.py`): `POST /api/v1/earthquake/hazard` `{lat, lon, radius_km?}` and
  `POST /api/v1/flood/forecast` `{level_now_m, level_1h_ago_m, level_3h_ago_m, rain_3h_mm,
  rain_6h_mm, rain_24h_mm}`; the old earthquake and flood bodies are removed.
- `data-speak` sentences updated for both partials; `data-speak-priority="alert"` on the hazard
  card only when `result.elevated`.

---

## Part C. Phase 3: live data, watcher (`routers/live.py`, `services/gdacs.py`,
`services/openmeteo.py`, `services/watcher.py`, templates `live/*`)

- `gdacs.recent_alerts(limit=30) -> list[GdacsEvent]`: GDACS (UN/EC Global Disaster Alert and
  Coordination System) events of type EQ, TC, FL, VO, DR, WF, Orange and Red levels first.
  `GdacsEvent(id, kind ("EQ"|"TC"|"FL"|"VO"|"DR"|"WF"), kind_name, name, alert_level
  ("Green"|"Orange"|"Red"), country, lat, lon, from_date, to_date, url, summary)`.
  Cache 10 minutes. Raises `ServiceError` on failure.
- `openmeteo.forecast(lat, lon, days=3) -> Forecast`: Open-Meteo (no key) hourly precipitation,
  wind gusts, temperature; derived `precip_24h_max_mm`, `precip_72h_total_mm`, `max_gust_kmh`,
  `advisories: list[Advisory(level "info"|"watch"|"warning", text)]` using the India
  Meteorological Department rainfall classes (heavy 64.5-115.5 mm/24 h, very heavy 115.6-204.4,
  extremely heavy >= 204.5) and gust >= 63 km/h; clearly labelled as a heuristic from model
  forecast data, not an official warning. Cache 15 minutes.
- `watcher`: `async start(app)` / `async stop(app)` (already called from the lifespan);
  does nothing unless `settings.watch_quakes`. Polls `usgs.recent_events` every
  `watch_interval_seconds`; an event with magnitude >= `watch_min_magnitude` and not seen before
  triggers: reverse geocode, nearby services (failure tolerated), `alerting.build_earthquake_message(
  ..., reported=True)`, `alerting.dispatch` (honours dry run). Dedupe by event id persisted in
  `PROJECT_DIR / "state" / "watcher.json"`; the first poll after a fresh start marks existing events
  as seen without alerting. `status() -> WatcherStatus(enabled, running, dry_run, min_magnitude,
  interval_s, last_poll, last_error, handled: list[HandledEvent(id, magnitude, place, at,
  outcome)])` (last 20, newest first).
- Router endpoints: `live_page` GET /live, `live_gdacs` GET /live/gdacs (HTMX partial),
  `live_forecast` POST /live/forecast (field `place`; HTMX partial), `live_watcher` GET
  /live/watcher (HTMX partial). Templates `live/page.html`, `live/_gdacs.html`,
  `live/_forecast.html`, `live/_watcher.html`; Leaflet map of GDACS events using the existing
  `data-ndp-map` / `data-markers-src` helpers in `static/app.js`; `data-speak` on results.

## Part D. Phase 3: assistant, guidance, evals (`services/assistant.py`, `guidance.py`,
`routers/assistant.py`, `templates/assistant/*`, `evals/`, `ndp/evals.py`)

- `ndp.guidance`: corpus of public-domain preparedness guidance in `data/guidance/*.md` with
  front matter `title`, `source_url`, `hazard`, `retrieved`. Fetched by `ndp-fetch-guidance`
  (`ndp.guidance:main`) from US federal sources (Ready.gov / FEMA, USGS, NOAA / National Weather
  Service, CDC); only sources whose terms allow reuse; every chunk keeps its source URL.
  `search(query: str, hazard: str | None = None, k: int = 4) -> list[Chunk(text, title,
  source_url, hazard, score)]` using TF-IDF (scikit-learn) over ~800-character chunks, built lazily
  and cached.
- `ndp.services.assistant`: `ask(session_id: str, message: str) -> AssistantTurn(answer: str,
  tools_used: list[ToolCall(name, summary)], sources: list[Source(title, url)], refused: bool)`.
  Built on `ndp.services.llm.create_message` with a manual tool loop (at most
  `settings.assistant_max_tool_rounds` rounds; always append `response.content` unchanged so
  thinking blocks survive; never `tool_choice` forced; no sampling parameters; `max_tokens` 4096).
  Prompt caching on the stable system prompt and tool list (`cache_control`).
  Tools (all read-only): `recent_earthquakes`, `earthquake_hazard`, `geocode`, `nearby_services`,
  `weather_forecast`, `gdacs_alerts`, `flood_forecast` (Hoppers Crossing only, says so),
  `preparedness_guidance`, `draft_alert` (composes text only). **No tool sends anything.**
  System prompt rules: use tools for facts; never claim to predict earthquakes; say that outputs
  are statistical or model data, not official warnings; point to local emergency numbers for
  emergencies; cite guidance sources by title and URL; short answers. Tool failures are returned
  to the model as `is_error` results, not raised.
- Sessions are in memory keyed by an unguessable cookie, capped (20 messages per session, 200
  sessions LRU). Per-client rate limit (10 messages per minute) so a public deployment cannot burn
  the key. Input capped at 1500 characters.
- Endpoints: `assistant_page` GET /assistant, `assistant_ask` POST /assistant/ask (field
  `message`; partial `assistant/_turn.html` with the answer, tool chips and sources),
  `assistant_reset` POST /assistant/reset. Not-configured state when `llm_configured` is false
  (page explains how to enable it, form disabled).
- Evals: `evals/assistant_cases.jsonl` (at least 24 cases: question, `expect_tools_any`,
  `must_mention`, `must_not` regexes such as promises to predict a quake, notes) and
  `ndp.evals` with `main(argv)` (`ndp-eval assistant`) that runs the cases against the real API
  when a key is set (reports tool-use rate, must/must-not pass rates, mean tool rounds, token
  usage and an estimated cost) and prints a clear "skipped: no key" otherwise. A pytest suite uses
  a scripted fake client to cover the tool loop, an erroring tool, max rounds, the refusal stop
  reason, rate limiting, the absence of any sending tool, and the not-configured state.

## Part E. Phase 3: triage and translation (`services/triage.py`, `services/translate.py`,
`routers/sentiment.py`, `routers/alerts.py`, templates `sentiment/*`, `alerts/*`)

- `triage.triage_post(text: str) -> Triage` via `llm.parse` with a Pydantic schema:
  `needs: list[Literal["food","water","shelter","medical","rescue","missing_person","donation",
  "information","other"]]`, `urgency: Literal["low","medium","high","critical"]`, `summary: str`
  (<= 200 chars), `location_mentioned: str | None`, `language: str`, `confidence: Literal["low",
  "medium","high"]`. Sentiment page gets a "Triage a post" card (textarea, max 1500 characters) and
  a "Triage 5 sample posts" button that samples from a chosen dataset's CSV; endpoints
  `sentiment_triage` POST /sentiment/triage and `sentiment_triage_sample` POST
  /sentiment/triage/sample. Not-configured state. `evals/triage_cases.jsonl` (at least 16 handwritten
  posts with expected needs and urgency band) and `ndp.evals_triage.run(...)` which the
  coordinator wires into `ndp-eval triage`.
- `translate.translate_alert(message: str, languages: list[str]) -> dict[str, str]` via the shared
  client: preserve numbers, phone numbers, coordinates and proper names; output only the
  translation; every translation gets a one-line "machine-translated, verify before relying"
  note in that language or English. Languages limited to the picker's list (Hindi, Tamil,
  Malayalam, Telugu, Bengali, Nepali, Spanish, French, Arabic, Japanese, Chinese, Indonesian).
- Alerts UI: in the send form, a checkbox list of languages and a "Preview translations" button
  (`alerts_translate` POST /alerts/translate; returns editable `translation_<code>` textareas inside
  the send form so a human reviews them); `alerts_send` appends any `translation_*` fields to the
  message body. Hidden with a short note when the LLM is not configured.
