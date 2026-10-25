# Natural Disaster Prediction

Risk estimation and alerting for earthquakes, tsunamis, floods, hurricanes and storms,
plus sentiment analysis of public posts about past disasters. Started as a VIT University
capstone project in 2020 (Flask 1, TensorFlow 1, Bootstrap 3); modernised in 2026 to
FastAPI, scikit-learn, Tailwind, HTMX and Leaflet.

> Outputs are statistical estimates from historical datasets, plus live data from other
> organisations. They are not operational forecasts or warnings. Earthquakes cannot be
> predicted: the earthquake page reports a historical hazard rate, and real-time alerts come
> from the USGS feed, not from a model.

[Original demo video (2020)](https://www.youtube.com/embed/Gad5vz1Pkcc)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
uv sync --all-extras        # dependencies (+ optional Twilio for SMS alerts)
uv run ndp-fetch-data       # 43 MB of CSVs into data/raw/ (pinned to a known commit)
uv run ndp-train            # trains every model once, writes models/ and plots
uv run ndp-serve            # http://localhost:8000
```

`ndp-train --quick` trains in a couple of minutes by subsampling and skipping the slow
sentiment pass; use it for a first look. The full run takes longer, mostly in the
SentiWordNet scoring of ~4,500 forum posts.

Or with Docker, which fetches data and trains on first start:

```bash
docker compose up --build
```

## Configuration

Copy `.env.example` to `.env`. Everything is optional; features whose keys are missing
show a "not configured" notice instead of failing.

| Variable | Used for |
|---|---|
| `NDP_OPENWEATHER_API_KEY` | Live weather page (OpenWeatherMap free tier) |
| `NDP_SMTP_*`, `NDP_ALERT_*_EMAIL` | Alert emails (use an app password) |
| `NDP_TWILIO_*`, `NDP_ALERT_TO_NUMBER` | Alert SMS |
| `NDP_ENABLE_COVID` | COVID-19 pages; on by default, set `false` to hide them |
| `NDP_WATCH_QUAKES` | Watch the USGS feed and alert automatically for new strong events (default off) |
| `ANTHROPIC_API_KEY` | Enables the assistant, post triage and alert translation |
| `NDP_ANTHROPIC_MODEL` | Model for those features (default `claude-opus-5-5`; `claude-sonnet-5-5` is cheaper) |
| `NDP_ALERTS_DRY_RUN` | Compose and check alerts but send nothing |
| `NDP_DEBUG` | Auto-reload and verbose errors |

## What is in the box

| Page | What it does | Model and data |
|---|---|---|
| Earthquake | Historical hazard for a place: rate and probability of M >= 6, 7, 8 within a radius, plus the live USGS map and the week's strongest real earthquakes | Gutenberg-Richter rates from the USGS catalogue 1965-2016 |
| Flood | River level 1, 3 and 6 hours ahead at one gauge (Hoppers Crossing, Melbourne) | Gradient boosting on hourly rainfall and level, compared with "no change" |
| Tsunami | Source magnitude from location and maximum wave height | Random forest on NOAA NCEI records |
| Hurricane | Maximum wind from date and position | Gradient boosting on HURDAT2, tested on 2005 onward |
| Storm | Weather class from temperature, pressure, humidity and wind | Random forest on a small hourly sample |
| Weather | Live nearest-neighbour check of a city against nearby stations | OpenWeatherMap (key needed) |
| Live | Global disaster alerts (GDACS), weather outlook for a place, status of the real-event watcher | GDACS, Open-Meteo |
| Assistant | Plain-language questions answered with the live tools and cited preparedness guidance | Claude (key needed) |
| Alerts | Hospitals and police near a place or epicentre, then alert your contacts by email or SMS, optionally in other languages | OpenStreetMap, USGS |
| Sentiment | Sentiment of forum posts about seven past disasters against monetary impact, plus AI triage of a post | SentiWordNet, Claude (key needed for triage) |
| Compare | Four regressors per hazard next to a "predict the mean" baseline | scikit-learn |
| COVID-19 | Live world and India totals, 2020 India case curve, nearest ICMR test centres, maps | disease.sh, JHU, ICMR snapshots |

### How good are the models?

Every model is trained once by `ndp-train` and its page shows a model card with its held-out
error next to a baseline, plus a plain verdict. The honest summary from the last full run:

| Model | Held-out result | Baseline | Skill |
|---|---|---|---|
| Flood, 3 h ahead | RMSE 0.023 m | "level stays the same" 0.026 m | +0.11 (weak) |
| Tsunami magnitude | RMSE 0.65 | the mean, 0.77 | +0.15 (weak) |
| Hurricane wind | RMSE 24.5 kt | the mean, 25.9 kt | +0.05 (barely) |
| Storm class | accuracy 74% | most common class, 25% | +0.65 (useful, small sample) |
| Earthquake magnitude regression | R-squared -0.01 | the mean | none (kept only in Compare) |

Things worth knowing:

- Earthquake magnitude is not predictable from location, depth and date. The 2020 model returned
  about 5.9 everywhere. The page now reports a historical hazard rate instead. Counts include
  aftershocks, so busy regions all read "high" or "very high", and rates from fewer than about ten
  events are noisy.
- The flood model beats "no change" at every horizon, but only modestly, and it mostly misses
  sudden rises (recall under 10% for rises of 10 cm or more within 3 hours). The original project's
  1.44 m flood level is not informative here: the gauge's base level drifted upward and sat above
  it for most of 2017 and 2018. The page therefore shows forecast levels and the change, never a
  "flood / no flood" verdict.
- Compare uses shuffled splits and is flattering; the chronological splits above are the honest
  numbers. When no model beats the baseline, the page says so.

### Live data

`/live` shows GDACS (UN and EU Global Disaster Alert and Coordination System) events from the
last 30 days on a map, and a weather outlook for any place from Open-Meteo with advisories based
on India Meteorological Department rainfall classes and a 63 km/h gust threshold. These are
heuristics on model data, clearly labelled, and not official warnings. The same page shows the
real-event watcher described under alerts below.

### Assistant

`/assistant` answers questions such as "what should I do during an earthquake?" or "weather
outlook for Chennai this week" using Claude and nine read-only tools: recent earthquakes,
earthquake hazard, geocoding, nearby hospitals and police, weather outlook, GDACS alerts, the
Hoppers Crossing flood forecast, preparedness guidance, and a drafter for alert text. **No tool can
send anything**; it can only draft an alert for you to review on the Alerts page. It is told never
to claim it can predict earthquakes and to send people to local emergency services.

- Guidance comes from a small committed corpus (`data/guidance/`, 25 pages from the US National
  Weather Service, CDC and USGS, public domain; each file records its source URL, retrieval date
  and licence note). Ready.gov refused scripted access and Indian NDMA pages have no licence we
  could verify, so neither is included. Refresh with `uv run ndp-fetch-guidance`.
- It is rate limited (10 messages per minute per client), chats are kept in server memory for a
  limited time, and message text is sent to Anthropic. It needs `ANTHROPIC_API_KEY`.
- `uv run ndp-eval assistant` runs 36 handwritten cases (tool use, caveats, prompt injection,
  requests to predict or send, several languages) against the real API and reports pass rates,
  tool rounds and estimated cost. `ndp-eval triage` does the same for post triage. Without a key
  they print "skipped". The graders are plain regular expressions and a small handwritten set shows
  agreement with the author's own expectations, not accuracy.
- **Not verified:** this repository was developed without an Anthropic key, so the Claude-powered
  features (assistant, triage, translation) were tested only against scripted fakes. Run
  `ndp-eval` with your key before relying on them.

### Post triage and translated alerts

On the Sentiment page a post can be triaged into needs (food, water, shelter, medical, rescue and
so on) and an urgency band, or five sample posts from a disaster dataset can be triaged at once.
In an alert, you can tick languages and preview translations; each appears in an editable box so a
person reviews it before sending, and is labelled machine-translated. Both need the API key.

### Voice

The Voice button in the header reads results aloud (off by default; your choice is remembered),
and the Speak button takes spoken input. Both use the browser's built-in speech support, so no
keys, installs or servers are involved, and they appear only where the browser supports them.
Every result also gets a small "Read aloud" button.

Examples of what you can say on a form page: "latitude 28 point 6 longitude 77 point 2 depth
ten", then "estimate". Also "open flood", "read result", "stop", "help", "find nearby help", and
"send alert" followed by "confirm". Out-of-range values are refused rather than filled. The
phrase list lives in `src/ndp/static/voice.js` and is covered by `tests/voice/`
(`node --test tests/voice/*.test.mjs`, also run by pytest when Node is installed).

### Landing page and language

The home page shows a 3D globe (globe.gl) with this week's USGS earthquakes: hover for details,
click to fly to one and jump to the estimate form for that location. Without WebGL it falls back
to a flat map. The language picker uses Google Translate, loaded only after you choose a
language other than English.

### Earthquake alerts

The 2020 app emailed and texted a hard-coded contact whenever its model predicted a magnitude
above 6. That chain is back, with more care:

1. **From the earthquake result**, the "Find nearby hospitals and police" button searches
   OpenStreetMap around the location and prepares an alert listing the nearest hospitals and
   police stations with their phone numbers where mapped.
2. **From the live panel**, the week's strongest real USGS earthquakes each have a "Nearby help"
   button that does the same with the reported magnitude, so the chain is usable with real events.
3. **Sending** always asks for confirmation (also by voice: say "send alert", then "confirm"). The
   message goes to the contacts configured in `.env`; it never contacts hospitals or police
   directly. `NDP_ALERTS_DRY_RUN=true` checks everything without sending.
4. **Automatic mode** is the real-event watcher (`NDP_WATCH_QUAKES=true`). It polls the USGS feed
   every few minutes, and for a new event at or above `NDP_WATCH_MIN_MAGNITUDE` (default 6.0) it
   looks up the nearest hospitals and police and alerts your configured contacts, once per event.
   It ignores events older than 24 hours and, after a fresh start, only records what already
   exists. Try it with `NDP_ALERTS_DRY_RUN=true` first. The 2020 version alerted from the model's
   own prediction; that trigger is gone because the model cannot tell where an earthquake will
   happen.

A JSON API mirrors the forms at `/api/v1/<hazard>/predict`; interactive docs at `/docs`.

## Layout

```
src/ndp/
  main.py          FastAPI app factory, ndp-serve
  config.py        pydantic-settings (NDP_* env vars)
  data.py          dataset manifest + ndp-fetch-data
  forms.py, web.py form parsing, Jinja helpers (HTMX partial vs full page)
  routers/         one module per page + api.py
  services/        OpenWeatherMap, OpenStreetMap, USGS, GDACS, Open-Meteo, Claude (llm.py), alerting
  ml/              one module per model, registry.py, plots.py, train.py (ndp-train)
  templates/       Jinja2 + Tailwind + HTMX; <page>/page.html and <page>/_result.html
  static/          tailwind.css (compiled), app.js, voice.js, i18n.js, globe.js, senimg/,
                   generated/ (train-time plots, git-ignored)
data/raw/          fetched CSVs (git-ignored)
data/guidance/     preparedness guidance corpus (committed, public domain sources)
evals/             handwritten eval cases for the assistant and post triage
models/            trained artifacts (git-ignored)
state/             watcher state (git-ignored)
docs/              interface contracts the port was built against
```

## Development

```bash
uv run ruff check .
uv run pytest -q
node --test tests/voice/*.test.mjs   # also run by pytest when Node is installed
./scripts/build_css.sh               # recompile Tailwind after adding utility classes
NDP_DEBUG=true uv run ndp-serve
```

## What changed from the 2020 version

- Models train once from the command line and are loaded from disk; the old app retrained
  inside every HTTP request (one route for 10,000 epochs).
- Several legacy routes never used the submitted values (one-hot encoding of the input
  string, or predicting for dataset row 0). Every model now receives the user's input.
- The earthquake pipeline no longer reshapes four feature columns into three.
- All credentials moved to environment variables. The keys that were committed in 2020
  must be considered compromised and rotated.
- Templates and static assets, previously only inside `capstone.rar`, are in the tree. Google Maps
  and jQuery were removed; maps use Leaflet with OpenStreetMap tiles and the live USGS feed. The
  landing-page globe, voice features and language picker were rebuilt on current browser APIs.
- MySQL dependency removed (it stored only the last few queries).
- Datasets moved out of the working tree and are fetched by script.

## Deploying

The Docker image fetches the datasets and trains the models on first start (about two minutes),
then serves on `$PORT` (default 8000). `render.yaml` is a Render blueprint with a persistent disk
so a restart does not retrain; it sets `FORWARDED_ALLOW_IPS=*` so the assistant's per-client rate
limit sees real client addresses behind Render's proxy. Roughly 1 GB of RAM is needed for the
trained comparison models. Alerts default to dry run in that blueprint; flip `NDP_ALERTS_DRY_RUN`
deliberately. Pages send security headers (`nosniff`, frame and referrer policy, a permissions
policy that allows the microphone for voice input only on this origin). Tailwind is compiled to a
28 KB stylesheet, so pages do not ship the in-browser compiler.

## License

MIT, see `LICENSE`.
