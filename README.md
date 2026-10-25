# Natural Disaster Prediction

Risk estimation and alerting for earthquakes, tsunamis, floods, hurricanes and storms,
plus sentiment analysis of public posts about past disasters. Started as a VIT University
capstone project in 2020 (Flask 1, TensorFlow 1, Bootstrap 3); modernised in 2026 to
FastAPI, scikit-learn, Tailwind, HTMX and Leaflet.

> Outputs are statistical estimates from historical datasets. They are not operational
> forecasts or warnings. In particular, earthquake magnitude cannot be forecast from
> location and date; that page is kept as a historical-hazard illustration.

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
| `NDP_AUTO_ALERT` | Send the earthquake alert automatically at M >= 6 (default off) |
| `NDP_ALERTS_DRY_RUN` | Compose and check alerts but send nothing |
| `NDP_DEBUG` | Auto-reload and verbose errors |

## What is in the box

| Page | Model (trained once by `ndp-train`) | Data |
|---|---|---|
| Earthquake | MLP regressor, magnitude from lat/lon/depth/date | USGS 1965-2016 |
| Tsunami | MLP regressor, source magnitude from location and max wave height | NOAA NCEI |
| Flood | Linear regression, river level from cumulative rainfall (flood at 1.44 m) | Melbourne Water |
| Hurricane | Linear regression, max wind from date and position | NOAA HURDAT2 |
| Storm | Decision tree, weather description from observations | hourly weather sample |
| Weather | Live KNN against nearby stations | OpenWeatherMap |
| Alerts | Hospitals and police near a place or epicentre, then alert your contacts by email/SMS | OpenStreetMap (Nominatim, Overpass), USGS |
| Sentiment | SentiWordNet rule set over forum posts for seven disasters | scraped posts (2011-2018) |
| Compare | Random forest, MLP, SVM and linear regression per hazard | as above |
| COVID-19 | Live world and India totals, 2020 case-count model, nearest test centres, state and city maps | disease.sh, JHU, ICMR snapshots |

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
4. **Automatic mode** (`NDP_AUTO_ALERT=true`) sends the alert as soon as the model result is
   M >= 6. Be aware that the regression model is a weak illustration (R-squared near zero): across
   the globe it returns roughly 5.8 to 5.95, so in practice it almost never reaches 6. Real events
   come from the USGS panel.

A JSON API mirrors the forms at `/api/v1/<hazard>/predict`; interactive docs at `/docs`.

## Layout

```
src/ndp/
  main.py          FastAPI app factory, ndp-serve
  config.py        pydantic-settings (NDP_* env vars)
  data.py          dataset manifest + ndp-fetch-data
  forms.py, web.py form parsing, Jinja helpers (HTMX partial vs full page)
  routers/         one module per page + api.py
  services/        OpenWeatherMap, OpenStreetMap, email/SMS clients
  ml/              one module per model, registry.py, plots.py, train.py (ndp-train)
  templates/       Jinja2 + Tailwind + HTMX; <page>/page.html and <page>/_result.html
  static/          app.js (Leaflet helpers), senimg/, generated/ (train-time plots)
data/raw/          fetched CSVs (git-ignored)
models/            trained artifacts (git-ignored)
docs/CONTRACT.md   the interface contract the port was built against
```

## Development

```bash
uv run ruff check .
uv run pytest -q
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

## License

MIT, see `LICENSE`.
