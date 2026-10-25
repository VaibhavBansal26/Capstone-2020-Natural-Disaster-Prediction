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
| `NDP_ENABLE_COVID` | Mounts the archived 2020 COVID module |
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
| Alerts | Hospitals and police near a place, send email/SMS | OpenStreetMap (Nominatim, Overpass) |
| Sentiment | SentiWordNet rule set over forum posts for seven disasters | scraped posts (2011-2018) |
| Compare | Random forest, MLP, SVM and linear regression per hazard | as above |
| COVID (flag) | 2020 case-count model, test-centre and state maps | JHU, ICMR snapshots |

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
- Templates and static assets, previously only inside `capstone.rar`, are in the tree.
  Google Maps, Google Translate, Web Speech scripts and jQuery were removed; maps use Leaflet
  with OpenStreetMap tiles and the live USGS feed.
- MySQL dependency removed (it stored only the last few queries).
- Datasets moved out of the working tree and are fetched by script.

## License

MIT, see `LICENSE`.
