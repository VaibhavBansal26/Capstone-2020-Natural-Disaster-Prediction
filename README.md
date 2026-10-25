# Natural Disaster Prediction

Estimates, live data and alerting for earthquakes, tsunamis, floods, hurricanes and storms.
Started as a VIT University capstone in 2020 (Flask 1, TensorFlow 1) and rebuilt in 2026 on
FastAPI, scikit-learn, HTMX, Tailwind and Leaflet, with live feeds and a Claude-powered assistant.

> Outputs are statistical estimates and other organisations' live data. They are not operational
> forecasts or warnings. Earthquakes cannot be predicted: the earthquake page reports a historical
> hazard rate, and real-time alerts come from the USGS feed, not from a model.

**Looking for the 2020 version?** It is preserved unchanged at the tag `v1-2020` and the branch
`legacy-2020`. See [docs/legacy-2020.md](docs/legacy-2020.md) for what changed and where each old
file went. The [2020 demo video](https://www.youtube.com/embed/Gad5vz1Pkcc) shows that version.

## What it does

| Page | |
|---|---|
| **Home** | A live 3D globe of this week's USGS earthquakes, with module cards |
| **Earthquake** | Historical hazard for any place (rate and probability of M 6, 7, 8), the live USGS map, the week's strongest real earthquakes with one-click "Nearby help" |
| **Flood** | River level 1, 3 and 6 hours ahead for one gauge, shown next to "no change" |
| **Tsunami, Hurricane, Storm** | Small models with their held-out skill shown on the page |
| **Live** | GDACS global alerts, a weather outlook for any place, and the real-event alert watcher |
| **Assistant** | Plain-language questions answered with live tools and cited guidance (Claude key needed) |
| **Alerts** | Hospitals and police near a place or epicentre, then alert your contacts by email or SMS, optionally in other languages |
| **Sentiment** | Forum-post sentiment for seven past disasters, plus AI triage of a post |
| **Compare** | Four regressors per hazard next to a "predict the mean" baseline |
| **COVID-19** | Live totals, the 2020 India case curve, nearest test centres |

Voice works on every page: a Voice button reads results aloud and a Speak button takes spoken
form values and commands ("latitude 28 point 6 longitude 77 point 2 radius 300, estimate",
"open flood", "send alert", then "confirm"). It uses the browser's built-in speech support.
Alerts always need a confirmation, and nothing in the app contacts hospitals or police directly.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 for you.

```bash
uv sync --all-extras        # dependencies (+ optional Twilio for SMS)
uv run ndp-fetch-data       # 43 MB of CSVs into data/raw/ (pinned to a known commit)
uv run ndp-train            # trains every model once (about two minutes)
uv run ndp-serve            # http://localhost:8000
```

`ndp-train --quick` skips the slow sentiment pass. Or use Docker, which fetches data and trains on
first start:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Everything is optional configuration. Copy `.env.example` to `.env`; features whose keys are
missing show a "not configured" notice. The Claude features need `ANTHROPIC_API_KEY`.

## How good are the models?

Each page shows a model card with held-out error beside a baseline. Honest summary:

| Model | Held-out result | Baseline | Skill |
|---|---|---|---|
| Flood, 3 h ahead | RMSE 0.023 m | "stays the same", 0.026 m | +0.11 (weak) |
| Tsunami magnitude | RMSE 0.65 | the mean, 0.77 | +0.15 (weak) |
| Hurricane wind | RMSE 24.5 kt | the mean, 25.9 kt | +0.05 (barely) |
| Storm class | accuracy 74% | most common class, 25% | +0.65 (small sample) |

The flood model mostly misses sudden rises, and the 2020 "flood level" of 1.44 m is not
informative on this gauge, so the page shows forecast levels and never a flood verdict. Details,
splits and caveats are in [docs/models.md](docs/models.md).

## Not verified

This repository was developed without an Anthropic API key. The assistant, post triage and alert
translation were tested only against scripted fakes. Run `uv run ndp-eval assistant` and
`uv run ndp-eval triage` with your key before relying on them. Real email and SMS sending, and a
live strong earthquake arriving while the watcher runs, were also not exercised.

## Documentation

- [Architecture](docs/architecture.md): layout, request flow, conventions, endpoints, tests
- [Models](docs/models.md): what each model does, how it was evaluated, its limits
- [Data](docs/data.md): datasets, the committed guidance corpus, live sources
- [Deployment](docs/deployment.md): Docker, Render, configuration, privacy and security
- [The 2020 version](docs/legacy-2020.md): what it was and where everything went

## Development

```bash
uv run ruff check .
uv run pytest -q                       # ml, services, web and the Node voice tests
NDP_DEBUG=true uv run ndp-serve        # auto-reload
./frontend/build.sh                    # recompile Tailwind after adding utility classes
```

## License

MIT, see [LICENSE](LICENSE).
