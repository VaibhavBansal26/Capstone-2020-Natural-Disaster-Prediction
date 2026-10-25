# Deployment

## Docker

```bash
docker build -f deploy/Dockerfile -t natural-disaster-prediction .
docker run -p 8000:8000 --env-file .env natural-disaster-prediction
# or
docker compose -f deploy/docker-compose.yml up --build
```

On start the entrypoint downloads any missing datasets and runs `ndp-train --if-missing`, which trains only
what has no saved model or plots yet, then serves on `$PORT` (default 8000). A first start measured about
three minutes to become healthy (about 40 seconds of download, then about a minute of training, plus
startup); later starts take seconds. Mount volumes at `/app/data/raw` and `/app/models` (the
compose file does) so a restart does not download or retrain; point `NDP_GENERATED_DIR` and
`NDP_STATE_DIR` at mounted paths too if plots and the watcher's dedupe state should survive a rebuild. `/healthz` is the health check. With every model
loaded the app measured about 480 MB resident, so a 1 GB plan leaves headroom; the two random-forest
comparison models (about 150 MB on disk) are the bulk of it.

## Render

`render.yaml` is a blueprint for one Docker web service with a 2 GB persistent disk at
`/var/data`; `NDP_DATA_DIR`, `NDP_MODELS_DIR`, `NDP_GENERATED_DIR` and `NDP_STATE_DIR` point into it. Secrets are declared with
`sync: false` and set in the dashboard. It defaults `NDP_ALERTS_DRY_RUN=true`, keeps the watcher off
and uses the cheaper `claude-sonnet-5-5` for the public assistant. It sets
`FORWARDED_ALLOW_IPS=*` so the assistant's per-client rate limit sees real client addresses behind
Render's proxy; do not set that on a host without a trusted proxy.

## Configuration

Copy `.env.example` to `.env`. Everything is optional and features whose keys are missing show a
"not configured" notice.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` or `NDP_ANTHROPIC_API_KEY` | Assistant, post triage, alert translation |
| `NDP_ANTHROPIC_MODEL` | Model for those features (default `claude-opus-5-5`) |
| `NDP_OPENWEATHER_API_KEY` | Weather page |
| `NDP_SMTP_*`, `NDP_ALERT_FROM_EMAIL`, `NDP_ALERT_TO_EMAIL` | Alert email (use an app password) |
| `NDP_TWILIO_*`, `NDP_ALERT_TO_NUMBER` | Alert SMS (install the `alerts` extra) |
| `NDP_ALERTS_DRY_RUN` | Compose and check alerts but send nothing |
| `NDP_ALERT_RADIUS_KM` | Search radius for hospitals and police |
| `NDP_ALERTS_PASSCODE` | If set, sending an alert requires this passcode (recommended on a public site) |
| `NDP_ALERTS_ALLOW_CUSTOM_PHONE` | Let the alert form choose the SMS recipient (off by default) |
| `NDP_WATCH_QUAKES`, `NDP_WATCH_MIN_MAGNITUDE`, `NDP_WATCH_INTERVAL_SECONDS` | Real-event watcher |
| `NDP_ENABLE_COVID` | COVID-19 pages (on by default) |
| `NDP_DATA_DIR`, `NDP_MODELS_DIR`, `NDP_GENERATED_DIR`, `NDP_STATE_DIR`, `NDP_GUIDANCE_DIR` | Locations of data, trained models, training plots, watcher state and guidance |
| `NDP_ASSISTANT_MAX_TOOL_ROUNDS` | Cap on tool calls per assistant answer (default 6) |
| `NDP_DEBUG` | Auto-reload and verbose errors |

## Privacy and security

- When the Claude features are on, the text users type is sent to Anthropic. Assistant chats are
  held in server memory only (up to 200 sessions of 20 questions, least recently used evicted) and are
  lost on restart; nothing is written to disk.
- Pages send `X-Content-Type-Options`, `X-Frame-Options`, a referrer policy and a permissions
  policy that allows the microphone only for this origin.
- The language picker loads Google Translate only after a non-English language is chosen.
- Sending an alert is unauthenticated unless `NDP_ALERTS_PASSCODE` is set. It is rate limited (5 sends per
  10 minutes per client address), ignores the phone field unless explicitly enabled, and the Render
  blueprint defaults to dry run. The assistant, post triage and translation previews are rate limited
  too, because each call spends the owner's Claude budget.
- Never commit `.env`. The credentials that were committed in 2020 must be treated as compromised.
