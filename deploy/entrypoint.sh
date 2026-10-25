#!/usr/bin/env sh
# Fetch datasets and train whatever is missing, then serve.
# NDP_DATA_DIR / NDP_MODELS_DIR / NDP_GENERATED_DIR / NDP_STATE_DIR let a host keep all of it on
# one persistent disk. `ndp-train --if-missing` is a no-op once every model and plot exists, and
# trains only the new ones when a release adds a model.
set -eu
cd /app
uv run --no-sync ndp-fetch-data --quiet
uv run --no-sync ndp-train --if-missing
exec uv run --no-sync uvicorn ndp.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" "$@"
