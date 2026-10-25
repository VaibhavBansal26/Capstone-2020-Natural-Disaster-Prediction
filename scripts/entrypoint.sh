#!/usr/bin/env sh
# Fetch datasets and train models on first start, then serve.
# NDP_DATA_DIR / NDP_MODELS_DIR let a host mount one persistent disk for both.
set -eu
cd /app
MODELS_DIR="${NDP_MODELS_DIR:-models}"
uv run --no-sync ndp-fetch-data --quiet
if [ ! -f "$MODELS_DIR/flood_level.joblib" ]; then
  echo "no trained models found, training (one-off, a few minutes) ..."
  uv run --no-sync ndp-train
fi
exec uv run --no-sync uvicorn ndp.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" "$@"
