#!/usr/bin/env sh
# Fetch datasets and train models on first start, then serve.
set -eu
cd /app
uv run --no-sync ndp-fetch-data --quiet
if [ ! -f models/flood_level.joblib ]; then
  echo "no trained models found, training (one-off, a few minutes) ..."
  uv run --no-sync ndp-train
fi
exec uv run --no-sync uvicorn ndp.main:app --host 0.0.0.0 --port 8000 "$@"
