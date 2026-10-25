# The 2020 version

This project began as a VIT University capstone in 2020 (Flask 1, TensorFlow 1, MySQL, Bootstrap
3, Google Maps). The original code is preserved, unchanged, at:

- tag `v1-2020` (commit `26cdd7d`)
- branch `legacy-2020`

The [2020 demo video](https://www.youtube.com/embed/Gad5vz1Pkcc) shows that version.

## Where each 2020 file went

| 2020 file | Now |
|---|---|
| `prediction.py` (3,600-line Flask app, about 45 routes) | `src/ndp/routers/*`, `src/ndp/services/*`, `src/ndp/ml/*` |
| `app.py`, `apps.py`, `earthpd.py`, `model.py`, `models.py`, `weather_prediction.py` | earlier drafts of the same routes; superseded |
| `capstone.rar` (templates and static files, only present inside the archive) | `src/ndp/templates/`, `src/ndp/static/` (rewritten) |
| `*.csv` at the repository root | fetched into `data/raw/` by `ndp-fetch-data` |
| `model.pkl`, `model_columns.pkl`, `earthquake_model.ckpt.*`, `visual.dot` | replaced by `ndp-train` artifacts in `models/` |
| `requirements.txt`, `requirements1.txt` | `pyproject.toml` and `uv.lock` |
| `db.yaml` and the MySQL connection | removed; it stored only the last few queries |

## What changed in behaviour

- Models train once from the command line instead of inside every request.
- Several routes never used what the user typed (the input was one-hot encoded into nothing, or
  the model predicted for dataset row 0). Every model now receives the user's input.
- The earthquake "magnitude predictor" is replaced by a historical hazard estimate; real-time
  alerts come from the live USGS feed.
- The flood "level from rainfall" fit is replaced by an hourly forecast compared with persistence.
- Credentials moved to environment variables. The keys committed in 2020 must be treated as
  compromised and rotated.
- Google Maps, Google Translate, jQuery and Bootstrap 3 were replaced by Leaflet, HTMX and Tailwind.
  The globe, voice features, nearby-help alerts and COVID pages from 2020 were rebuilt.
