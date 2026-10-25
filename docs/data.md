# Data

## Raw datasets (not in git)

`uv run ndp-fetch-data` downloads the CSVs into `data/raw/`. They come from the last
pre-modernisation commit of this repository (`26cdd7d`), so the bytes are pinned. The manifest with
original sources is in `src/ndp/data.py`.

| File | Used by | Original source |
|---|---|---|
| database.csv | earthquake hazard, catalogue, compare | USGS significant earthquakes 1965-2016 |
| tsunami.csv | tsunami, compare | NOAA NCEI global historical tsunami database |
| hurricane-atlantic.csv, hurricane-pacific.csv | hurricane, compare | NOAA HURDAT2 |
| Hoppers Crossing-Hourly-*.csv | flood, compare | Melbourne Water |
| data.csv, flood1.csv | legacy flood series | rainfall, dam and forest-cover series |
| weather_data.csv | storm classifier | hourly weather with description |
| *_2011.csv, *_2013.csv, *_2015.csv, *_2016.csv, *_2018.csv | sentiment, triage samples | scraped forum posts about seven disasters |
| time_series_covid19_confirmed_global.csv | COVID-19 curve | JHU CSSE |
| TestCentres_with_geolocation.csv, first.csv, firsti.csv, main.csv, state.csv | COVID-19 maps | 2020 snapshots |

## Committed data

- `data/guidance/`: 25 preparedness pages from the US National Weather Service, CDC and USGS,
  stored as Markdown with front matter (`title`, `source_url`, `hazard`, `retrieved`,
  `license_note`). They are US federal works; only article text is kept. Refresh with
  `uv run ndp-fetch-guidance`. Ready.gov blocks scripted access and Indian NDMA pages have no
  licence we could verify, so neither is included.
- `data/evals/`: handwritten cases for `ndp-eval assistant` (36) and `ndp-eval triage` (22).

## Live sources (called at request time)

USGS earthquake feeds, GDACS, Open-Meteo, OpenStreetMap Nominatim and Overpass, OpenWeatherMap
(key needed) and disease.sh. Each degrades to a visible message when unreachable.
