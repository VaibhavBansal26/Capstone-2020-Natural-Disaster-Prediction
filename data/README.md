# Data

Raw datasets are **not** stored in the working tree. Run

```bash
uv run ndp-fetch-data
```

to download them into `data/raw/` (git-ignored). The fetcher pulls each file
from the last pre-modernisation commit of this repository, so the bytes are
pinned and reproducible. The manifest with original sources lives in
`src/ndp/data.py`.

| File | Used by | Original source |
|---|---|---|
| database.csv | earthquake | USGS significant earthquakes 1965-2016 |
| tsunami.csv | tsunami | NOAA NCEI global historical tsunami database |
| hurricane-atlantic.csv, hurricane-pacific.csv | hurricane | NOAA HURDAT2 |
| Hoppers Crossing-Hourly-*.csv | flood | Melbourne Water |
| data.csv, flood1.csv | flood (sequence variant) | rainfall / dam / forest cover series |
| weather_data.csv | storm classifier | hourly weather with description |
| *_2015.csv, *_2016.csv, *_2018.csv, *_2011.csv, *_2013.csv | sentiment | scraped forum posts |
| time_series_covid19_confirmed_global.csv | covid  | JHU CSSE |
| TestCentres_with_geolocation.csv, first.csv, firsti.csv, main.csv, state.csv | covid  | 2020 snapshots |
