# Phase 1 implementation contract

Shared by the ML modules (`src/ndp/ml/`), the routers (`src/ndp/routers/`) and the
templates (`src/ndp/templates/`). Names here are binding; change them only by editing
this file first.

Principles
- Nothing trains inside an HTTP handler. `uv run ndp-train` trains once and saves artifacts
  through `ndp.ml.registry.save(name, artifact, metadata)`; handlers call `registry.load(name)`.
- Raw CSVs are read from `get_settings().data_dir` via `ndp.ml.features.load_csv(filename)`.
- Train-time plots go through `ndp.ml.plots.save_current(sub, filename)` and are served as
  `/static/generated/<sub>/<filename>`; templates reference them with
  `url_for('static', path='generated/<sub>/<filename>')`.
- Per-request results are rendered as HTML (numbers, Tailwind bars); no files are written per request.
- No secrets anywhere in code or templates. Keys come from `ndp.config.Settings`.
- The user's input must actually reach every model (the legacy code often ignored it).

## 1. ML modules

All predict helpers return frozen dataclasses (defined in each module) and raise
`registry.ModelNotTrained` when the artifact is missing.

### `ndp.ml.features` (shared helpers)
- `load_csv(filename: str, **read_csv_kwargs) -> pd.DataFrame` from `settings.data_dir`;
  raises `DataMissing(filename)` (subclass of `FileNotFoundError`) with a hint to run `ndp-fetch-data`.
- `earthquake_date_to_epoch(s: str) -> float` accepts `%m/%d/%Y` and `%Y-%m-%dT%H:%M:%S.%fZ`.
- `parse_hurdat_coord(s: str) -> float` ("28.0N" -> 28.0, "94.8W" -> -94.8).
- `tolerance_accuracy(y_true, y_pred, tol_frac=0.1) -> float` = % of rows with |err| < tol_frac * max(|y_true|).
- `date_to_int(d: datetime.date) -> int` -> yyyymmdd.

### `ndp.ml.flood`
- `MODEL = "flood_level"`, `FLOOD_THRESHOLD_M = 1.44`
- `train() -> dict`: merge the two Hoppers Crossing CSVs (outer on `Date/Time`), fillna(0) on
  `Cumulative rainfall (mm)` and `Level (m)`; X=[Cumulative rainfall (mm)], y=Level (m);
  LinearRegression; split test_size=.2 random_state=0. Metadata: r2, rmse, n_rows, coef, intercept.
  Plots `generated/flood/`: `f1.png` rainfall over time, `f2.png` river level over time,
  `f3.png` level vs cumulative rainfall, `f4.png` train fit, `f5.png` test fit.
- `predict(cumulative_rainfall_mm: float) -> FloodPrediction(level_m: float, is_flood: bool, threshold_m: float)`

### `ndp.ml.earthquake`
- `MODEL = "earthquake_magnitude"`, `STATS = "earthquake_stats"`
- `train()`: `database.csv`; features `[Latitude, Longitude, Depth, date_epoch]` -> `Magnitude`;
  `Pipeline(StandardScaler, MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=500, random_state=0))`;
  split .2 rs 0; metadata r2, rmse, mae, n_rows. Also save `STATS` = dict
  `{min_mag, max_mag, mean_mag, n_events, busiest_year, busiest_year_count, by_year: {year: n},
  by_month: {1..12: n}, magnitude_bins: {"5-6": n, ..., "9-10": n}}` and plots `generated/earthquake/`:
  `e1.png` magnitude histogram, `e2.png` events by month, `e3.png` events by year, `e4.png` magnitude vs depth.
- `predict(lat: float, lon: float, depth_km: float, on: datetime.date) -> EarthquakePrediction(
  magnitude: float, severity: str, alert: bool)`; severity: `<4 "minor"`, `4-5 "light"`, `5-6 "moderate"`, `>=6 "strong"`; alert = magnitude >= 6.
- `stats() -> dict` loads `STATS`.

### `ndp.ml.tsunami`
- `MODEL = "tsunami_magnitude"`
- `train()`: `tsunami.csv`; features `[LATITUDE, LONGITUDE, MAXIMUM_HEIGHT]` (fillna mean) -> `PRIMARY_MAGNITUDE`
  (drop rows where target is NaN); `Pipeline(StandardScaler, MLPRegressor(hidden_layer_sizes=(100,), max_iter=1500, random_state=0))`;
  split .25 rs 0; metadata r2, rmse, n_rows. Plots `generated/tsunami/`: `t1.png` max height vs magnitude,
  `t2.png` events per decade, `t3.png` source locations scatter (lon vs lat).
- `predict(lat, lon, max_height_m) -> TsunamiPrediction(magnitude: float, risk: bool)`; risk = magnitude >= 6.5.

### `ndp.ml.hurricane`
- `MODEL = "hurricane_wind"`
- `train()`: concat atlantic + pacific HURDAT2; parse lat/lon with `parse_hurdat_coord`; keep rows with
  `Maximum Wind` >= 0 (HURDAT2 uses -99 for missing); features `[Date (int yyyymmdd), Latitude, Longitude]`
  -> `Maximum Wind` (knots); `Pipeline(StandardScaler, LinearRegression)`; split .2 rs 4; metadata r2, rmse, n_rows.
  Plots `generated/hurricane/`: `h1.png` predicted vs actual, `h2.png` max wind over time, `h3.png` wind distribution.
- `predict(on: datetime.date, lat, lon) -> HurricanePrediction(max_wind_kt: float, category: str, is_hurricane: bool)`;
  category by Saffir-Simpson on knots: `<34 "Tropical depression"`, `34-63 "Tropical storm"`, `64-82 "Category 1"`,
  `83-95 "Category 2"`, `96-112 "Category 3"`, `113-136 "Category 4"`, `>=137 "Category 5"`; is_hurricane = >= 64.

### `ndp.ml.storm`
- `MODEL = "storm_classifier"`
- `train()`: `weather_data.csv`; features `[Temperature, Pressure, Humidity, Wind Direction, Wind Speed]` ->
  `Weather Description`; `DecisionTreeClassifier(min_samples_split=20, random_state=99)`; split .2 rs 99 stratified;
  metadata accuracy, classes (list), n_rows. Plots `generated/storm/`: `tree.png` (plot_tree, max_depth=3),
  `classes.png` class distribution.
- `predict(temperature, pressure, humidity, wind_direction, wind_speed) -> StormPrediction(label: str, top: list[tuple[str, float]])`
  (top = up to 3 (class, probability) sorted desc).

### `ndp.ml.compare`
- `HAZARDS: dict[str, HazardSpec]` keyed `"earthquake" | "tsunami" | "flood" | "hurricane"`, where
  `HazardSpec(name, csv_loader, features: list[str], target: str, input_fields: list[InputField])` and
  `InputField(name, label, kind)` with kind `"float" | "date"`. Input fields (form names) per hazard:
  - earthquake: `lat`, `lon`, `depth` -> features `[Latitude, Longitude, Depth]` -> `Magnitude`
  - tsunami: `lat`, `lon`, `max_height` -> `[LATITUDE, LONGITUDE, MAXIMUM_HEIGHT]` -> `PRIMARY_MAGNITUDE`
  - flood: `current_rainfall`, `cumulative_rainfall` -> `[Current rainfall (mm), Cumulative rainfall (mm)]` -> `Level (m)`
  - hurricane: `date` (kind date -> yyyymmdd int), `lat`, `lon` -> `[Date, Latitude, Longitude]` -> `Maximum Wind`
- `MODEL_NAMES = ["Random forest", "Neural network", "SVM", "Linear regression"]` with
  RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0), MLPRegressor((100,), max_iter=1500, random_state=0),
  SVR(), LinearRegression(); every model inside `Pipeline(StandardScaler, model)`.
- `train(hazard, max_rows=15000)`: subsample with random_state=0 when larger, split .25 rs 0; metrics per model:
  `accuracy_pct` (tolerance_accuracy), `r2`, `rmse`. Save `f"compare_{hazard}"` artifact
  `{"models": {name: pipeline}, "features": [...], "target": str}`, metadata `{"metrics": {name: {...}}, "n_rows": int}`.
  Plot `generated/compare/{hazard}_accuracy.png` bar chart of accuracy_pct.
- `train_all()`; `predict(hazard, values: list[float]) -> list[ModelResult(name, prediction, accuracy_pct, r2, rmse)]`
  ordered as MODEL_NAMES; `metrics(hazard) -> dict | None`.

### `ndp.ml.sentiment`
- `DISASTERS: list[Disaster]` with `Disaster(slug, name, csv, monetary_impact_busd, image)` in this order:
  `japan-2016-earthquake` ("Japan 2016 earthquake", Japan2016_Earthquake.csv, 1.21, senimg/japan.jpg),
  `japan-2011-tsunami` ("Japan 2011 tsunami", japan_2011_tsunami.csv, 360, senimg/live.jpg),
  `chennai-2015-flood` ("Chennai 2015 flood", chennai_flood_2015.csv, 42.04, senimg/chennai.jpg),
  `kerala-2018-flood` ("Kerala 2018 flood", kerala_flood_2018.csv, 112.72, senimg/Kerala.jpeg),
  `nepal-2015-earthquake` ("Nepal 2015 earthquake", Nepal_Earthquake_2015.csv, 0.013, senimg/indonesia.jpg),
  `indonesia-2018-tsunami` ("Indonesia (Palu) 2018 tsunami", paul_Indonesia_tsunami_2018.csv, 0.0131, senimg/indonesia.jpg),
  `phailin-2013-cyclone` ("Cyclone Phailin 2013", phailin_cyclone_2013.csv, 0.655, senimg/phailin.jpg).
- `SCORES = "sentiment_scores"`; `ensure_corpora()` downloads stopwords, punkt, punkt_tab,
  averaged_perceptron_tagger, averaged_perceptron_tagger_eng, sentiwordnet, wordnet (quiet, idempotent).
- `score_text(text: str) -> float` ports the legacy SentiWordNet rule set (adverb/adjective/verb weights 1.0/1.2/1.5, 0.4 scaling).
- `score_dataset(disaster) -> dict(score, n_posts, n_sentences)` reproducing the legacy aggregation
  (sentence split on ".", per-post sum, mean per (date, user), then sum).
- `train()`: scores every disaster, saves `SCORES` = `{slug: {score, n_posts, n_sentences}}`,
  plot `generated/sentiment/impact_vs_sentiment.png` (scatter + fitted line). Can take several minutes.
- `scores() -> dict` loads `SCORES`.

### `ndp.ml.covid` (trained by `ndp-train` while NDP_ENABLE_COVID is true)
- `MODEL = "covid_india_cases"`, `EPOCH = date(2020, 1, 22)`
- `train()`: JHU CSV, rows where `Country/Region == "India"` summed; x = day index, y = cumulative confirmed;
  `Pipeline(StandardScaler, MLPRegressor((64, 64, 64), max_iter=3000, random_state=0))`; metadata last_date, n_days, r2.
  Plots `generated/covid/`: `actual.png`, `forecast.png` (next 90 days).
- `predict(on: datetime.date) -> CovidPrediction(day_index: int, cases: int, last_data_date: datetime.date)`
- `test_centres() -> list[dict(name, address, city, lat, lon)]` from `TestCentres_with_geolocation.csv` (finite coords only).
- `state_snapshot() -> list[dict(state, lat, lon, active, confirmed, deaths, recovered)]` from `state.csv`;
  `city_snapshot() -> list[dict(city, lat, lon, active, confirmed, deaths)]` from `main.csv`.

### `ndp.ml.train`
- `TRAINERS: dict[str, Callable[[], dict]]` in order: flood, earthquake, tsunami, hurricane, storm,
  compare (train_all), sentiment, covid (only when `--include-covid` or settings.enable_covid).
- `main(argv)`: positional names (subset) optional; flags `--include-covid`, `--skip-sentiment`, `--quick`
  (sets compare max_rows=3000 and skips sentiment); prints one summary line per model with its headline metric.

## 2. Routers and endpoint names (FastAPI `name=` used by `url_for` in templates)

Every POST handler accepts `application/x-www-form-urlencoded`; when the request carries
`HX-Request: true` it returns only the result partial, otherwise the full page with `result` set.
Invalid input returns the partial with `error` set (HTTP 200, so HTMX swaps it in).

| Router module | Endpoint name | Method / path | Form fields | Template(s) |
|---|---|---|---|---|
| pages | `index` | GET / | | `index.html` |
| earthquake | `earthquake_page` | GET /earthquake | | `earthquake/page.html` |
| earthquake | `earthquake_predict` | POST /earthquake/predict | lat, lon, depth, date (yyyy-mm-dd) | `earthquake/_result.html` |
| earthquake | `earthquake_graphs` | GET /earthquake/graphs | | `earthquake/graphs.html` |
| tsunami | `tsunami_page` | GET /tsunami | | `tsunami/page.html` |
| tsunami | `tsunami_predict` | POST /tsunami/predict | lat, lon, max_height | `tsunami/_result.html` |
| flood | `flood_page` | GET /flood | | `flood/page.html` |
| flood | `flood_predict` | POST /flood/predict | cumulative_rainfall | `flood/_result.html` |
| hurricane | `hurricane_page` | GET /hurricane | | `hurricane/page.html` |
| hurricane | `hurricane_predict` | POST /hurricane/predict | date, lat, lon | `hurricane/_result.html` |
| storm | `storm_page` | GET /storm | | `storm/page.html` |
| storm | `storm_predict` | POST /storm/predict | temperature, pressure, humidity, wind_direction, wind_speed | `storm/_result.html` |
| weather | `weather_page` | GET /weather | | `weather/page.html` |
| weather | `weather_predict` | POST /weather/predict | city, lat, lon | `weather/_result.html` |
| alerts | `alerts_page` | GET /alerts | | `alerts/page.html` |
| alerts | `alerts_search` | POST /alerts/search | place, radius_km | `alerts/_results.html` |
| alerts | `alerts_send` | POST /alerts/send | place, message, phone (optional) | `alerts/_sent.html` |
| sentiment | `sentiment_page` | GET /sentiment | | `sentiment/page.html` |
| sentiment | `sentiment_score` | GET /sentiment/{slug} | | `sentiment/_score.html` |
| sentiment | `sentiment_score_text` | POST /sentiment/score | text | `sentiment/_text_score.html` |
| pages | `compare_page` | GET /compare | | `compare/page.html` |
| pages | `compare_predict` | POST /compare/{hazard} | per `ndp.ml.compare.HAZARDS[hazard].input_fields` | `compare/_result.html` |
| covid (flag) | `covid_page` | GET /covid | | `covid/page.html` |
| covid | `covid_predict` | POST /covid/predict | date | `covid/_result.html` |
| covid | `covid_maps` | GET /covid/maps | | `covid/maps.html` |
| covid | `covid_geojson` | GET /covid/data/{layer} (`centres`,`states`,`cities`) | | JSON |
| (main) | `healthz` | GET /healthz | | JSON |
| api | `api_*` | POST /api/v1/{earthquake,tsunami,flood,hurricane,storm}/predict | JSON body | JSON |

## 3. Template context

Common: `request`, `settings` (always injected by `ndp.web.render`), `url_for`.
Every full page also receives `meta` = `registry.metadata(MODEL)` (dict or None) and `trained: bool`.
Every `_result.html` partial receives `inputs` (dict of the parsed form values) and either
`result` (the dataclass from §1) or `error` (str).

- `index.html`: `modules: list[dict(name, blurb, endpoint, status)]` with status in
  `"ready" | "needs training" | "not configured" | "archived"`.
- `earthquake/page.html`: `recent: list[dict(lat, lon, depth, date, magnitude, alert)]` (in-memory, newest first,
  max 8), `stats: dict | None` (from `earthquake.stats()`), `usgs_feed_url` (string).
- `earthquake/graphs.html`: `stats`, `plots: list[tuple[str, str]]` (title, static path).
- `tsunami/page.html`, `flood/page.html`, `hurricane/page.html`, `storm/page.html`: `plots` as above (may be empty).
- `storm/page.html` also `classes: list[str]`.
- `weather/page.html`: `configured: bool`; `_result.html`: `result = WeatherResult(prediction, actual, matched: bool,
  station_count: int, labels: list[str], city, lat, lon)`.
- `alerts/page.html`: `email_configured`, `sms_configured`; `_results.html`: `place: dict(name, lat, lon)`,
  `places: list[dict(name, kind, lat, lon, address, phone)]`, `radius_km`; `_sent.html`: `channels: list[str]`,
  `skipped: list[str]`, `place`, `message`.
- `sentiment/page.html`: `rows: list[dict(slug, name, image, monetary_impact_busd, score (float|None), n_posts)]`,
  `plot` (static path or None); `_score.html`: `row` (same dict) ; `_text_score.html`: `text`, `score`, `label`
  ("positive" if > 0, "negative" if < 0, else "neutral").
- `compare/page.html`: `hazards: list[dict(key, title, input_fields: list[InputField], metrics: dict | None, plot: str | None)]`;
  `_result.html`: `hazard` (dict as above), `results: list[ModelResult]`, `inputs`, `unit` (str: "magnitude", "m", "kt").
- `covid/page.html`: `centres_count`, `last_data_date`, `plots`; `_result.html`: `result: CovidPrediction`;
  `covid/maps.html`: nothing beyond common (fetches `/covid/data/{layer}` with JS).
- `errors/not_trained.html`: `model_name`.

## 4. Static assets
- Keep `static/senimg/*` (photos used by the sentiment page) and add `static/app.js` (Leaflet helpers:
  `ndpMap(id, lat, lon, zoom)`, `ndpUsgsLayer(map, url)`, `ndpMarkers(map, items)`).
- Remove every other legacy static file (old plots, Google Maps / speech scripts, Snap.svg, cv.js, me.jpg).
