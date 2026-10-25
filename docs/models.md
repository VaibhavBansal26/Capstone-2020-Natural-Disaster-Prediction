# Models and how good they are

Every model is trained once by `uv run ndp-train`. Each page shows a model card built from the
JSON sidecar saved next to the artifact: held-out error, a baseline, a skill score
(`1 - rmse / baseline_rmse`) and how the data was split. Time-ordered data is always split
chronologically. The numbers below are from the last full run.

| Model | Question it answers | Data | Split | Held-out result | Baseline | Skill |
|---|---|---|---|---|---|---|
| Flood forecast | River level 1, 3, 6 h ahead at one gauge | Hoppers Crossing hourly rainfall and level | Chronological, last 20% (2017-05 to 2018-12) | 3 h RMSE 0.023 m, R-squared 0.88 | "stays the same", 0.026 m | +0.11 |
| Earthquake hazard | How often do M >= 6, 7, 8 happen within a radius | USGS M >= 5.5 catalogue, 1965 to 2016 | Back-test before and after 2000 | Active cells hold 99% of later M >= 6 events in 31% of the globe | none | n/a |
| Tsunami | Source magnitude from place and wave height | NOAA NCEI | Random 75/25 and 5-fold CV | RMSE 0.65, CV R-squared 0.23 | the mean, 0.77 | +0.15 |
| Hurricane | Maximum wind from date and position | HURDAT2 | Train 1851-2004, test 2005-2015 | RMSE 24.5 kt | the mean, 25.9 kt | +0.05 |
| Storm | Weather class from observations | 1,020 weather observations | Stratified random 80/20, CV for model choice | Accuracy 74% | most common class, 25% | +0.65 |
| Sentiment | SentiWordNet score per disaster | 4,175 forum posts | n/a | descriptive | n/a | n/a |
| COVID-19 case curve | Cumulative India cases (2020 data) | JHU CSSE | Chronological, last 20% of days | RMSE 28,551 | the mean, 55,846 | +0.49 |

## What to be careful about

- **Earthquakes cannot be predicted.** The 2020 model returned about 5.9 everywhere (held-out
  R-squared -0.01) and was removed. The hazard module reports a historical rate with
  Gutenberg-Richter extrapolation and Poisson probabilities. Counts include aftershocks (no
  declustering), so busy regions read "high" or "very high". Rates from fewer than about ten events
  are noisy. A local b-value is used only with at least 50 events; otherwise the global value.
- **The flood model is for one gauge and is only modestly better than "no change".** It beats
  persistence at 1, 3 and 6 hours, but recall for rises of 10 cm or more is 0% at 1 h, 7% at 3 h
  and 2% at 6 h on the held-out period, so it mostly misses sudden rises. The 1.44 m "flood
  level" from the 2020 project is not informative: the gauge's base level drifted upward and sat
  above it for most of 2017 and 2018, so pages show forecast levels and changes, never a flood
  verdict. Stuck-sensor stretches (about 10,000 hours at exactly 1.40 or 1.54 m) and logger zeros
  are masked before training.
- **Hurricane skill is close to nothing.** Position and time of year say little about wind.
- **Storm accuracy comes from a small sample** with no timestamps, so leakage between neighbouring
  hours cannot be ruled out.
- **Compare uses shuffled 75/25 splits** and is more flattering than the chronological numbers
  above. When no model beats the baseline within 2%, the page says so.
- **The COVID case curve** has only 117 days behind it and is an extrapolation.

## Claude-powered features

The assistant, post triage and alert translation use Claude through `services/llm.py`
(default model `claude-opus-5-5`, configurable). They were developed without an API key and tested
against scripted fakes. `uv run ndp-eval assistant` and `ndp-eval triage` run the handwritten cases
in `data/evals/` against the real API. Those cases show agreement with the author's expectations
and are not an accuracy benchmark.
