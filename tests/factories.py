"""Builders for model results, shared by the web tests. They return the real dataclasses so
tests never need trained models."""

from __future__ import annotations

from ndp.ml import flood, hazard


def make_hazard(**overrides) -> hazard.HazardEstimate:
    values = dict(
        lat=35.7,
        lon=139.7,
        radius_km=300.0,
        n_events=212,
        years=52.0,
        annual_rate_m55=4.08,
        b_value=0.91,
        b_is_global=False,
        annual_rate={"6.0": 1.27, "7.0": 0.156, "8.0": 0.0192},
        prob_1y={"6.0": 0.719, "7.0": 0.144, "8.0": 0.019},
        prob_10y={"6.0": 1.0, "7.0": 0.79, "8.0": 0.175},
        max_magnitude=9.1,
        max_date="2011-03-11",
        depth_median_km=33.0,
        nearest_event={"distance_km": 18.4, "magnitude": 6.1, "year": 1987},
        level="very high",
        elevated=True,
    )
    values.update(overrides)
    return hazard.HazardEstimate(**values)


def quiet_hazard(**overrides) -> hazard.HazardEstimate:
    return make_hazard(
        lat=0.0,
        lon=-30.0,
        n_events=0,
        annual_rate_m55=0.0,
        b_value=1.05,
        b_is_global=True,
        annual_rate={"6.0": 0.0, "7.0": 0.0, "8.0": 0.0},
        prob_1y={"6.0": 0.0, "7.0": 0.0, "8.0": 0.0},
        prob_10y={"6.0": 0.0, "7.0": 0.0, "8.0": 0.0},
        max_magnitude=None,
        max_date=None,
        depth_median_km=None,
        nearest_event=None,
        level="low",
        elevated=False,
        **overrides,
    )


def make_forecast(levels=(1.43, 1.46, 1.52), now=1.42) -> flood.FloodForecast:
    horizons = [
        flood.HorizonForecast(
            hours=h,
            level_m=lv,
            is_flood=lv > flood.FLOOD_THRESHOLD_M,
            threshold_m=flood.FLOOD_THRESHOLD_M,
            persistence_level_m=now,
        )
        for h, lv in zip((1, 3, 6), levels, strict=True)
    ]
    return flood.FloodForecast(
        horizons=horizons, any_flood=any(h.is_flood for h in horizons), peak_level_m=max(levels)
    )
