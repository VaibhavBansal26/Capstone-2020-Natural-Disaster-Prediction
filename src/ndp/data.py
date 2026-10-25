"""Raw dataset manifest and fetcher.

The CSV files the models train on used to be committed at the repo root. They
now live outside the tree (data/raw/, git-ignored) and are downloaded from the
last pre-modernisation commit of this repository, which pins their content.

Run:  uv run ndp-fetch-data            # download everything that is missing
      uv run ndp-fetch-data --force    # re-download
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from ndp.config import get_settings

REPO = "VaibhavBansal26/Capstone-2020-Natural-Disaster-Prediction"
PINNED_COMMIT = "26cdd7d"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{PINNED_COMMIT}/"


@dataclass(frozen=True)
class Dataset:
    filename: str
    used_by: str
    source: str


DATASETS: tuple[Dataset, ...] = (
    Dataset("database.csv", "earthquake", "USGS significant earthquakes 1965-2016 (via Kaggle)"),
    Dataset("tsunami.csv", "tsunami", "NOAA NCEI global historical tsunami database"),
    Dataset("hurricane-atlantic.csv", "hurricane", "NOAA HURDAT2 Atlantic (via Kaggle)"),
    Dataset("hurricane-pacific.csv", "hurricane", "NOAA HURDAT2 NE/Central Pacific (via Kaggle)"),
    Dataset("Hoppers Crossing-Hourly-Rainfall.csv", "flood", "Melbourne Water, Hoppers Xing"),
    Dataset("Hoppers Crossing-Hourly-River-Level.csv", "flood", "Melbourne Water, Hoppers Xing"),
    Dataset("data.csv", "flood (LSTM variant)", "rainfall / dam capacity / forest cover series"),
    Dataset("flood1.csv", "flood (LSTM variant)", "derived from data.csv"),
    Dataset("weather_data.csv", "storm classifier", "hourly weather observations with description"),
    Dataset("Japan2016_Earthquake.csv", "sentiment", "scraped forum posts"),
    Dataset("japan_2011_tsunami.csv", "sentiment", "scraped forum posts"),
    Dataset("chennai_flood_2015.csv", "sentiment", "scraped forum posts"),
    Dataset("kerala_flood_2018.csv", "sentiment", "scraped forum posts"),
    Dataset("Nepal_Earthquake_2015.csv", "sentiment", "scraped forum posts"),
    Dataset("paul_Indonesia_tsunami_2018.csv", "sentiment", "scraped forum posts"),
    Dataset("phailin_cyclone_2013.csv", "sentiment", "scraped forum posts"),
    Dataset("time_series_covid19_confirmed_global.csv", "covid", "JHU CSSE"),
    Dataset("TestCentres_with_geolocation.csv", "covid", "ICMR test centres"),
    Dataset("first.csv", "covid", "snapshot generated in 2020"),
    Dataset("firsti.csv", "covid", "snapshot generated in 2020"),
    Dataset("main.csv", "covid", "snapshot generated in 2020"),
    Dataset("state.csv", "covid", "snapshot generated in 2020"),
)


def fetch(force: bool = False, quiet: bool = False) -> int:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for ds in DATASETS:
            target = settings.data_dir / ds.filename
            if target.exists() and not force:
                continue
            url = RAW_BASE + quote(ds.filename)
            if not quiet:
                print(f"fetching {ds.filename} ...", end=" ", flush=True)
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
                tmp.replace(target)
            downloaded += 1
            if not quiet:
                print(f"{target.stat().st_size / 1024:.0f} KB")
    if not quiet:
        print(f"{downloaded} file(s) downloaded into {settings.data_dir}")
    return downloaded


def missing() -> list[str]:
    data_dir = get_settings().data_dir
    return [ds.filename for ds in DATASETS if not (data_dir / ds.filename).exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the raw datasets into data/raw/")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        fetch(force=args.force, quiet=args.quiet)
    except httpx.HTTPError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
