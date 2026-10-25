"""``ndp-train``: train every model once and save the artifacts and plots.

uv run ndp-train                     # everything except covid
uv run ndp-train --quick             # smaller compare subsample, no sentiment
uv run ndp-train flood storm         # a subset
uv run ndp-train --include-covid     # also the COVID-19 model (on by default)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections.abc import Callable
from functools import partial

from sklearn.exceptions import ConvergenceWarning

from ndp.config import get_settings
from ndp.ml import compare, covid, earthquake, flood, hurricane, sentiment, storm, tsunami
from ndp.ml.features import DataMissing

TRAINERS: dict[str, Callable[[], dict]] = {
    "flood": flood.train,
    "earthquake": earthquake.train,
    "tsunami": tsunami.train,
    "hurricane": hurricane.train,
    "storm": storm.train,
    "compare": compare.train_all,
    "sentiment": sentiment.train,
    "covid": covid.train,
}

QUICK_COMPARE_MAX_ROWS = 3000


def _headline(name: str, result: dict) -> str:
    if name == "storm":
        return f"accuracy={result['accuracy']:.3f}"
    if name == "compare":
        parts = []
        for hazard, meta in result.items():
            best = max(meta["metrics"].items(), key=lambda kv: kv[1]["accuracy_pct"])
            parts.append(f"{hazard}: best {best[0]} {best[1]['accuracy_pct']:.1f}%")
        return "; ".join(parts)
    if name == "sentiment":
        scores = ", ".join(f"{slug}={value:.2f}" for slug, value in result["scores"].items())
        return f"{result['n_disasters']} disasters scored ({scores})"
    if "r2" in result:
        extra = f", rmse={result['rmse']:.3f}" if "rmse" in result else ""
        return f"r2={result['r2']:.3f}{extra}"
    return ", ".join(f"{k}={v}" for k, v in result.items() if not isinstance(v, dict | list))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ndp-train", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "names",
        nargs="*",
        choices=[*TRAINERS, []],
        metavar="NAME",
        help=f"subset to train; one of {', '.join(TRAINERS)} (default: all)",
    )
    parser.add_argument(
        "--include-covid",
        action="store_true",
        help="also train the COVID-19 model (included anyway while NDP_ENABLE_COVID is on)",
    )
    parser.add_argument("--skip-sentiment", action="store_true", help="skip the slow NLTK scoring")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"compare max_rows={QUICK_COMPARE_MAX_ROWS} and skip sentiment",
    )
    return parser.parse_args(argv)


def select(args: argparse.Namespace) -> dict[str, Callable[[], dict]]:
    include_covid = args.include_covid or get_settings().enable_covid or "covid" in args.names
    skip_sentiment = args.skip_sentiment or args.quick
    chosen: dict[str, Callable[[], dict]] = {}
    for name, trainer in TRAINERS.items():
        if args.names and name not in args.names:
            continue
        if name == "covid" and not include_covid:
            continue
        if name == "sentiment" and skip_sentiment and "sentiment" not in args.names:
            continue
        if name == "compare" and args.quick:
            trainer = partial(compare.train_all, max_rows=QUICK_COMPARE_MAX_ROWS)
        chosen[name] = trainer
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    trainers = select(args)
    settings = get_settings()
    print(f"models -> {settings.models_dir}\nplots  -> {settings.generated_dir}")
    failures = 0
    total_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name, trainer in trainers.items():
            start = time.perf_counter()
            try:
                result = trainer()
            except DataMissing as exc:
                failures += 1
                print(f"[{name:<10}] SKIPPED  {exc}", file=sys.stderr)
                continue
            elapsed = time.perf_counter() - start
            print(f"[{name:<10}] {elapsed:7.1f}s  {_headline(name, result)}", flush=True)
    print(f"done in {time.perf_counter() - total_start:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
