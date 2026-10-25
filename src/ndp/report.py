"""`ndp-report`: a statistics report built from the repository, the datasets and the trained models.

Nothing in the report is typed by hand. It reads

* the working tree (lines of code, files, test functions),
* the raw CSVs in ``data/raw`` (rows, columns, date spans, size),
* the JSON sidecars next to the trained artifacts in ``models/`` (metrics, baselines, splits),
* the recorded training run (seconds per trainer), and
* optionally times the model-backed endpoints in-process (``--bench``).

Anything that is not available (no trained models, no datasets) is reported as such instead of
failing, so the command works on a fresh checkout.

    uv run ndp-report                          # print Markdown
    uv run ndp-report --write docs/statistics.md
    uv run ndp-report --bench --write docs/statistics.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ndp import __version__
from ndp.config import PACKAGE_DIR, PROJECT_DIR, get_settings
from ndp.ml import compare, registry, sentiment

# --------------------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------------------


def table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], align_right: Sequence[int] = ()
) -> str:
    """A GitHub-flavoured Markdown table. Columns listed in `align_right` are right-aligned."""
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---:" if i in align_right else "---" for i in range(len(headers))) + "|"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if value != 0 and (abs(value) < 10 ** (-digits) or abs(value) >= 1e6):
        return f"{value:.3g}"
    return f"{value:,.{digits}f}"


def fmt_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / 1024 / 1024:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} B"


# --------------------------------------------------------------------------------------
# Repository statistics
# --------------------------------------------------------------------------------------

AREAS: list[tuple[str, Callable[[str], bool]]] = [
    (
        "Application: ml (training and prediction)",
        lambda p: p.startswith("src/ndp/ml/") and p.endswith(".py"),
    ),
    (
        "Application: services (external clients)",
        lambda p: p.startswith("src/ndp/services/") and p.endswith(".py"),
    ),
    ("Application: routers", lambda p: p.startswith("src/ndp/routers/") and p.endswith(".py")),
    ("Application: evals", lambda p: p.startswith("src/ndp/evals/") and p.endswith(".py")),
    (
        "Application: core modules",
        lambda p: re.fullmatch(r"src/ndp/[^/]+\.py", p) is not None,
    ),
    (
        "Templates (Jinja2 HTML)",
        lambda p: p.startswith("src/ndp/templates/") and p.endswith(".html"),
    ),
    ("Front-end JavaScript", lambda p: p.startswith("src/ndp/static/js/") and p.endswith(".js")),
    (
        "Front-end CSS (hand-written)",
        lambda p: (
            p.startswith("src/ndp/static/css/")
            and p.endswith(".css")
            and not p.endswith("tailwind.css")
        ),
    ),
    ("Tests: Python", lambda p: p.startswith("tests/") and p.endswith(".py")),
    ("Tests: JavaScript", lambda p: p.startswith("tests/") and p.endswith(".mjs")),
    (
        "Documentation (Markdown)",
        lambda p: p.endswith(".md") and not p.startswith("data/guidance/"),
    ),
    ("Preparedness guidance corpus", lambda p: p.startswith("data/guidance/")),
    ("Eval cases", lambda p: p.startswith("data/evals/")),
]


def tracked_files(root: Path) -> list[str]:
    """Files in the repository: `git ls-files` when available, else a directory walk."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=20, check=True
        ).stdout
        files = [line for line in out.splitlines() if line]
        if files:
            return [f for f in files if (root / f).is_file()]
    except (OSError, subprocess.SubprocessError):
        pass
    skip = {".git", ".venv", "node_modules", "__pycache__", "models", "raw", "generated"}
    return [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not skip.intersection(p.relative_to(root).parts)
    ]


def count_lines(path: Path) -> tuple[int, int]:
    """(non-blank lines, total lines) of a text file; binary files count as zero."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0, 0
    lines = text.splitlines()
    return sum(1 for ln in lines if ln.strip()), len(lines)


def repo_stats(root: Path = PROJECT_DIR) -> dict[str, Any]:
    files = tracked_files(root)
    areas = []
    for label, match in AREAS:
        picked = [f for f in files if match(f)]
        code = sum(count_lines(root / f)[0] for f in picked)
        areas.append({"area": label, "files": len(picked), "lines": code})
    py_tests = sum(
        len(re.findall(r"^\s*(?:async\s+)?def test_", (root / f).read_text(encoding="utf-8"), re.M))
        for f in files
        if f.startswith("tests/") and f.endswith(".py") and (root / f).is_file()
    )
    js_tests = sum(
        len(re.findall(r"^test\(", (root / f).read_text(encoding="utf-8"), re.M))
        for f in files
        if f.startswith("tests/") and f.endswith(".mjs")
    )
    return {"total_files": len(files), "areas": areas, "py_tests": py_tests, "js_tests": js_tests}


def git_revision(root: Path = PROJECT_DIR) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return f"{sha}{' (with uncommitted changes)' if dirty else ''}" if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


def _dates_span(values: pd.Series) -> str:
    values = values.dropna()
    return f"{values.min():%Y-%m-%d} to {values.max():%Y-%m-%d}" if len(values) else "n/a"


def _year(y: int) -> str:
    return f"{-y} BC" if y < 0 else str(y)


def _span_for(filename: str, path: Path) -> str:
    """Date coverage for the datasets that have a time axis; 'n/a' for the rest."""
    try:
        if filename == "database.csv":
            from ndp.ml.features import earthquake_date_to_epoch

            col = pd.read_csv(path, usecols=["Date"])["Date"].dropna()
            return _dates_span(pd.to_datetime(col.map(earthquake_date_to_epoch), unit="s"))
        if filename == "tsunami.csv":
            col = pd.read_csv(path, usecols=["YEAR"])["YEAR"]
            years = pd.to_numeric(col, errors="coerce").dropna()
            return f"{_year(int(years.min()))} to {_year(int(years.max()))}"
        if filename.startswith("hurricane-"):
            dates = pd.read_csv(path, usecols=["Date"])["Date"].astype(str)
            return _dates_span(pd.to_datetime(dates, format="%Y%m%d", errors="coerce"))
        if filename.startswith("Hoppers Crossing"):
            col = pd.read_csv(path, usecols=["Date/Time"])["Date/Time"]
            return _dates_span(pd.to_datetime(col, errors="coerce"))
        if filename.startswith("time_series_covid19"):
            columns = pd.read_csv(path, nrows=0).columns[4:]
            return _dates_span(
                pd.to_datetime(pd.Series(list(columns)), format="%m/%d/%y", errors="coerce")
            )
    except Exception:
        return "n/a"
    return "n/a"


def dataset_stats(data_dir: Path) -> list[dict[str, Any]]:
    from ndp.data import DATASETS

    rows = []
    for ds in DATASETS:
        path = data_dir / ds.filename
        if not path.exists():
            rows.append({"file": ds.filename, "used_by": ds.used_by, "present": False})
            continue
        df = pd.read_csv(path, low_memory=False)
        span = _span_for(ds.filename, path)
        rows.append(
            {
                "file": ds.filename,
                "used_by": ds.used_by,
                "present": True,
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
                "size": path.stat().st_size,
                "missing_pct": float(df.isna().mean().mean() * 100) if len(df) else 0.0,
                "span": span,
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

MODEL_LABELS = {
    "flood_level": "Flood forecast (3 h headline)",
    "tsunami_magnitude": "Tsunami magnitude",
    "hurricane_wind": "Hurricane wind",
    "storm_classifier": "Storm classifier",
    "covid_india_cases": "COVID-19 India cases",
}


def model_rows() -> list[list[str]]:
    rows = []
    for name, label in MODEL_LABELS.items():
        meta = registry.metadata(name)
        if not meta:
            rows.append([label, "not trained", "", "", "", "", ""])
            continue
        if "accuracy" in meta:
            result = f"accuracy {meta['accuracy']:.1%}"
            baseline = f"majority class {meta['baseline_accuracy']:.1%}"
        else:
            result = f"RMSE {fmt_num(meta.get('rmse'))}, R² {fmt_num(meta.get('r2'), 2)}"
            baseline = f"RMSE {fmt_num(meta.get('baseline_rmse'))}"
        cv = ""
        if "cv_r2_mean" in meta:
            cv = f"{meta['cv_r2_mean']:.2f} ± {meta['cv_r2_std']:.2f}"
        elif "cv_accuracy_mean" in meta:
            cv = f"{meta['cv_accuracy_mean']:.1%}"
        rows.append(
            [
                label,
                fmt_num(meta.get("n_rows")),
                result,
                baseline,
                f"{meta['skill']:+.2f}" if "skill" in meta else "n/a",
                cv or "n/a",
                meta.get("split", ""),
            ]
        )
    return rows


def flood_rows() -> list[list[str]]:
    meta = registry.metadata("flood_level") or {}
    out = []
    for hours, h in sorted((meta.get("horizons") or {}).items(), key=lambda kv: int(kv[0])):
        out.append(
            [
                f"{hours} h",
                fmt_num(h["rmse"], 4),
                fmt_num(h["persistence_rmse"], 4),
                f"{h['skill_vs_persistence']:+.2f}",
                fmt_num(h.get("rise_events")),
                "n/a" if h.get("rise_recall") is None else f"{h['rise_recall']:.0%}",
                "n/a" if h.get("rise_precision") is None else f"{h['rise_precision']:.0%}",
            ]
        )
    return out


def compare_rows() -> dict[str, tuple[str, list[list[str]]]]:
    out: dict[str, tuple[str, list[list[str]]]] = {}
    for hazard in compare.HAZARDS:
        meta = registry.metadata(compare.artifact_name(hazard))
        if not meta:
            continue
        rows = []
        for name in compare.MODEL_NAMES:
            m = meta["metrics"].get(name)
            if m:
                rows.append(
                    [
                        name,
                        f"{m['accuracy_pct']:.1f}%",
                        fmt_num(m["r2"], 2),
                        fmt_num(m["rmse"]),
                        f"{m.get('skill', 0.0):+.2f}",
                    ]
                )
        n_rows = meta.get("n_rows")
        out[hazard] = (f"{n_rows:,}" if isinstance(n_rows, int) else "n/a", rows)
    return out


def hazard_examples() -> list[list[str]]:
    from ndp.ml import hazard

    if not registry.is_trained(hazard.MODEL):
        return []
    places = [
        ("Tokyo", 35.7, 139.7),
        ("Los Angeles", 34.05, -118.25),
        ("Kathmandu", 27.7, 85.3),
        ("Santiago", -33.45, -70.67),
        ("Delhi", 28.6, 77.2),
        ("London", 51.5, -0.12),
        ("Mid-Atlantic ridge", 0.0, -30.0),
        ("South Atlantic (open ocean)", -30.0, -10.0),
    ]
    rows = []
    for name, lat, lon in places:
        e = hazard.predict(lat, lon, 300.0)
        rows.append(
            [
                name,
                fmt_num(e.n_events),
                f"{e.annual_rate_m55:.2f}",
                f"{e.b_value:.2f}" + (" (global)" if e.b_is_global else " (local)"),
                f"{e.prob_10y['6.0']:.0%}",
                f"{e.prob_10y['7.0']:.0%}",
                f"{e.prob_10y['8.0']:.1%}",
                e.level,
            ]
        )
    return rows


def flood_examples() -> list[list[str]]:
    from ndp.ml import flood

    if not registry.is_trained(flood.MODEL):
        return []
    cases = [
        ("Quiet day", (1.42, 1.41, 1.40, 2.0, 3.0, 5.0)),
        ("Heavy rain, level rising", (1.90, 1.70, 1.50, 40.0, 60.0, 90.0)),
    ]
    rows = []
    for name, args in cases:
        f = flood.predict(*args)
        rows.append(
            [
                name,
                *(f"{h.level_m:.3f} m" for h in f.horizons),
                f"{f.peak_level_m - args[0]:+.3f} m",
            ]
        )
    return rows


def sentiment_rows() -> list[list[str]]:
    meta = registry.metadata(sentiment.SCORES) or {}
    scores = meta.get("scores") or {}
    detail = {}
    try:
        detail = sentiment.scores()
    except Exception:
        pass
    rows = []
    for d in sentiment.DISASTERS:
        s = detail.get(d.slug, {})
        rows.append(
            [
                d.name,
                f"{d.monetary_impact:g}",
                fmt_num(scores.get(d.slug), 1) if d.slug in scores else "n/a",
                fmt_num(s.get("n_posts")),
                fmt_num(s.get("n_sentences")),
            ]
        )
    return rows


def artifact_sizes() -> list[list[str]]:
    models_dir = get_settings().models_dir
    if not models_dir.exists():
        return []
    items = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_size, reverse=True)
    return [[p.stem, fmt_size(p.stat().st_size)] for p in items]


def training_rows() -> tuple[list[list[str]], str]:
    if not registry.is_trained("training_run"):
        return [], ""
    seconds = registry.load("training_run")["seconds"]
    meta = registry.metadata("training_run") or {}
    rows = [[name, f"{secs:.1f} s"] for name, secs in seconds.items()]
    note = f"Last full run total: {meta.get('last_run_total_seconds', 'n/a')} s"
    return rows, note


# --------------------------------------------------------------------------------------
# Front-end footprint and benchmark
# --------------------------------------------------------------------------------------


def static_rows() -> list[list[str]]:
    static = PACKAGE_DIR / "static"
    rows = []
    for sub in ("css", "js"):
        for p in sorted((static / sub).glob("*")):
            raw = p.read_bytes()
            rows.append(
                [f"{sub}/{p.name}", fmt_size(len(raw)), fmt_size(len(gzip.compress(raw, 6)))]
            )
    return rows


def bench(runs: int = 40) -> list[list[str]]:
    """Time model-backed endpoints in-process (no network, one worker). Pages that call external
    services are excluded on purpose: their latency is the other service's."""
    from fastapi.testclient import TestClient

    from ndp.main import create_app

    client = TestClient(create_app())
    hx = {"HX-Request": "true"}
    cases: list[tuple[str, Callable[[], Any]]] = [
        ("GET /", lambda: client.get("/")),
        ("GET /earthquake", lambda: client.get("/earthquake")),
        (
            "POST /earthquake/predict (hazard)",
            lambda: client.post(
                "/earthquake/predict",
                data={"lat": "35.7", "lon": "139.7", "radius_km": "300"},
                headers=hx,
            ),
        ),
        (
            "POST /flood/predict (3 models)",
            lambda: client.post(
                "/flood/predict",
                data={
                    "level_now": "1.42",
                    "level_1h_ago": "1.41",
                    "level_3h_ago": "1.40",
                    "rain_3h": "2",
                    "rain_6h": "3",
                    "rain_24h": "5",
                },
                headers=hx,
            ),
        ),
        (
            "POST /storm/predict",
            lambda: client.post(
                "/storm/predict",
                data={
                    "temperature": "42",
                    "pressure": "1013",
                    "humidity": "93",
                    "wind_direction": "60",
                    "wind_speed": "4",
                },
                headers=hx,
            ),
        ),
        (
            "POST /compare/earthquake (5 models)",
            lambda: client.post(
                "/compare/earthquake",
                data={"lat": "28.6", "lon": "77.2", "depth": "10"},
                headers=hx,
            ),
        ),
        (
            "POST /api/v1/storm/predict (JSON)",
            lambda: client.post(
                "/api/v1/storm/predict",
                json={
                    "temperature_f": 42,
                    "pressure_hpa": 1013,
                    "humidity_pct": 93,
                    "wind_direction_deg": 60,
                    "wind_speed": 4,
                },
            ),
        ),
    ]
    rows = []
    for label, call in cases:
        call()  # warm-up: lazy model loading is not what a steady-state request pays
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            resp = call()
            times.append((time.perf_counter() - t0) * 1000)
            if resp.status_code >= 500:
                times = []
                break
        if not times:
            rows.append([label, "n/a", "n/a", "n/a", "failed (model not trained?)"])
            continue
        times.sort()
        rows.append(
            [
                label,
                f"{statistics.median(times):.1f}",
                f"{times[int(0.95 * (len(times) - 1))]:.1f}",
                f"{times[-1]:.1f}",
                str(runs),
            ]
        )
    return rows


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def build(include_bench: bool = False, root: Path = PROJECT_DIR) -> str:
    settings = get_settings()
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = [
        "# Statistics",
        "",
        f"Generated by `uv run ndp-report` on {now}, application version {__version__}, "
        f"commit {git_revision(root)}, Python {sys.version.split()[0]}.",
        "Every number below is computed from the repository, the datasets in `data/raw/` and the "
        "trained models in `models/`. Regenerate with `uv run ndp-report --write docs/statistics.md` "
        "(add `--bench` to time the endpoints).",
        "",
    ]

    # ---- repository
    stats = repo_stats(root)
    out += ["## Repository", ""]
    rows = [[a["area"], a["files"], f"{a['lines']:,}"] for a in stats["areas"]]
    rows.append(["**All tracked files**", stats["total_files"], ""])
    out += [table(["Area", "Files", "Non-blank lines"], rows, align_right=(1, 2)), ""]
    out += [
        f"Test functions: {stats['py_tests']} Python and {stats['js_tests']} JavaScript "
        "(parametrised cases run as several tests, so `pytest` reports more).",
        "",
    ]

    # ---- datasets
    out += ["## Datasets", ""]
    ds = dataset_stats(settings.data_dir)
    if not any(d["present"] for d in ds):
        out += ["No datasets found. Run `uv run ndp-fetch-data`.", ""]
    else:
        rows = [
            [
                d["file"],
                d["used_by"],
                fmt_num(d["rows"]),
                d["cols"],
                fmt_size(d["size"]),
                f"{d['missing_pct']:.1f}%",
                d["span"],
            ]
            if d["present"]
            else [d["file"], d["used_by"], "missing", "", "", "", ""]
            for d in ds
        ]
        out += [
            table(
                ["File", "Used by", "Rows", "Columns", "Size", "Missing cells", "Span"],
                rows,
                align_right=(2, 3, 4, 5),
            ),
            "",
        ]
        total = sum(d.get("size", 0) for d in ds if d["present"])
        out += [f"Total raw data: {fmt_size(total)} in {sum(d['present'] for d in ds)} files.", ""]

    # ---- models
    out += ["## Models", ""]
    out += [
        table(
            ["Model", "Rows", "Held-out result", "Baseline", "Skill", "CV", "Split"],
            model_rows(),
            align_right=(1, 4),
        ),
        "",
        "Skill is `1 - RMSE / baseline RMSE` (classifier: `(accuracy - baseline) / (1 - baseline)`). "
        "0 means no better than the baseline; negative means worse.",
        "",
    ]

    flood = flood_rows()
    if flood:
        out += [
            "### Flood forecast by horizon",
            "",
            table(
                [
                    "Horizon",
                    "Model RMSE (m)",
                    "Persistence RMSE (m)",
                    "Skill vs persistence",
                    "Rises of 10 cm or more",
                    "Recall",
                    "Precision",
                ],
                flood,
                align_right=(1, 2, 3, 4, 5, 6),
            ),
            "",
            'Persistence is the forecast "the level stays where it is". Recall is the share of the '
            "held-out rises of 10 cm or more that the model also forecast.",
            "",
        ]
        examples = flood_examples()
        if examples:
            out += [
                "Example forecasts (inputs: level now, 1 h ago, 3 h ago in metres; rain over 3, 6, 24 h in mm):",
                "",
                table(
                    ["Case", "+1 h", "+3 h", "+6 h", "Peak vs now"],
                    examples,
                    align_right=(1, 2, 3, 4),
                ),
                "",
            ]

    meta = registry.metadata("earthquake_hazard")
    if meta:
        bt = meta.get("backtest", {})
        out += [
            "### Earthquake hazard",
            "",
            f"Catalogue: {meta['n_events']:,} events over {meta['years_span']:.0f} years "
            f"({meta.get('year_min')} to {meta.get('year_max')}), completeness magnitude "
            f"{meta.get('mc')}, global b-value {meta.get('b_value_global', 0):.2f}.",
            "",
            f"Back-test (cells with events before 2000, tested on 2000 onward): the active cells cover "
            f"{bt.get('share_of_globe_in_active_cells', 0):.0%} of the globe and hold "
            f"{bt.get('share_of_test_events_in_active_cells', 0):.0%} of the {bt.get('test_events_m6', 0):,} later "
            f"M 6 and larger events, a lift of {bt.get('lift', 0):.1f} over a uniform guess.",
            "",
        ]
        rows = hazard_examples()
        if rows:
            out += [
                "Example estimates (300 km radius):",
                "",
                table(
                    [
                        "Place",
                        "Events",
                        "Rate M 5.5+ per year",
                        "b-value",
                        "P(M 6+ in 10 y)",
                        "P(M 7+ in 10 y)",
                        "P(M 8+ in 10 y)",
                        "Level",
                    ],
                    rows,
                    align_right=(1, 2, 4, 5, 6),
                ),
                "",
                "Counts include aftershocks (no declustering), which inflates busy regions. "
                "Places with fewer than about ten events have noisy rates.",
                "",
            ]

    compare_all = compare_rows()
    if compare_all:
        out += ["### Model comparison (shuffled 75/25 splits)", ""]
        for hazard, (n_rows, rows) in compare_all.items():
            out += [
                f"**{compare.HAZARDS[hazard].name}** ({n_rows} rows after subsampling)",
                "",
                table(
                    ["Model", "Tolerance accuracy", "R²", "RMSE", "Skill"],
                    rows,
                    align_right=(1, 2, 3, 4),
                ),
                "",
            ]
        out += [
            "Tolerance accuracy is the share of test rows within 10% of the largest target value, "
            "which is generous: the mean-only baseline often scores above 90%. Use RMSE and skill.",
            "",
        ]

    out += ["### Sentiment scores", ""]
    out += [
        table(
            [
                "Disaster",
                "Monetary impact (2020 figure, unit undocumented)",
                "Sentiment score",
                "Posts",
                "Sentences",
            ],
            sentiment_rows(),
            align_right=(1, 2, 3, 4),
        ),
        "",
        "The score is the sum over (date, user) groups of mean sentence sentiment from SentiWordNet, "
        "so it grows with the amount of discussion as well as its tone.",
        "",
    ]

    # ---- training and footprint
    rows, note = training_rows()
    if rows:
        out += [
            "## Training time",
            "",
            table(["Trainer", "Seconds"], rows, align_right=(1,)),
            "",
            note,
            "",
        ]
    sizes = artifact_sizes()
    if sizes:
        out += [
            "## Model artifacts on disk",
            "",
            table(["Artifact", "Size"], sizes, align_right=(1,)),
            "",
        ]
    out += [
        "## Front-end payload",
        "",
        table(["File", "Size", "Gzipped"], static_rows(), align_right=(1, 2)),
        "",
        "Leaflet, HTMX and globe.gl load from a CDN; the globe library is only loaded on the home page.",
        "",
    ]

    if include_bench:
        out += [
            "## Endpoint latency (in-process, single worker)",
            "",
            table(
                ["Request", "Median (ms)", "p95 (ms)", "Max (ms)", "Runs"],
                bench(),
                align_right=(1, 2, 3),
            ),
            "",
            "Measured with FastAPI's test client on the machine that generated this file, after one "
            "warm-up call per endpoint. It includes template rendering and model inference but no "
            "network, so treat it as a relative guide, not a service-level figure.",
            "",
        ]
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ndp-report", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--write", metavar="PATH", help="write the Markdown here instead of printing"
    )
    parser.add_argument("--bench", action="store_true", help="also time the model-backed endpoints")
    args = parser.parse_args(argv)
    text = build(include_bench=args.bench)
    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target} ({len(text.splitlines())} lines)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
