"""Agreement check for post triage against ``evals/triage_cases.jsonl``.

READ THIS BEFORE QUOTING A NUMBER. The cases are 22 posts written by the author, each with the
labels the author thinks are acceptable (``expect_needs_any``: at least one predicted need must be
in this list; ``expect_urgency``: the predicted band must be one of these). The score therefore
measures **agreement with the author's own labels** on a handful of easy-to-hard examples in four
language styles. It is not an accuracy benchmark, it says nothing about real-world posts, and it
must not be used to argue that triage is safe to rely on. A person verifies every suggestion in
the product regardless of this number.

Grading is deterministic (no model judges another model):

* ``needs_overlap_rate``: share of answered cases whose predicted needs intersect the accepted set.
* ``urgency_in_band_rate``: share of answered cases whose urgency is one of the accepted bands.
* ``under_triage_rate``: share of answered cases whose urgency is *below* the lowest accepted band
  (the dangerous direction for a triage aid).
* ``parse_failure_rate``: share of all cases where the call raised ``ServiceError`` (API error,
  refusal, or output that did not match the schema).
* ``tokens``: totals from ``llm.usage_totals()`` if the shared client exposes it, else ``None``
  (``llm.parse`` currently returns only the parsed object, so usage is not visible here).

Without an API key the run prints a skip message and returns ``{"skipped": True, ...}``.

    uv run python -m ndp.evals_triage          # coordinator wires this into ``ndp-eval triage``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ndp.services import ServiceError, llm, triage

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "triage_cases.jsonl"
_BAND_ORDER = {band: i for i, band in enumerate(triage.URGENCY_BANDS)}


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or CASES_PATH
    if not path.exists():
        raise FileNotFoundError(f"triage eval cases not found at {path}")
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def grade_case(case: dict[str, Any], result: triage.Triage) -> dict[str, bool]:
    """Deterministic grading of one answer against the author's labels."""
    accepted_needs = set(case.get("expect_needs_any", []))
    accepted_bands = list(case.get("expect_urgency", []))
    needs_ok = bool(accepted_needs & set(result.needs)) if accepted_needs else True
    urgency_ok = result.urgency in accepted_bands if accepted_bands else True
    lowest = min((_BAND_ORDER[b] for b in accepted_bands if b in _BAND_ORDER), default=None)
    under = lowest is not None and _BAND_ORDER[result.urgency] < lowest
    return {"needs_ok": needs_ok, "urgency_ok": urgency_ok, "under_triaged": under}


def _usage() -> dict | None:
    getter = getattr(llm, "usage_totals", None)
    return getter() if callable(getter) else None


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 3) if total else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case rows (``error`` set or the three grade flags) into the report."""
    answered = [r for r in rows if not r.get("error")]
    n_ans = len(answered)
    return {
        "n_cases": len(rows),
        "n_answered": n_ans,
        "needs_overlap_rate": _rate(sum(r["needs_ok"] for r in answered), n_ans),
        "urgency_in_band_rate": _rate(sum(r["urgency_ok"] for r in answered), n_ans),
        "under_triage_rate": _rate(sum(r["under_triaged"] for r in answered), n_ans),
        "parse_failure_rate": _rate(len(rows) - n_ans, len(rows)),
    }


def run(limit: int | None = None, path: Path | None = None) -> dict[str, Any]:
    """Run the cases through ``triage.triage_post`` and return agreement rates."""
    if not llm.is_configured():
        message = (
            "SKIPPED: no Anthropic API key. Set ANTHROPIC_API_KEY (or NDP_ANTHROPIC_API_KEY) to "
            "run the triage eval; it makes one model call per case."
        )
        print(message)
        return {"skipped": True, "reason": message}
    cases = load_cases(path)
    if limit is not None:
        cases = cases[: max(0, limit)]
    before = _usage()
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {"text": case["text"][:80], "notes": case.get("notes", "")}
        try:
            result = triage.triage_post(case["text"])
        except ServiceError as exc:
            row["error"] = str(exc)
        else:
            row.update(grade_case(case, result), needs=result.needs, urgency=result.urgency)
        rows.append(row)
    report = summarise(rows)
    after = _usage()
    if before is not None and after is not None:
        report["tokens"] = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    else:
        report["tokens"] = None
    report["model"] = llm.model_name()
    report["cases"] = rows
    report["caveat"] = (
        "Agreement with the author's own labels on a small handwritten set; not an accuracy "
        "benchmark."
    )
    _print(report)
    return report


def _print(report: dict[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    print(f"Triage eval ({report['model']}): {report['n_answered']}/{report['n_cases']} answered")
    print(f"  needs overlap       {pct(report['needs_overlap_rate'])}")
    print(f"  urgency in band     {pct(report['urgency_in_band_rate'])}")
    print(f"  under-triaged       {pct(report['under_triage_rate'])}")
    print(f"  call/parse failures {pct(report['parse_failure_rate'])}")
    print(f"  tokens              {report['tokens'] if report['tokens'] else 'not reported'}")
    print(f"  {report['caveat']}")
    for row in report["cases"]:
        if row.get("error"):
            print(f"  ERROR  {row['text']!r}: {row['error']}")
        elif not (row["needs_ok"] and row["urgency_ok"]):
            print(
                f"  MISS   {row['text']!r}: got {row['needs']} / {row['urgency']} ({row['notes']})"
            )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
