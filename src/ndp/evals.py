"""Evaluation harness: ``ndp-eval assistant`` (and ``ndp-eval triage``).

    uv run python -m ndp.evals assistant            # needs ANTHROPIC_API_KEY
    uv run python -m ndp.evals assistant --check    # validate the case file only, no API
    uv run python -m ndp.evals assistant --limit 5 --json /tmp/report.json
    uv run python -m ndp.evals triage               # delegates to ndp.evals_triage.run

Without an API key the assistant suite prints ``skipped: no ANTHROPIC_API_KEY`` and exits 0, so
it is safe in CI. With a key it asks every case in ``evals/assistant_cases.jsonl`` in a fresh
session (real tool calls against live services, real model), grades the answer and reports pass
rates, mean tool rounds, token usage and an estimated cost.

Case fields (JSON per line): ``id``, ``category``, ``question``, ``expect_tools_any`` (at least one
of these tools must be called), ``expect_no_tools`` (no tool may be called), ``forbid_tools``,
``must_mention`` (every regex must match), ``must_not`` (no regex may match), ``expect_sources``
(at least one source link), ``notes``.

THE GRADERS ARE DETERMINISTIC CODE, NOT A JUDGE, AND THEY HAVE LIMITS. Read a pass rate with
these in mind:

* Regexes see words, not meaning. ``must_not`` cannot tell "I can predict" from a quoted
  "I can predict nothing"; the patterns are written to avoid the common negations
  ("cannot", "can't") but a determined paraphrase slips through in either direction.
* ``must_mention`` proves a keyword is present, not that the claim around it is right. A cited
  URL is checked for presence, not for whether the excerpt supports the sentence.
* Tool expectations check which tools were called, not whether the arguments were sensible or
  the result was used correctly.
* Cases hit live services (USGS, Open-Meteo, GDACS, OpenStreetMap) and a non-deterministic model,
  so the same case can pass on one run and fail on the next. Run it more than once before
  trusting a difference of a few points.
* 30-odd handwritten cases are a smoke test and a regression tripwire, not a benchmark.
* Cost is an estimate from the PRICES table (token counts times list prices); the table is a
  snapshot and must be updated when prices or the default model change.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "assistant_cases.jsonl"

# USD per million tokens: (input, output, cache read). Cache writes (5 minute TTL) are billed at
# 1.25x the input price. Snapshot of the claude-api skill's model catalogue (shared/models.md and
# shared/prompt-caching.md); verify against the current pricing page before quoting a number.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),  # prompts of 100K tokens or fewer
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00, 1.00),
}
CACHE_WRITE_MULTIPLIER = 1.25

# Applied to every case: the assistant must never claim to have sent anything.
GLOBAL_MUST_NOT = (
    r"\b(i have|i've|i just|i)\s+(sent|dispatched|emailed|texted|messaged)\b",
    r"\b(alert|alerts|message|messages|sms|email|emails|text)\s+(was|were|has been|have been)\s+"
    r"(sent|dispatched|delivered)\b",
    r"\bi\s*(will|'ll|am going to)\s+(send|email|text|message)\b",
    r"\bi\s+predict\b",
)


@dataclass(frozen=True)
class Verdict:
    id: str
    passed: bool
    tools_ok: bool
    mention_ok: bool
    must_not_ok: bool
    problems: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------------


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or CASES_PATH
    if not path.exists():
        raise FileNotFoundError(f"assistant eval cases not found at {path}")
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: not valid JSON ({exc})") from exc
    return cases


def validate_cases(cases: Sequence[dict[str, Any]]) -> list[str]:
    """Problems with the case file itself (so a typo cannot silently weaken a test)."""
    from ndp.services import assistant

    known_tools = {t.name for t in assistant.TOOLS}
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        cid = str(case.get("id", "?"))
        if cid in seen or cid == "?":
            problems.append(f"{cid}: missing or duplicate id")
        seen.add(cid)
        if not str(case.get("question", "")).strip():
            problems.append(f"{cid}: empty question")
        for key in ("expect_tools_any", "forbid_tools"):
            for name in case.get(key, []):
                if name not in known_tools:
                    problems.append(f"{cid}: {key} names unknown tool {name!r}")
        if case.get("expect_no_tools") and case.get("expect_tools_any"):
            problems.append(f"{cid}: expect_no_tools contradicts expect_tools_any")
        for key in ("must_mention", "must_not"):
            for pattern in case.get(key, []):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    problems.append(f"{cid}: bad regex in {key}: {pattern!r} ({exc})")
    return problems


# --------------------------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------------------------


def _search(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None


def grade_case(
    case: dict[str, Any],
    answer: str,
    tools_called: Sequence[str],
    source_urls: Sequence[str] = (),
    refused: bool = False,
) -> Verdict:
    """Deterministic grade of one answer. Pure function: no I/O, no model."""
    problems: list[str] = []
    called = set(tools_called)

    tools_ok = True
    expected_any = case.get("expect_tools_any") or []
    if expected_any and not called & set(expected_any):
        tools_ok = False
        problems.append(f"called none of {expected_any} (called: {sorted(called) or 'nothing'})")
    if case.get("expect_no_tools") and called:
        tools_ok = False
        problems.append(f"expected no tool calls, got {sorted(called)}")
    forbidden = called & set(case.get("forbid_tools") or [])
    if forbidden:
        tools_ok = False
        problems.append(f"called forbidden tools {sorted(forbidden)}")
    if case.get("expect_sources") and not source_urls:
        tools_ok = False
        problems.append("no source links returned")

    mention_ok = True
    for pattern in case.get("must_mention") or []:
        if not _search(pattern, answer):
            mention_ok = False
            problems.append(f"missing /{pattern}/")

    must_not_ok = True
    for pattern in [*(case.get("must_not") or []), *GLOBAL_MUST_NOT]:
        if _search(pattern, answer):
            must_not_ok = False
            problems.append(f"forbidden /{pattern}/ matched")

    if refused:
        problems.append("API refusal")
    if not answer.strip():
        problems.append("empty answer")

    passed = tools_ok and mention_ok and must_not_ok and not refused and bool(answer.strip())
    return Verdict(str(case.get("id", "?")), passed, tools_ok, mention_ok, must_not_ok, problems)


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    """Estimated USD for token totals, or None for a model missing from PRICES."""
    price = PRICES.get(model)
    if price is None:
        return None
    inp, out, cache_read = price
    total = (
        usage.get("input_tokens", 0) * inp
        + usage.get("output_tokens", 0) * out
        + usage.get("cache_read_tokens", 0) * cache_read
        + usage.get("cache_write_tokens", 0) * inp * CACHE_WRITE_MULTIPLIER
    )
    return total / 1_000_000


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def summarise(rows: Sequence[dict[str, Any]], model: str = "") -> dict[str, Any]:
    """Aggregate per-case rows into the report."""
    answered = [r for r in rows if not r.get("error")]
    with_tool_expectation = [r for r in answered if r["case"].get("expect_tools_any")]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    for r in answered:
        for key in usage:
            usage[key] += r["usage"].get(key, 0)
    verdicts = [r["verdict"] for r in answered]
    cost = estimate_cost(model, usage)
    return {
        "model": model,
        "n_cases": len(rows),
        "n_answered": len(answered),
        "n_errors": len(rows) - len(answered),
        "pass_rate": _rate(sum(v.passed for v in verdicts), len(rows)),
        "tool_use_rate": _rate(
            sum(r["verdict"].tools_ok for r in with_tool_expectation), len(with_tool_expectation)
        ),
        "must_mention_rate": _rate(sum(v.mention_ok for v in verdicts), len(verdicts)),
        "must_not_rate": _rate(sum(v.must_not_ok for v in verdicts), len(verdicts)),
        "mean_tool_rounds": (
            sum(r["rounds"] for r in answered) / len(answered) if answered else None
        ),
        "tokens": usage,
        "estimated_cost_usd": cost,
        "cost_per_case_usd": cost / len(answered) if cost is not None and answered else None,
    }


def run_assistant(
    cases: Sequence[dict[str, Any]],
    ask: Callable[[str, str], Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Ask every case in a fresh session and grade it. ``ask`` is injectable for tests."""
    from ndp.services import ServiceError, assistant, llm

    ask = ask or assistant.ask
    model = model or llm.model_name()
    rows: list[dict[str, Any]] = []
    for case in cases:
        session_id = assistant.new_session_id()
        row: dict[str, Any] = {"case": case}
        try:
            turn = ask(session_id, case["question"])
        except ServiceError as exc:
            row["error"] = str(exc)
        else:
            names = [c.name for c in turn.tools_used]
            row["answer"] = turn.answer
            row["tools"] = names
            row["rounds"] = turn.rounds
            row["usage"] = asdict(turn.usage)
            row["verdict"] = grade_case(
                case, turn.answer, names, [s.url for s in turn.sources], turn.refused
            )
        finally:
            assistant.sessions.reset(session_id)
        rows.append(row)
    report = summarise(rows, model)
    report["cases"] = [
        {
            "id": r["case"].get("id"),
            "category": r["case"].get("category"),
            "passed": r["verdict"].passed if "verdict" in r else False,
            "problems": r["verdict"].problems if "verdict" in r else [r.get("error", "error")],
            "tools": r.get("tools", []),
            "rounds": r.get("rounds"),
            "answer": (r.get("answer") or "")[:600],
        }
        for r in rows
    ]
    report["caveat"] = (
        "Regex graders on a small handwritten set against live services and a non-deterministic "
        "model: a smoke test, not a benchmark. See the module docstring for limits."
    )
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def print_report(report: dict[str, Any]) -> None:
    print(f"Assistant eval ({report['model']}): {report['n_cases']} cases")
    print(f"  pass rate            {_pct(report['pass_rate'])}")
    print(f"  tool-use rate        {_pct(report['tool_use_rate'])}  (cases that expect a tool)")
    print(f"  must_mention rate    {_pct(report['must_mention_rate'])}")
    print(f"  must_not rate        {_pct(report['must_not_rate'])}  (no forbidden phrase matched)")
    rounds = report["mean_tool_rounds"]
    print(f"  mean tool rounds     {'n/a' if rounds is None else f'{rounds:.2f}'}")
    t = report["tokens"]
    print(
        f"  tokens               in {t['input_tokens']:,}  out {t['output_tokens']:,}  "
        f"cache read {t['cache_read_tokens']:,}  cache write {t['cache_write_tokens']:,}"
    )
    cost = report["estimated_cost_usd"]
    if cost is None:
        print(f"  estimated cost       unknown (no price for {report['model']} in PRICES)")
    else:
        per = report["cost_per_case_usd"]
        print(f"  estimated cost       ${cost:.4f} total, ${per:.4f} per case")
    if report["n_errors"]:
        print(f"  API/service errors   {report['n_errors']}")
    for row in report["cases"]:
        if not row["passed"]:
            print(f"  FAIL {row['id']}: " + "; ".join(row["problems"]))
    print(f"  {report['caveat']}")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _assistant_command(args: argparse.Namespace) -> int:
    from ndp.services import llm

    try:
        cases = load_cases(args.cases)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    problems = validate_cases(cases)
    if problems:
        print("error: the case file has problems:")
        for p in problems:
            print(f"  {p}")
        return 2
    print(f"{len(cases)} assistant cases loaded and valid.")
    if args.check:
        return 0
    if not llm.is_configured():
        print(
            "skipped: no ANTHROPIC_API_KEY (set ANTHROPIC_API_KEY or NDP_ANTHROPIC_API_KEY to "
            "run the assistant eval; it makes real API calls)"
        )
        return 0
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    report = run_assistant(cases)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"report written to {args.json}")
    return 0


def _triage_command(args: argparse.Namespace) -> int:
    """Hook for the triage suite (``ndp.evals_triage.run``), written separately."""
    try:
        from ndp import evals_triage
    except ImportError:
        print("triage suite not available")
        return 0
    evals_triage.run(limit=args.limit)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ndp-eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="suite", required=True)

    a = sub.add_parser("assistant", help="run the assistant cases (needs an API key)")
    a.add_argument("--cases", type=Path, default=None, help="alternative cases file")
    a.add_argument("--limit", type=int, default=None, help="only the first N cases")
    a.add_argument("--ids", nargs="*", help="only these case ids")
    a.add_argument("--json", default=None, help="write the full report to this file")
    a.add_argument("--check", action="store_true", help="validate the case file and stop")
    a.set_defaults(func=_assistant_command)

    t = sub.add_parser("triage", help="run the post-triage cases (needs an API key)")
    t.add_argument("--limit", type=int, default=None, help="only the first N cases")
    t.set_defaults(func=_triage_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
