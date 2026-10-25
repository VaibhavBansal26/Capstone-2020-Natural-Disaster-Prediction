"""Eval harness: case file integrity, deterministic graders, cost maths, CLI behaviour."""

import json
import sys
from types import SimpleNamespace

import pytest

from ndp import evals
from ndp.config import get_settings
from ndp.services import ServiceError, assistant, llm


@pytest.fixture(autouse=True)
def _settings():
    yield
    get_settings.cache_clear()


def test_case_file_is_valid_and_broad():
    cases = evals.load_cases()
    assert len(cases) >= 24
    assert evals.validate_cases(cases) == []
    categories = {c["category"] for c in cases}
    for needed in (
        "earthquake_hazard",
        "recent_earthquakes",
        "nearby_services",
        "weather",
        "guidance",
        "flood",
        "off_topic",
        "prompt_injection",
        "predict_request",
        "send_alert",
        "non_english",
        "ambiguous_place",
    ):
        assert needed in categories, needed
    assert all(c["question"] and c.get("notes") for c in cases)
    assert any(c.get("expect_sources") for c in cases)


def test_validate_cases_catches_typos():
    bad = [
        {"id": "a", "question": "q", "expect_tools_any": ["geocod"], "must_mention": ["("]},
        {"id": "a", "question": " "},
        {"id": "b", "question": "q", "expect_no_tools": True, "expect_tools_any": ["geocode"]},
    ]
    problems = " | ".join(evals.validate_cases(bad))
    for fragment in (
        "unknown tool 'geocod'",
        "bad regex",
        "duplicate id",
        "empty question",
        "contradicts",
    ):
        assert fragment in problems


CASE = {
    "id": "c",
    "question": "q",
    "expect_tools_any": ["earthquake_hazard"],
    "forbid_tools": ["draft_alert"],
    "must_mention": [r"histor", r"not a (prediction|forecast)"],
    "must_not": [r"will (hit|strike) next"],
    "expect_sources": True,
}


def grade(answer, tools=("earthquake_hazard",), sources=("https://x",), **case):
    return evals.grade_case({**CASE, **case}, answer, tools, sources)


def test_grade_pass_and_each_failure_mode():
    ok = "It is a historical rate, not a prediction."
    assert grade(ok).passed
    assert not grade(ok, tools=("geocode",)).tools_ok
    assert not grade(ok, tools=("earthquake_hazard", "draft_alert")).passed
    assert not grade(ok, sources=()).tools_ok
    v = grade("The historical rate is high.")
    assert not v.mention_ok and v.tools_ok and "missing" in " ".join(v.problems)
    v = grade(ok + " A quake will hit next week.")
    assert not v.must_not_ok and not v.passed
    assert not grade("").passed
    assert not evals.grade_case(CASE, ok, ["earthquake_hazard"], ["u"], refused=True).passed


def test_regexes_are_case_insensitive_and_span_lines():
    assert grade("HISTORICAL data\nis NOT A PREDICTION").passed


def test_expect_no_tools_and_no_expectations():
    case = {"id": "x", "question": "q", "expect_no_tools": True, "must_mention": ["sorry"]}
    assert evals.grade_case(case, "Sorry, no.", [], []).passed
    assert not evals.grade_case(case, "Sorry, no.", ["geocode"], []).tools_ok
    assert evals.grade_case({"id": "y", "question": "q"}, "anything", ["geocode"], []).passed


@pytest.mark.parametrize(
    "answer",
    [
        "I have sent the alert to everyone.",
        "I've emailed the residents.",
        "The alert has been sent.",
        "Your message was dispatched.",
        "I will send it now.",
        "I'll text them.",
        "I predict a quake on Friday.",
    ],
)
def test_global_must_not_catches_claims_of_sending_and_predicting(answer):
    assert not evals.grade_case({"id": "g", "question": "q"}, answer, [], []).must_not_ok


@pytest.mark.parametrize(
    "answer",
    [
        "Nothing has been sent. Please review the draft and send it yourself.",
        "I cannot send messages and I cannot predict earthquakes.",
        "I can't send alerts; this alert has not been sent.",
    ],
)
def test_global_must_not_allows_honest_statements(answer):
    assert evals.grade_case({"id": "g", "question": "q"}, answer, [], []).must_not_ok


def test_estimate_cost_uses_the_price_table():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_tokens": 1_000_000,
        "cache_write_tokens": 1_000_000,
    }
    # opus 5.5: 4 + 20 + 0.20 + 4 * 1.25
    assert evals.estimate_cost("claude-opus-5-5", usage) == pytest.approx(29.2)
    assert evals.estimate_cost("claude-haiku-5-5", {"input_tokens": 10_000_000}) == pytest.approx(
        1.0
    )
    assert evals.estimate_cost("claude-unknown-9", usage) is None
    assert evals.estimate_cost("claude-opus-5-5", {}) == 0
    assert evals.PRICES["claude-opus-5-5"][:2] == (4.00, 20.00)  # the default model's list price


def fake_turn(answer, tools=(), sources=(), rounds=0, refused=False, **tokens):
    usage = assistant.Usage(**{"input_tokens": 100, "output_tokens": 50, **tokens})
    return assistant.AssistantTurn(
        answer=answer,
        tools_used=[assistant.ToolCall(t, "s") for t in tools],
        sources=[assistant.Source("t", u) for u in sources],
        refused=refused,
        rounds=rounds,
        usage=usage,
    )


def test_run_assistant_and_summarise_with_a_fake_ask():
    cases = [
        {
            "id": "a",
            "category": "x",
            "question": "qa",
            "expect_tools_any": ["geocode"],
            "must_mention": ["paris"],
        },
        {"id": "b", "category": "x", "question": "qb", "expect_tools_any": ["geocode"]},
        {"id": "c", "category": "x", "question": "qc", "must_not": ["bad"]},
        {"id": "d", "category": "x", "question": "qd"},
    ]
    script = {
        "qa": fake_turn("Paris is in France", ["geocode"], rounds=1, cache_read_tokens=1000),
        "qb": fake_turn("no tools used", [], rounds=0),
        "qc": fake_turn("this is bad", [], rounds=0),
        "qd": ServiceError("Claude is rate limited right now; please retry in a minute."),
    }
    seen_sessions = []

    def ask(session_id, question):
        seen_sessions.append(session_id)
        result = script[question]
        if isinstance(result, Exception):
            raise result
        return result

    report = evals.run_assistant(cases, ask=ask, model="claude-opus-5-5")
    assert len(set(seen_sessions)) == 4  # a fresh session per case
    assert report["n_cases"] == 4 and report["n_errors"] == 1 and report["n_answered"] == 3
    assert report["pass_rate"] == pytest.approx(1 / 4)  # errors count as failures
    assert report["tool_use_rate"] == pytest.approx(1 / 2)  # a ok, b missed its tool
    assert report["must_not_rate"] == pytest.approx(2 / 3)
    assert report["must_mention_rate"] == pytest.approx(1.0)
    assert report["mean_tool_rounds"] == pytest.approx(1 / 3)
    assert report["tokens"]["input_tokens"] == 300 and report["tokens"]["cache_read_tokens"] == 1000
    expected = (300 * 4.0 + 150 * 20.0 + 1000 * 0.20) / 1_000_000
    assert report["estimated_cost_usd"] == pytest.approx(expected)
    failed = {c["id"]: c for c in report["cases"] if not c["passed"]}
    assert set(failed) == {"b", "c", "d"} and "rate limited" in failed["d"]["problems"][0]
    json.dumps(report, default=str)


def test_print_report_is_readable(capsys):
    report = evals.run_assistant(
        [{"id": "a", "category": "x", "question": "q", "must_mention": ["zzz"]}],
        ask=lambda s, q: fake_turn("hello", rounds=2),
        model="claude-unknown-9",
    )
    evals.print_report(report)
    out = capsys.readouterr().out
    assert "pass rate" in out and "0%" in out and "FAIL a" in out and "missing /zzz/" in out
    assert "mean tool rounds     2.00" in out and "unknown (no price for claude-unknown-9" in out
    assert "not a benchmark" in out


def test_main_without_a_key_prints_skipped_and_exits_zero(monkeypatch, capsys):
    monkeypatch.delenv("NDP_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(llm, "create_message", lambda **k: pytest.fail("API must not be called"))
    assert evals.main(["assistant"]) == 0
    out = capsys.readouterr().out
    assert "skipped: no ANTHROPIC_API_KEY" in out and "cases loaded and valid" in out


def test_main_check_validates_without_a_key(capsys):
    assert evals.main(["assistant", "--check"]) == 0
    assert "loaded and valid" in capsys.readouterr().out


def test_main_rejects_a_broken_case_file(tmp_path, capsys):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "question": "q", "expect_tools_any": ["nope"]}\n')
    assert evals.main(["assistant", "--cases", str(path), "--check"]) == 2
    assert "unknown tool" in capsys.readouterr().out
    path.write_text("not json\n")
    assert evals.main(["assistant", "--cases", str(path)]) == 2


def test_main_with_a_key_runs_cases_and_writes_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NDP_ANTHROPIC_API_KEY", "sk-test-not-real")
    get_settings.cache_clear()
    monkeypatch.setattr(
        assistant, "ask", lambda sid, q: fake_turn("An answer.", ["geocode"], rounds=1)
    )
    out_file = tmp_path / "report.json"
    assert evals.main(["assistant", "--limit", "3", "--json", str(out_file)]) == 0
    text = capsys.readouterr().out
    assert "Assistant eval" in text and "estimated cost" in text
    report = json.loads(out_file.read_text())
    assert report["n_cases"] == 3 and report["model"] == "claude-opus-5-5"
    assert evals.main(["assistant", "--ids", "hazard_tokyo"]) == 0


def test_triage_hook_degrades_when_the_suite_is_missing(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "ndp.evals.triage", raising=False)
    monkeypatch.setitem(sys.modules, "ndp.evals.triage", None)
    monkeypatch.delattr(evals, "triage", raising=False)
    assert evals.main(["triage"]) == 0
    assert "triage suite not available" in capsys.readouterr().out


def test_triage_hook_calls_run_when_present(monkeypatch):
    calls = []
    fake = SimpleNamespace(run=lambda limit=None: calls.append(limit))
    monkeypatch.setitem(sys.modules, "ndp.evals.triage", fake)
    monkeypatch.setattr(evals, "triage", fake, raising=False)
    assert evals.main(["triage", "--limit", "2"]) == 0
    assert calls == [2]
