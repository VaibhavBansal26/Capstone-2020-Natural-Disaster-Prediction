"""The statistics report must work on a bare checkout and be computed, not typed."""

from pathlib import Path

import pytest

from ndp import report
from ndp.config import get_settings


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """Empty data and models directories: what a fresh checkout looks like."""
    monkeypatch.setenv("NDP_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_table_alignment_and_shape():
    md = report.table(["A", "B"], [[1, 2]], align_right=(1,))
    assert md.splitlines() == ["| A | B |", "|---|---:|", "| 1 | 2 |"]


def test_number_and_size_formatting():
    assert report.fmt_num(None) == "n/a"
    assert report.fmt_num(12345) == "12,345"
    assert report.fmt_num(0.123456, 3) == "0.123"
    assert report.fmt_num(0.0000012) == "1.2e-06"
    assert report.fmt_size(1536) == "2 KB" and report.fmt_size(3 * 1024 * 1024) == "3.0 MB"


def test_count_lines_ignores_blank_lines_and_binary(tmp_path):
    text = tmp_path / "a.py"
    text.write_text("x = 1\n\n\ny = 2\n")
    assert report.count_lines(text) == (2, 4)
    binary = tmp_path / "b.bin"
    binary.write_bytes(b"\xff\xfe\x00\x80")
    assert report.count_lines(binary) == (0, 0)


def test_repo_stats_counts_areas_and_test_functions(tmp_path):
    (tmp_path / "src/ndp/ml").mkdir(parents=True)
    (tmp_path / "src/ndp/ml/a.py").write_text("a = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").write_text(
        "def test_one():\n    pass\n\nasync def test_two():\n    pass\n"
    )
    (tmp_path / "tests/y.test.mjs").write_text('test("a", () => {});\ntest("b", () => {});\n')
    stats = report.repo_stats(tmp_path)
    by_area = {a["area"]: a for a in stats["areas"]}
    assert by_area["Application: ml (training and prediction)"]["lines"] == 1
    assert stats["py_tests"] == 2 and stats["js_tests"] == 2


def test_year_formatting_for_bce_dates():
    assert report._year(-2000) == "2000 BC" and report._year(2017) == "2017"


def test_report_on_a_bare_checkout_says_what_is_missing(bare):
    text = report.build(root=Path(__file__).resolve().parents[2])
    assert text.startswith("# Statistics")
    assert "No datasets found" in text
    assert "not trained" in text  # model table rows
    assert "## Repository" in text and "## Front-end payload" in text


def test_report_reads_metrics_from_saved_metadata(bare):
    from ndp.ml import registry

    registry.save(
        "hurricane_wind",
        object(),
        {
            "rmse": 24.5,
            "r2": 0.04,
            "baseline_rmse": 25.9,
            "skill": 0.054,
            "n_rows": 74904,
            "split": "chronological",
        },
    )
    rows = {r[0]: r for r in report.model_rows()}
    row = rows["Hurricane wind"]
    assert row[1] == "74,904" and "RMSE 24.5" in row[2] and row[4] == "+0.05"


def test_main_writes_a_file(bare, tmp_path, capsys):
    target = tmp_path / "out" / "stats.md"
    assert report.main(["--write", str(target)]) == 0
    assert target.read_text().startswith("# Statistics")
    assert "wrote" in capsys.readouterr().out
