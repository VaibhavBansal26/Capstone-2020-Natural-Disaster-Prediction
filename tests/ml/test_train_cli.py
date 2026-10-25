"""The ndp-train CLI: which trainers run, failure handling, and the recorded timings."""

import pytest

from ndp.config import get_settings
from ndp.ml import registry, train
from ndp.ml.features import DataMissing


@pytest.fixture
def fake_trainers(monkeypatch):
    calls = []

    def make(name, result=None, error=None):
        def run():
            calls.append(name)
            if error:
                raise error
            return result or {"r2": 0.5}

        return run

    trainers = {
        "flood": make("flood"),
        "storm": make("storm"),
        "sentiment": make("sentiment"),
        "covid": make("covid"),
    }
    monkeypatch.setattr(train, "TRAINERS", trainers)
    monkeypatch.setattr(train, "_headline", lambda name, result: "ok")
    return trainers, calls, make


def _args(*argv):
    return train._parse(list(argv))


def test_default_runs_everything_enabled(fake_trainers, monkeypatch):
    monkeypatch.setenv("NDP_ENABLE_COVID", "true")
    get_settings.cache_clear()
    assert list(train.select(_args())) == ["flood", "storm", "sentiment", "covid"]


def test_covid_can_be_switched_off(fake_trainers, monkeypatch):
    monkeypatch.setenv("NDP_ENABLE_COVID", "false")
    get_settings.cache_clear()
    assert "covid" not in train.select(_args())
    assert "covid" in train.select(_args("--include-covid"))
    assert "covid" in train.select(_args("covid"))


def test_quick_skips_sentiment_unless_named(fake_trainers):
    assert "sentiment" not in train.select(_args("--quick"))
    assert "sentiment" in train.select(_args("--quick", "sentiment"))
    assert "sentiment" not in train.select(_args("--skip-sentiment"))


def test_named_subset(fake_trainers):
    assert list(train.select(_args("storm", "flood"))) == ["flood", "storm"]


def test_main_records_timings_and_merges_partial_runs(fake_trainers, tmp_path, monkeypatch):
    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NDP_ENABLE_COVID", "false")
    get_settings.cache_clear()
    assert train.main(["flood", "storm"]) == 0
    first = registry.load("training_run")["seconds"]
    assert set(first) == {"flood", "storm"}
    assert train.main(["storm"]) == 0
    merged = registry.load("training_run")["seconds"]
    assert set(merged) == {"flood", "storm"}  # flood kept from the earlier run
    meta = registry.metadata("training_run")
    assert meta["last_run_quick"] is False and meta["last_run_total_seconds"] >= 0


def test_missing_data_is_reported_and_exits_nonzero(fake_trainers, tmp_path, monkeypatch, capsys):
    trainers, calls, make = fake_trainers
    monkeypatch.setitem(trainers, "flood", make("flood", error=DataMissing("x.csv")))
    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NDP_ENABLE_COVID", "false")
    get_settings.cache_clear()
    assert train.main(["flood", "storm"]) == 1
    assert "SKIPPED" in capsys.readouterr().err
    assert calls == ["flood", "storm"]  # one failure does not stop the others


def test_headline_formats_each_result_shape():
    train_headline = train.__dict__["_headline"]  # real function, not the fixture's stub
    assert "skill=+0.10" in train_headline(
        "tsunami", {"r2": 0.3, "skill": 0.1, "baseline_rmse": 0.8}
    )
    assert "r2=0.500" in train_headline("covid", {"r2": 0.5})


def test_if_missing_trains_only_what_is_absent(fake_trainers, tmp_path, monkeypatch):
    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NDP_GENERATED_DIR", str(tmp_path / "plots"))
    monkeypatch.setenv("NDP_ENABLE_COVID", "false")
    get_settings.cache_clear()
    names = ["flood", "storm", "sentiment", "covid"]
    monkeypatch.setattr(train, "ARTIFACTS", {n: [f"{n}_art"] for n in names})
    monkeypatch.setattr(train, "PLOT_DIRS", {n: n for n in names})
    for done in ("flood", "sentiment", "covid"):
        registry.save(f"{done}_art", object(), {})
        (tmp_path / "plots" / done).mkdir(parents=True)
        (tmp_path / "plots" / done / "p.png").write_bytes(b"png")
    assert train.is_missing("storm") and not train.is_missing("flood")
    assert list(train.select(_args("--if-missing"))) == ["storm"]


def test_a_model_without_its_plots_counts_as_missing(tmp_path, monkeypatch):
    # Models on a persistent volume, plots lost with a rebuilt image: retrain to redraw them.
    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NDP_GENERATED_DIR", str(tmp_path / "plots"))
    get_settings.cache_clear()
    monkeypatch.setattr(train, "ARTIFACTS", {"flood": ["flood_art"]})
    monkeypatch.setattr(train, "PLOT_DIRS", {"flood": "flood"})
    registry.save("flood_art", object(), {})
    assert train.is_missing("flood")


def test_if_missing_with_nothing_to_do_exits_cleanly(fake_trainers, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NDP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NDP_ENABLE_COVID", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(train, "is_missing", lambda name: False)
    assert train.main(["--if-missing"]) == 0
    assert "nothing to train" in capsys.readouterr().out


def test_every_trainer_has_artifact_and_plot_mappings():
    """A trainer added without a mapping would never be trained by `--if-missing`."""
    for name in train.TRAINERS:
        assert train.ARTIFACTS.get(name), f"{name} has no ARTIFACTS entry"
        assert name in train.PLOT_DIRS, f"{name} has no PLOT_DIRS entry"
