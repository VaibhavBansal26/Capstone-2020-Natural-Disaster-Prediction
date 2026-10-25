import pytest

from ndp.ml import registry


def test_save_then_load_roundtrip():
    registry.save("unit_test_model", {"answer": 42}, {"metric": 0.5})
    assert registry.is_trained("unit_test_model")
    assert registry.load("unit_test_model") == {"answer": 42}
    meta = registry.metadata("unit_test_model")
    assert meta["metric"] == 0.5 and meta["name"] == "unit_test_model"


def test_missing_model_raises_typed_error():
    with pytest.raises(registry.ModelNotTrained) as exc:
        registry.load("does_not_exist")
    assert exc.value.name == "does_not_exist"
