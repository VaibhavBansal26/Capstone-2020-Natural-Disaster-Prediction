from ndp import data


def test_manifest_has_no_duplicates():
    names = [d.filename for d in data.DATASETS]
    assert len(names) == len(set(names))


def test_every_dataset_has_source_and_consumer():
    for ds in data.DATASETS:
        assert ds.source and ds.used_by


def test_raw_base_is_pinned_to_a_commit():
    assert data.PINNED_COMMIT in data.RAW_BASE
    assert data.RAW_BASE.startswith("https://raw.githubusercontent.com/")
