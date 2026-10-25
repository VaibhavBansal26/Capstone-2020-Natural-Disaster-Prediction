import datetime as dt

from ndp.services.usgs import QuakeEvent, collapse_duplicates

T0 = dt.datetime(2026, 10, 24, 2, 9, tzinfo=dt.UTC)


def quake(id_, mag=7.0, lat=28.0, lon=139.9, seconds=0):
    return QuakeEvent(
        id_,
        mag,
        "Bonin Islands, Japan region",
        lat,
        lon,
        473.0,
        T0 + dt.timedelta(seconds=seconds),
        "",
    )


def test_identical_contributions_from_two_networks_become_one_event():
    kept = collapse_duplicates([quake("pt26298050"), quake("pt26297050")])
    assert len(kept) == 1


def test_core_network_id_is_preferred_over_a_secondary_one():
    kept = collapse_duplicates([quake("pt1", mag=7.0), quake("us6000abcd", mag=6.9, seconds=20)])
    assert [e.id for e in kept] == ["us6000abcd"]


def test_different_earthquakes_are_not_merged():
    far = quake("pt3", lat=-14.0, lon=165.5)  # same time, thousands of km away
    later = quake("pt4", seconds=3600)  # same place an hour later (aftershock)
    kept = collapse_duplicates([quake("pt1"), far, later])
    assert {e.id for e in kept} == {"pt1", "pt3", "pt4"}


def test_a_second_event_with_a_very_different_magnitude_is_kept():
    kept = collapse_duplicates([quake("pt1", mag=7.0), quake("pt2", mag=5.0, seconds=30)])
    assert len(kept) == 2
