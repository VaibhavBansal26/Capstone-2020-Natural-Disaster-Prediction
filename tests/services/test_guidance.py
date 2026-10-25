"""Guidance corpus: extraction, chunking, TF-IDF search, fetching (mocked), committed corpus."""

import httpx
import pytest

from ndp import guidance
from ndp.config import get_settings

PAGE = """
<html><head><title>x</title><script>var a = 1;</script><style>p{}</style></head>
<body>
<header><nav><a href="/">Home</a><a href="/about">About</a></nav></header>
<div class="cms-content">
  <div class="breadcrumb">Weather.gov &gt; Safety</div>
  <p>During a flood, water levels can change quickly. Move to higher ground at once.
     If you are told to evacuate, do so immediately.</p>
  <ul><li><strong>Stay informed:</strong> listen to NOAA Weather Radio.</li>
      <li>Never drive into flooded roads.</li></ul>
  <h2>Resources</h2><ul><li>Some link title</li></ul>
</div>
<footer>Contact us. Privacy policy.</footer>
</body></html>
"""


def write_doc(directory, slug, title, hazard, body, url=None):
    (directory / f"{slug}.md").write_text(
        guidance._front_matter(
            {
                "title": title,
                "source_url": url or f"https://example.gov/{slug}",
                "hazard": hazard,
                "retrieved": "2026-01-01",
                "license_note": "test",
            }
        )
        + "\n"
        + body
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def corpus(tmp_path):
    write_doc(
        tmp_path,
        "quake",
        "Quake safety",
        "earthquake",
        "## During\n\nDrop, cover and hold on under a sturdy table until the shaking stops. "
        "Stay away from windows and heavy furniture during an earthquake.",
    )
    write_doc(
        tmp_path,
        "flood",
        "Flood safety",
        "flood",
        "## During\n\nTurn around, do not drown. Never drive through flood water; move to "
        "higher ground and listen to weather radio.",
    )
    write_doc(
        tmp_path,
        "kit",
        "Emergency kit",
        "general",
        "## Supplies\n\n- Water, one gallon per person per day\n- Food for three days\n"
        "- First aid kit and medicines\n- Flashlight and batteries",
    )
    guidance.reset_cache()
    yield tmp_path
    guidance.reset_cache()


# ---- extraction -------------------------------------------------------------------------


def test_extract_text_drops_chrome_and_keeps_article():
    text = guidance.extract_text(PAGE, root="cms-content")
    assert "Move to higher ground" in text
    assert "- Stay informed: listen to NOAA Weather Radio." in text
    assert "Home" not in text and "Privacy" not in text and "Weather.gov" not in text
    assert "var a" not in text and "p{}" not in text
    assert "Some link title" not in text  # "Resources" link-list sections are removed


def test_extract_text_falls_back_to_main_then_body():
    with_main = "<body><p>outside</p><main><p>inside the main element text</p></main></body>"
    assert guidance.extract_text(with_main) == "inside the main element text"
    assert "outside" in guidance.extract_text("<body><p>outside</p></body>")


def test_headings_become_markdown_and_bullets_stay_adjacent():
    text = guidance.extract_text(
        "<main><h2>Title</h2><ul><li>first item</li><li>second item</li></ul></main>"
    )
    assert text == "## Title\n\n- first item\n- second item"


def test_junk_sentences_are_dropped():
    html = "<main><p>Real advice here. Please share your story with us.</p></main>"
    assert guidance.extract_text(html) == "Real advice here."


# ---- files, chunking, search -------------------------------------------------------------


def test_front_matter_round_trip():
    meta, body = guidance.parse_document(
        guidance._front_matter({"title": 'A "quoted" title: yes', "hazard": "flood"}) + "\nBody"
    )
    assert meta == {"title": 'A "quoted" title: yes', "hazard": "flood"}
    assert body == "Body"


def test_chunk_text_respects_size_and_keeps_heading_context():
    sentence = "Move to higher ground right away and stay there until officials say it is safe. "
    body = "## Flood\n\n" + sentence * 30 + "\n\n## Other\n\nShort paragraph."
    chunks = guidance.chunk_text(body, size=400)
    assert len(chunks) > 3
    assert all(len(c) <= 400 + 40 for c in chunks)  # a chunk may carry its "## heading" lead
    assert all(c.startswith("## ") for c in chunks)  # continuation chunks keep the heading
    assert any("Short paragraph." in c for c in chunks)
    assert not any(
        c.strip().startswith("## Other") and len(c.strip().splitlines()) == 1 for c in chunks
    )


def test_chunk_text_splits_long_bullet_lists_by_line():
    body = "\n".join(f"- item number {i} with some words to fill the line" for i in range(60))
    chunks = guidance.chunk_text(body, size=300)
    assert len(chunks) > 5
    assert all(line.startswith("- ") for c in chunks for line in c.splitlines())


def test_search_ranks_the_right_document_first(corpus):
    top = guidance.search("what to do during an earthquake", directory=corpus)[0]
    assert top.hazard == "earthquake" and top.source_url == "https://example.gov/quake"
    assert top.score > 0 and "Drop, cover" in top.text
    flood = guidance.search("can I drive through flood water", directory=corpus)[0]
    assert flood.title == "Flood safety"


def test_search_hazard_filter_includes_general_pages(corpus):
    hits = guidance.search("water food supplies during emergency", hazard="flood", directory=corpus)
    assert {h.hazard for h in hits} <= {"flood", "general"}
    assert any(h.hazard == "general" for h in hits)


def test_search_is_deterministic_and_handles_odd_input(corpus):
    a = guidance.search("earthquake", k=3, directory=corpus)
    guidance.reset_cache()
    b = guidance.search("earthquake", k=3, directory=corpus)
    assert a == b
    assert guidance.search("", directory=corpus) == []
    assert guidance.search("zzzzqqqq xylophone", directory=corpus) == []
    assert guidance.search("earthquake", k=0, directory=corpus) == []


def test_search_on_missing_directory_is_empty(tmp_path):
    guidance.reset_cache()
    assert guidance.search("earthquake", directory=tmp_path / "nope") == []


def test_at_most_two_chunks_per_page(tmp_path):
    write_doc(
        tmp_path,
        "big",
        "Big page",
        "flood",
        "\n\n".join(f"flood water advice number {i} " * 12 for i in range(30)),
    )
    guidance.reset_cache()
    hits = guidance.search("flood water advice", k=10, directory=tmp_path)
    assert len(hits) == guidance.MAX_PER_SOURCE
    guidance.reset_cache()


# ---- fetching (mocked network) ------------------------------------------------------------

SRC = guidance.Source(
    "t-one", "T one", "https://example.gov/one", "flood", "note", root="cms-content"
)
SRC2 = guidance.Source("t-two", "T two", "https://example.gov/two", "flood", "note")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_all_writes_files_with_front_matter(tmp_path):
    long_page = PAGE.replace("Move to higher ground", "Move to higher ground " + "x " * 300)
    client = _client(lambda request: httpx.Response(200, text=long_page))
    results = guidance.fetch_all(tmp_path, [SRC], client=client, today="2026-10-25")
    assert [r.ok for r in results] == [True]
    meta, body = guidance.parse_document((tmp_path / "t-one.md").read_text())
    assert meta["source_url"] == SRC.url and meta["retrieved"] == "2026-10-25"
    assert meta["hazard"] == "flood" and meta["license_note"] == "note" and meta["title"] == "T one"
    assert "higher ground" in body


def test_failed_fetch_keeps_the_previous_file(tmp_path):
    (tmp_path / "t-two.md").write_text("old content")

    def handler(request):
        return httpx.Response(403, text="Access Denied")

    results = guidance.fetch_all(tmp_path, [SRC2], client=_client(handler))
    assert not results[0].ok and "403" in results[0].message
    assert (tmp_path / "t-two.md").read_text() == "old content"


def test_too_short_extraction_is_a_failure(tmp_path):
    client = _client(lambda request: httpx.Response(200, text="<main><p>tiny</p></main>"))
    results = guidance.fetch_all(tmp_path, [SRC2], client=client)
    assert not results[0].ok and "too little" in results[0].message
    assert not (tmp_path / "t-two.md").exists()


def test_main_returns_nonzero_when_a_page_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        guidance,
        "fetch_all",
        lambda dest=None: [guidance.FetchResult("a", "u", False, 0, "HTTP 500")],
    )
    assert guidance.main(["--dest", str(tmp_path)]) == 1
    assert "0/1 pages saved" in capsys.readouterr().out


# ---- the committed corpus ------------------------------------------------------------------


def test_committed_corpus_loads_and_is_well_formed():
    docs = guidance.load_documents(get_settings().guidance_dir)
    assert len(docs) >= 20
    for doc in docs:
        assert doc.source_url.startswith("https://") and doc.retrieved and doc.license_note
        assert doc.hazard in guidance.HAZARDS, doc.slug
        assert "Access Denied" not in doc.body and len(doc.body) >= guidance.MIN_PAGE_CHARS - 100
    assert {d.hazard for d in docs} >= {"earthquake", "flood", "tsunami", "heat", "wildfire"}
    hosts = {d.source_url.split("/")[2] for d in docs}
    assert hosts <= {"www.cdc.gov", "www.weather.gov", "www.usgs.gov"}


def test_committed_corpus_answers_what_to_do_during_an_earthquake():
    guidance.reset_cache()
    hits = guidance.search("what to do during an earthquake")
    assert hits and hits[0].hazard == "earthquake"
    assert "earthquake" in hits[0].title.lower()
    assert all(h.source_url.startswith("https://") for h in hits)
    assert "cover" in " ".join(h.text.lower() for h in hits)


def test_committed_corpus_flood_driving_cites_nws():
    hits = guidance.search("is it safe to drive through a flooded road turn around don't drown")
    assert any("weather.gov" in h.source_url for h in hits[:2])
