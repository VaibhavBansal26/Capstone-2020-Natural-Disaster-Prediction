"""Preparedness guidance corpus: fetcher (``ndp-fetch-guidance``) and TF-IDF search.

Why this exists: the assistant must answer "what should I do during an earthquake?" from
vetted public sources and cite them, not from the model's memory. The corpus is a handful of
plain-text pages from US federal agencies (NOAA / National Weather Service, USGS, CDC) saved
under ``data/guidance/<slug>.md`` with a small front matter block::

    ---
    title: "During a Flood"
    source_url: "https://www.weather.gov/safety/flood-during"
    hazard: "flood"
    retrieved: "2026-10-25"
    license_note: "..."
    ---

The files are committed to the repository so the app works offline. Run
``uv run ndp-fetch-guidance`` (or ``uv run python -m ndp.guidance``) to refresh them.

Licensing. Only sources whose reuse terms are stated on the agency's own site are included, and
every file records them in ``license_note``. US federal government works are generally in the
public domain, but individual pages can embed third-party material, so only the article text is
stored (no images) and the note says so. NWS additionally asks that its content is not modified
and then presented as official government material: the assistant therefore cites pages as
"excerpts" and never as official instructions.

Ready.gov / FEMA pages (the obvious first choice) answer scripted clients with an Akamai
"Access Denied" page. This module does not try to get around that: those pages are left out.

Search is classic TF-IDF over ~800-character chunks (scikit-learn, word 1-2-grams, sublinear tf),
built lazily on first use and cached. Results are deterministic: ties break on document order.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import httpx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ndp import __version__
from ndp.config import get_settings

USER_AGENT = (
    f"natural-disaster-prediction/{__version__} "
    "(+https://github.com/VaibhavBansal26; preparedness guidance fetcher)"
)

CHUNK_CHARS = 800
MAX_PER_SOURCE = 2
MIN_PAGE_CHARS = 500  # a page with less article text than this is a failed extraction

HAZARDS = (
    "earthquake",
    "flood",
    "tsunami",
    "hurricane",
    "heat",
    "wildfire",
    "general",
)

# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------

NWS_NOTE = (
    "National Weather Service web pages are in the public domain unless noted otherwise "
    "(https://www.weather.gov/disclaimer); reuse must not imply NOAA/NWS endorsement or present "
    "modified content as official government material. Text excerpt only."
)
USGS_NOTE = (
    "USGS-authored information is in the U.S. public domain; some images on USGS sites are "
    "copyrighted, so only text is stored "
    "(https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)."
)
CDC_NOTE = (
    "Most CDC.gov content is not subject to copyright and may be freely reused; contractor or "
    "third-party items can be exceptions, so only article text is stored "
    "(https://www.cdc.gov/other/agencymaterials.html)."
)


@dataclass(frozen=True)
class Source:
    slug: str
    title: str
    url: str
    hazard: str
    license_note: str
    root: str | None = None  # tag name or CSS class of the article container, if not <main>


def _nws(slug: str, title: str, path: str, hazard: str) -> Source:
    return Source(
        slug=f"nws-{slug}",
        title=title,
        url=f"https://www.weather.gov/safety/{path}",
        hazard=hazard,
        license_note=NWS_NOTE,
        root="cms-content",
    )


def _cdc(slug: str, title: str, path: str, hazard: str) -> Source:
    return Source(
        slug=f"cdc-{slug}",
        title=title,
        url=f"https://www.cdc.gov/{path}",
        hazard=hazard,
        license_note=CDC_NOTE,
    )


# Order is the fetch order and the tie-break order in search results.
SOURCES: tuple[Source, ...] = (
    Source(
        slug="usgs-earthquake-during",
        title="USGS: What should I do DURING an earthquake?",
        url="https://www.usgs.gov/faqs/what-should-i-do-during-earthquake",
        hazard="earthquake",
        license_note=USGS_NOTE,
    ),
    _cdc(
        "earthquake-during",
        "CDC: Safety Guidelines: During an Earthquake",
        "earthquakes/safety/stay-safe-during-an-earthquake.html",
        "earthquake",
    ),
    _cdc(
        "earthquake-after",
        "CDC: Safety Guidelines: After an Earthquake",
        "earthquakes/safety/stay-safe-after-an-earthquake.html",
        "earthquake",
    ),
    _cdc(
        "earthquake-prepare",
        "CDC: Preparing for Earthquakes",
        "earthquakes/safety/index.html",
        "earthquake",
    ),
    _nws("flood-before", "NWS: Before a Flood", "flood-before", "flood"),
    _nws("flood-during", "NWS: During a Flood", "flood-during", "flood"),
    _nws("flood-after", "NWS: After a Flood", "flood-after", "flood"),
    _nws(
        "turn-around-dont-drown",
        "NWS: Turn Around, Don't Drown",
        "flood-turn-around-dont-drown",
        "flood",
    ),
    _cdc("flood-prepare", "CDC: Preparing for Floods", "floods/safety/index.html", "flood"),
    _nws("tsunami-before", "NWS: Before a Tsunami", "tsunami-before", "tsunami"),
    _nws("tsunami-during", "NWS: During a Tsunami", "tsunami-during", "tsunami"),
    _nws("tsunami-after", "NWS: After a Tsunami", "tsunami-after", "tsunami"),
    _cdc("tsunami-safety", "CDC: Tsunamis and Your Safety", "tsunamis/about/index.html", "tsunami"),
    _nws("hurricane", "NWS: Hurricane Safety Tips and Resources", "hurricane", "hurricane"),
    _cdc(
        "hurricane-prepare",
        "CDC: Preparing for Hurricanes or Other Tropical Storms",
        "hurricanes/safety/index.html",
        "hurricane",
    ),
    _nws("heat-during", "NWS: During a Heat Wave", "heat-during", "heat"),
    _nws("heat-vulnerable", "NWS: Heat Safety Tips and Resources", "heat", "heat"),
    _cdc("heat-health", "CDC: About Heat and Your Health", "heat-health/about/index.html", "heat"),
    _cdc(
        "heat-risk-factors",
        "CDC: People at Increased Risk for Heat-Related Illness",
        "heat-health/risk-factors/index.html",
        "heat",
    ),
    _cdc(
        "wildfire-prepare",
        "CDC: Preparing for Wildfires",
        "wildfires/safety/index.html",
        "wildfire",
    ),
    _cdc(
        "wildfire-smoke", "CDC: Wildfires and Your Safety", "wildfires/about/index.html", "wildfire"
    ),
    _cdc(
        "emergency-supplies",
        "CDC: Prepare Your Health: Personal Needs (emergency supplies)",
        "prepare-your-health/take-action/personal-needs.html",
        "general",
    ),
    _cdc(
        "power-outage",
        "CDC: Prepare Your Health: Power Sources (power outages)",
        "prepare-your-health/take-action/power-sources.html",
        "general",
    ),
    _cdc(
        "family-communication-plan",
        "CDC: Prepare Your Health: Stay Connected (family communication plan)",
        "prepare-your-health/plan-ahead/stay-connected.html",
        "general",
    ),
    _cdc(
        "after-a-disaster",
        "CDC: What to Do to Prevent Getting Hurt or Sick After a Disaster",
        "natural-disasters/response/index.html",
        "general",
    ),
)

# --------------------------------------------------------------------------------------------
# HTML to text
# --------------------------------------------------------------------------------------------

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "svg",
        "iframe",
        "button",
        "select",
        "template",
        "figure",
        "picture",
        "head",
    }
)
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
_HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "#### ", "h6": "#### "}
_BLOCKS = frozenset({"p", "div", "section", "article", "ul", "ol", "table", "tr", "blockquote"})
_BLOCK_END = _BLOCKS | {"li", "dt", "dd", "main", "td", "th"}
# Class tokens that mark page chrome even when it sits inside the article container.
_SKIP_CLASS = re.compile(
    r"^(breadcrumbs?|share\S*|social\S*|print|related\S*|sidebar|toc|skip\S*|pagination|"
    r"feedback\S*|usa-banner\S*|noindex|hide|hidden|sr-only|visually-hidden|"
    r"cdc-page-title-bar\S*|dfe-section__nav\S*|last-?reviewed\S*|page-?nav\S*)$",
    re.I,
)
_BOILERPLATE_LINE = re.compile(
    r"^(print|share|email|facebook|twitter|linkedin|pinterest|reddit|syndicate|"
    r"on this page|related pages?|more information|source:|last reviewed|"
    r"last updated|page last|skip to|español|get email updates|"
    r"was this page helpful|contact us|print minus)\b",
    re.I,
)


_JUNK_LINE = re.compile(
    r"^(\*\*)?get tsunami alerts|^[A-Za-z ]{3,30} resources$|\bpsa by\b|^\+?\s*(expand|collapse)",
    re.I,
)


_JUNK_SENTENCE = re.compile(
    r"share your story|have been a victim of|permission to use your story|"
    r"sharing this information may help|^read stories from|^watch the",
    re.I,
)


def _drop_junk_sentences(block: str) -> str:
    if block.startswith(("#", "- ")):
        return block
    kept = [x for x in _SENTENCE.split(block) if not _JUNK_SENTENCE.search(x)]
    return " ".join(kept)


class _Extractor(HTMLParser):
    """Collects headings, paragraphs and list items from the article container.

    Chrome (navigation, headers, footers, forms, share widgets) is dropped. The container is the
    first element matching ``root`` (a tag or a class), else ``<main>``, else ``<article>``, else
    the whole ``<body>``.
    """

    def __init__(self, root: str | None) -> None:
        super().__init__(convert_charrefs=True)
        self.root = root
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, is_skip, is_root)
        self.skip_depth = 0
        self.root_depth = 0
        self.fallback_root_depth = 0  # <main>/<article> when no explicit root
        self.root_seen = False
        self.body_blocks: list[str] = []
        self.root_blocks: list[str] = []
        self.main_blocks: list[str] = []
        self._buf: list[str] = []
        self._prefix = ""
        self._li_depth = 0

    # -- block handling ------------------------------------------------------------------
    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        prefix, self._prefix = self._prefix, ""
        if not text:
            return
        block = prefix + text
        self.body_blocks.append(block)
        if self.root_depth:
            self.root_blocks.append(block)
        if self.fallback_root_depth:
            self.main_blocks.append(block)

    def _matches_root(self, tag: str, attrs: dict[str, str]) -> bool:
        if not self.root:
            return False
        classes = (attrs.get("class") or "").split()
        return tag == self.root or self.root in classes or attrs.get("id") == self.root

    def handle_starttag(self, tag: str, attrs_list) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag in _VOID_TAGS:
            if tag in ("br", "hr") and not self.skip_depth:
                self._buf.append(" ")
            return
        is_skip = (
            tag in _SKIP_TAGS
            or any(_SKIP_CLASS.match(token) for token in attrs.get("class", "").split())
            or attrs.get("aria-hidden") == "true"
            or "hidden" in attrs
        )
        is_root = self._matches_root(tag, attrs) and not is_skip
        self.stack.append((tag, is_skip, is_root))
        if is_skip:
            self.skip_depth += 1
        if self.skip_depth:
            return
        if is_root:
            self._flush()
            self.root_depth += 1
            self.root_seen = True
        if not self.root and tag in ("main", "article"):
            self._flush()
            self.fallback_root_depth += 1
        if tag in _HEADINGS:
            self._flush()
            self._prefix = _HEADINGS[tag]
        elif tag == "li":
            self._flush()
            self._prefix = "- "
            self._li_depth += 1
        elif tag in _BLOCKS or tag == "dt" or tag == "dd":
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or not self.stack:
            return
        # Pop to the matching open tag (the HTML in the wild is not always balanced).
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                break
        else:
            return
        while len(self.stack) > i:
            t, is_skip, is_root = self.stack.pop()
            if not self.skip_depth and (t in _BLOCK_END or t in _HEADINGS):
                self._flush()
            if is_skip:
                self.skip_depth -= 1
            elif not self.skip_depth:
                if is_root:
                    self.root_depth -= 1
                if not self.root and t in ("main", "article"):
                    self.fallback_root_depth -= 1
                if t == "li":
                    self._li_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self._buf.append(data)

    def result(self) -> list[str]:
        self._flush()
        if self.root and self.root_blocks:
            return self.root_blocks
        if self.main_blocks:
            return self.main_blocks
        return self.body_blocks


_DROP_SECTIONS = frozenset(
    {
        "resources",
        "references",
        "related pages",
        "related resources",
        "learn more",
        "sources",
        "more information",
        "for more information",
        "additional resources",
    }
)
_DATE_ONLY = re.compile(
    r"^(updated|last updated|published|reviewed)?:?\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}, \d{4}$",
    re.I,
)


def _drop_link_sections(blocks: list[str]) -> list[str]:
    """Remove "Resources"-style sections (bare lists of link titles) up to the next heading of
    the same or a higher level."""
    out: list[str] = []
    skip_level = 0
    for block in blocks:
        if block.startswith("#"):
            level = len(block) - len(block.lstrip("#"))
            title = block.lstrip("# ").strip().lower().rstrip(":")
            if skip_level and level <= skip_level:
                skip_level = 0
            if title in _DROP_SECTIONS:
                skip_level = level
                continue
        if not skip_level:
            out.append(block)
    return out


def extract_text(markup: str, root: str | None = None) -> str:
    """Article text of an HTML page as light Markdown (headings, paragraphs, bullets)."""
    parser = _Extractor(root)
    parser.feed(markup)
    parser.close()
    blocks: list[str] = []
    seen: set[str] = set()
    for raw in parser.result():
        block = _drop_junk_sentences(html.unescape(raw).replace("\u00a0", " ").strip())
        if block.startswith("#"):  # NWS accordions render as "### +Heading"
            block = re.sub(r"^(#+ )\+\s*", r"\1", block)
        bare = block.lstrip("#- ").strip()
        if (
            len(bare) < 2
            or _BOILERPLATE_LINE.match(bare)
            or _JUNK_LINE.search(bare)
            or _DATE_ONLY.match(bare)
        ):
            continue
        if block in seen and not block.startswith("#"):
            continue  # menus repeated for mobile and desktop
        seen.add(block)
        blocks.append(block)
    blocks = _drop_link_sections(blocks)
    out = ""
    for i, block in enumerate(blocks):
        if i:
            out += "\n" if block.startswith("- ") and blocks[i - 1].startswith("- ") else "\n\n"
        out += block
    return out.strip()


def page_title(markup: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.S | re.I)
    return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip() if match else ""


# --------------------------------------------------------------------------------------------
# Front matter and files
# --------------------------------------------------------------------------------------------


def _front_matter(fields: dict[str, str]) -> str:
    lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in fields.items()]
    return "---\n" + "\n".join(lines) + "\n---\n"


def parse_document(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` front matter from the body. Values may be JSON strings or bare text."""
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            header, body = text[3:end], text[end + 4 :]
            for line in header.strip().splitlines():
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                value = value.strip()
                if value.startswith('"'):
                    try:
                        value = json.loads(value)
                    except ValueError:
                        value = value.strip('"')
                meta[key.strip()] = value
    return meta, body.strip()


def render_markdown(source: Source, text: str, retrieved: str) -> str:
    fields = {
        "title": source.title,
        "source_url": source.url,
        "hazard": source.hazard,
        "retrieved": retrieved,
        "license_note": source.license_note,
    }
    return _front_matter(fields) + "\n" + text.strip() + "\n"


# --------------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchResult:
    slug: str
    url: str
    ok: bool
    chars: int
    message: str


def fetch_source(
    client: httpx.Client, source: Source, destination: Path, today: str
) -> FetchResult:
    try:
        response = client.get(source.url)
    except httpx.HTTPError as exc:
        return FetchResult(source.slug, source.url, False, 0, f"request failed: {exc}")
    if response.status_code != 200:
        return FetchResult(source.slug, source.url, False, 0, f"HTTP {response.status_code}")
    text = extract_text(response.text, source.root)
    if len(text) < MIN_PAGE_CHARS:
        return FetchResult(
            source.slug, source.url, False, len(text), "too little article text extracted"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{source.slug}.md").write_text(
        render_markdown(source, text, today), encoding="utf-8"
    )
    return FetchResult(source.slug, source.url, True, len(text), "saved")


def fetch_all(
    destination: Path | None = None,
    sources: Sequence[Source] | None = None,
    client: httpx.Client | None = None,
    today: str | None = None,
) -> list[FetchResult]:
    """Download every source into ``destination`` (default ``settings.guidance_dir``).

    A failed page keeps whatever file was committed before, so a flaky network never empties
    the corpus. Returns one result per source.
    """
    destination = destination or get_settings().guidance_dir
    sources = SOURCES if sources is None else sources
    today = today or date.today().isoformat()
    own_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=20.0, follow_redirects=True
    )
    try:
        results = [fetch_source(client, s, destination, today) for s in sources]
    finally:
        if own_client:
            client.close()
    reset_cache()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ndp-fetch-guidance",
        description="Download the public-domain preparedness guidance corpus into data/guidance/.",
    )
    parser.add_argument("--dest", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    results = fetch_all(args.dest)
    for r in results:
        mark = "ok  " if r.ok else "FAIL"
        print(f"{mark} {r.slug:<28} {r.chars:>6} chars  {r.message}  {r.url}")
    failed = [r for r in results if not r.ok]
    print(f"{len(results) - len(failed)}/{len(results)} pages saved")
    return 1 if failed else 0


# --------------------------------------------------------------------------------------------
# Chunking and search
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    text: str
    title: str
    source_url: str
    hazard: str
    score: float = 0.0


@dataclass(frozen=True)
class Document:
    slug: str
    title: str
    source_url: str
    hazard: str
    retrieved: str
    license_note: str
    body: str


def load_documents(directory: Path) -> list[Document]:
    """All ``*.md`` files in a directory, sorted by file name (deterministic)."""
    docs: list[Document] = []
    if not directory.is_dir():
        return docs
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_document(path.read_text(encoding="utf-8"))
        if not body or not meta.get("source_url"):
            continue
        docs.append(
            Document(
                slug=path.stem,
                title=meta.get("title") or path.stem.replace("-", " ").title(),
                source_url=meta["source_url"],
                hazard=(meta.get("hazard") or "general").lower(),
                retrieved=meta.get("retrieved", ""),
                license_note=meta.get("license_note", ""),
                body=body,
            )
        )
    return docs


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(body: str, size: int = CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into chunks of about ``size`` characters.

    Paragraphs are never merged across a heading boundary unless they fit together, a heading is
    never left alone at the end of a chunk, and each continuation chunk starts with the nearest
    heading so it keeps its context.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[tuple[str, str]] = []  # (text, heading in force)
    heading = ""
    for para in paragraphs:
        if para.startswith("#"):
            heading = para.lstrip("# ").strip()
            pieces.append((para, heading))
            continue
        if len(para) <= size:
            pieces.append((para, heading))
            continue
        # A long paragraph or bullet list is split on lines, then on sentences.
        units: list[str] = []
        for line in para.split("\n"):
            units.extend(_SENTENCE.split(line) if len(line) > size else [line])
        current = ""
        for unit in units:
            if current and len(current) + len(unit) + 1 > size:
                pieces.append((current, heading))
                current = unit
            else:
                current = f"{current}\n{unit}" if current else unit
        if current:
            pieces.append((current, heading))

    chunks: list[str] = []
    current = ""
    current_heading = ""
    for text, piece_heading in pieces:
        is_heading = text.startswith("#")
        if current and len(current) + len(text) + 2 > size:
            chunks.append(current)
            current = ""
        if not current:
            lead = f"## {piece_heading}\n\n" if piece_heading and not is_heading else ""
            current = lead + text
        else:
            current += "\n\n" + text
        current_heading = piece_heading
    del current_heading
    if current:
        chunks.append(current)
    # Drop chunks that are only a heading line.
    return [c for c in chunks if len(re.sub(r"^#+ .*$", "", c, flags=re.M).strip()) > 0]


# sklearn's English stop words minus the words that distinguish "during" from "after" a hazard.
_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - {"during", "before", "after"}
_TOKEN = re.compile(r"[a-z][a-z0-9']+")


def _stem(word: str) -> str:
    word = word.replace("'", "")
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def analyze(text: str) -> list[str]:
    """Lowercase, drop stop words, strip plurals, add adjacent-word bigrams. Deterministic."""
    words = [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP_WORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


class Index:
    """TF-IDF index over the chunks of a set of documents."""

    def __init__(self, documents: Iterable[Document]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.documents = list(documents)
        self.chunks: list[Chunk] = []
        self._order: list[int] = []
        corpus: list[str] = []
        for doc_index, doc in enumerate(self.documents):
            for text in chunk_text(doc.body):
                self.chunks.append(Chunk(text, doc.title, doc.source_url, doc.hazard))
                self._order.append(doc_index)
                corpus.append(f"{doc.title}. {doc.hazard}. {text}")
        self._vectorizer = None
        self._matrix = None
        if corpus:
            self._vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform(corpus)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, hazard: str | None = None, k: int = 4) -> list[Chunk]:
        query = (query or "").strip()
        if not query or self._vectorizer is None or k <= 0:
            return []
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        wanted = (hazard or "").strip().lower() or None
        ranked = sorted(
            (
                (-float(score), position)
                for position, score in enumerate(scores)
                if score > 0 and (not wanted or self.chunks[position].hazard in (wanted, "general"))
            )
        )
        out: list[Chunk] = []
        per_source: dict[str, int] = {}
        for neg_score, position in ranked:
            chunk = self.chunks[position]
            if per_source.get(chunk.source_url, 0) >= MAX_PER_SOURCE:
                continue  # keep the answer diverse: at most two chunks from one page
            per_source[chunk.source_url] = per_source.get(chunk.source_url, 0) + 1
            out.append(
                Chunk(chunk.text, chunk.title, chunk.source_url, chunk.hazard, round(-neg_score, 4))
            )
            if len(out) >= k:
                break
        return out


_lock = threading.Lock()
_indexes: dict[str, Index] = {}


def reset_cache() -> None:
    with _lock:
        _indexes.clear()


def get_index(directory: Path | None = None) -> Index:
    directory = Path(directory or get_settings().guidance_dir)
    key = str(directory.resolve())
    with _lock:
        if key not in _indexes:
            _indexes[key] = Index(load_documents(directory))
        return _indexes[key]


def search(
    query: str, hazard: str | None = None, k: int = 4, directory: Path | None = None
) -> list[Chunk]:
    """Best guidance chunks for a question, each with its source URL.

    ``hazard`` (earthquake, flood, ...) restricts results to that hazard plus ``general`` pages.
    """
    return get_index(directory).search(query, hazard=hazard, k=min(k, 10))


if __name__ == "__main__":
    sys.exit(main())
