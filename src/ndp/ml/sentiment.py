"""SentiWordNet sentiment scoring of scraped forum posts about past disasters.

The scoring rules are a faithful port of the 2020 code: adverbs accumulate a
weighted score (RBS 1.5 / RBR 1.2 / RB 1.0) that then scales the next adjective
(JJS 1.5 / JJR 1.2 / JJ 1.0) or verb by ``(0.4 * adverb_score + 1)``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from ndp.ml import plots, registry
from ndp.ml.features import load_csv

SCORES = "sentiment_scores"


@dataclass(frozen=True)
class Disaster:
    slug: str
    name: str
    csv: str
    monetary_impact_busd: float
    image: str


DISASTERS: list[Disaster] = [
    Disaster(
        "japan-2016-earthquake",
        "Japan 2016 earthquake",
        "Japan2016_Earthquake.csv",
        1.21,
        "senimg/japan.jpg",
    ),
    Disaster(
        "japan-2011-tsunami", "Japan 2011 tsunami", "japan_2011_tsunami.csv", 360, "senimg/live.jpg"
    ),
    Disaster(
        "chennai-2015-flood",
        "Chennai 2015 flood",
        "chennai_flood_2015.csv",
        42.04,
        "senimg/chennai.jpg",
    ),
    Disaster(
        "kerala-2018-flood",
        "Kerala 2018 flood",
        "kerala_flood_2018.csv",
        112.72,
        "senimg/Kerala.jpeg",
    ),
    Disaster(
        "nepal-2015-earthquake",
        "Nepal 2015 earthquake",
        "Nepal_Earthquake_2015.csv",
        0.013,
        "senimg/indonesia.jpg",
    ),
    Disaster(
        "indonesia-2018-tsunami",
        "Indonesia (Palu) 2018 tsunami",
        "paul_Indonesia_tsunami_2018.csv",
        0.0131,
        "senimg/indonesia.jpg",
    ),
    Disaster(
        "phailin-2013-cyclone",
        "Cyclone Phailin 2013",
        "phailin_cyclone_2013.csv",
        0.655,
        "senimg/phailin.jpg",
    ),
]

# (nltk.data.find path, nltk.download package id)
_CORPORA = (
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/sentiwordnet", "sentiwordnet"),
    ("corpora/wordnet", "wordnet"),
)

# Legacy character sets, verbatim. ``_SP`` is spliced into a regex character class
# (note the implicit ``*-@`` range) and ``_SP_CHAR`` is added to the stopword set.
_SP_CHAR = [
    ".",
    "-",
    "*",
    "@",
    "$",
    "%",
    "&",
    "#",
    ",",
    '"',
    "'",
    "?",
    "!",
    ":",
    ";",
    ")",
    "]",
    "[",
    "(",
    "}",
    "{",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    " ",
    "  ",
    "   ",
    "~",
    "`",
    "|",
    "/",
]
_SP = [
    ".",
    "*",
    "-",
    "@",
    "$",
    "%",
    "&",
    "#",
    ",",
    '"',
    "'",
    "?",
    "!",
    ":",
    ";",
    ")",
    "(",
    "}",
    "{",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "~",
    "`",
    "|",
    "/",
]
_DELIMITER_RE = re.compile("[" + "".join(_SP) + "+]")

_ADJ = {"JJ", "JJR", "JJS"}
_ADV = {"RB", "RBR", "RBS"}
_VERB = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

_ADV_WEIGHT = {"RBS": 1.5, "RBR": 1.2, "RB": 1.0}
_ADJ_WEIGHT = {"JJS": 1.5, "JJR": 1.2, "JJ": 1.0}
_ADV_SCALE = 0.4

_lock = threading.Lock()
_ready = False
_stopwords: frozenset[str] = frozenset()


def ensure_corpora() -> None:
    """Download the NLTK resources we need if they are missing (quiet, idempotent)."""
    global _ready, _stopwords
    if _ready:
        return
    with _lock:
        if _ready:
            return
        for resource, package in _CORPORA:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(package, quiet=True)
        from nltk.corpus import stopwords

        words = set(stopwords.words("english"))
        words.update(_SP_CHAR)
        _stopwords = frozenset(words)
        _ready = True


@lru_cache(maxsize=65536)
def cal_score(word: str, tag: str) -> float:
    """Legacy synset scoring: max positive vs max negative over the synsets considered."""
    from nltk.corpus import sentiwordnet as swn

    s_pos: list[float] = []
    s_neg: list[float] = []
    for synset in swn.senti_synsets(word, tag):
        pos, neg = synset.pos_score(), synset.neg_score()
        s_pos.append(pos)
        s_neg.append(neg)
        if pos == 0.0 and neg == 0.0:
            # legacy stopped scanning at the first fully objective synset
            break
    if not s_pos:
        return 0.0
    max_pos, max_neg = max(s_pos), max(s_neg)
    return max_pos if max_pos > max_neg else -1.0 * max_neg


def filtered_text(tagged: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Lower-case, strip delimiters, drop numerics and stopwords (legacy filters 0-3)."""
    out: list[tuple[str, str]] = []
    for token, tag in tagged:
        word = _DELIMITER_RE.sub(" ", token.lower())
        if word.isnumeric() or word in _stopwords:
            continue
        out.append((word, tag))
    return out


def score_tagged_sentence(tagged: Iterable[tuple[str, str]]) -> float:
    """Sentiment score of one POS-tagged sentence under the legacy rule set."""
    score_post = 0.0
    adv_score = 0.0
    for word, tag in filtered_text(tagged):
        if tag in _ADV:
            adv_score += _ADV_WEIGHT[tag] * cal_score(word, "r")
        elif tag in _ADJ:
            adj_score = cal_score(word, "a")
            score_post += (_ADV_SCALE * adv_score + 1) * (_ADJ_WEIGHT[tag] * adj_score)
            adv_score = 0.0
        elif tag in _VERB:
            vb_score = cal_score(word, "v")
            score_post += (_ADV_SCALE * adv_score + 1) * vb_score
            adv_score = 0.0
    return score_post


def split_sentences(text: str) -> list[str]:
    """Legacy sentence split: a plain split on '.' (empty fragments included)."""
    return str(text).split(".")


def score_sentences(sentences: list[str]) -> list[float]:
    """Score many sentences at once (one tagger call for the whole batch)."""
    ensure_corpora()
    if not sentences:
        return []
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged = nltk.pos_tag_sents(tokens)
    return [score_tagged_sentence(sentence) for sentence in tagged]


def score_text(text: str) -> float:
    """Score free text the way one forum post was scored: sum over its '.'-split sentences."""
    return float(sum(score_sentences(split_sentences(text))))


def _normalise_date(value: str) -> str:
    # legacy: " ".join(date.split()[0:3]) -> "02 Dec 2015 08:22 " becomes "02 Dec 2015"
    return " ".join(str(value).split()[0:3])


def load_posts(disaster: Disaster) -> pd.DataFrame:
    """user/date/comment columns, NaNs dropped; the stray unnamed index column is ignored."""
    df = load_csv(disaster.csv)
    df = df[["user", "date", "comment"]].dropna()
    return df.reset_index(drop=True)


def score_dataset(disaster: Disaster) -> dict:
    """Legacy aggregation: sentence scores summed per post, mean per (date, user), then summed."""
    posts = load_posts(disaster)
    sentences: list[str] = []
    owners: list[int] = []
    for idx, comment in enumerate(posts["comment"]):
        for sentence in split_sentences(comment):
            sentences.append(sentence)
            owners.append(idx)
    sentence_scores = score_sentences(sentences)
    per_post = np.zeros(len(posts), dtype=float)
    np.add.at(per_post, owners, np.asarray(sentence_scores, dtype=float))
    frame = pd.DataFrame(
        {
            "date": posts["date"].map(_normalise_date),
            "user": posts["user"].astype(str),
            "sentiment_score": per_post,
        }
    )
    score = float(frame.groupby(["date", "user"])["sentiment_score"].mean().sum())
    return {"score": score, "n_posts": int(len(posts)), "n_sentences": int(len(sentences))}


def _plot(results: dict[str, dict]) -> None:
    by_slug = {d.slug: d for d in DISASTERS}
    xs = np.array([by_slug[slug].monetary_impact_busd for slug in results], dtype=float)
    ys = np.array([results[slug]["score"] for slug in results], dtype=float)
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=50, color="#2563eb", zorder=3)
    for slug, x, y in zip(results, xs, ys, strict=True):
        plt.annotate(
            by_slug[slug].name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8
        )
    if len(xs) >= 2 and np.ptp(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(xs.min(), xs.max(), 50)
        plt.plot(grid, slope * grid + intercept, color="#dc2626", label="linear fit")
        plt.legend()
    plt.title("Monetary impact vs forum sentiment")
    plt.xlabel("Monetary impact (billion USD)")
    plt.ylabel("Sentiment score")
    plots.save_current("sentiment", "impact_vs_sentiment.png")


def train() -> dict:
    ensure_corpora()
    results: dict[str, dict] = {}
    for disaster in DISASTERS:
        results[disaster.slug] = score_dataset(disaster)
    _plot(results)
    metadata = {
        "n_disasters": len(results),
        "n_posts": int(sum(r["n_posts"] for r in results.values())),
        "n_sentences": int(sum(r["n_sentences"] for r in results.values())),
        "scores": {slug: r["score"] for slug, r in results.items()},
    }
    registry.save(SCORES, results, metadata)
    return metadata


def scores() -> dict:
    return registry.load(SCORES)
