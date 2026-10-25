from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_text
from ndp.ml import registry, sentiment
from ndp.ml.features import DataMissing
from ndp.services import ServiceError, llm, triage
from ndp.services.ratelimit import RateLimiter
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/sentiment")

# Scores computed on demand because `ndp-train` has not run; kept for the life of the process.
_computed: dict[str, dict] = {}

# One click fills the triage box. The first mirrors the 2015 Chennai flood threads in the data.
TRIAGE_EXAMPLES = [
    {
        "label": "Chennai flood, rescue",
        "text": "Water is rising in Velachery, we are on the first floor with two kids and an "
        "elderly parent, please send a boat",
    },
    {
        "label": "Hindi-English mix, medicine",
        "text": "Bhai Kerala mein hamare ghar ke paas koi insulin bhej sakta hai? Papa ko diabetes "
        "hai aur do din se dawai khatam hai, road band hai",
    },
    {
        "label": "Spanish, missing person",
        "text": "Busco a mi hermano Luis, estaba en Lorca cuando empezo el terremoto y no contesta "
        "el telefono desde ayer",
    },
    {
        "label": "Relief camp info",
        "text": "Is the relief camp at the school in Kottayam open tonight? Which roads are "
        "passable from Ernakulam?",
    },
]


def _rows() -> list[dict]:
    try:
        scores = sentiment.scores()
    except registry.ModelNotTrained:
        scores = {}
    rows = []
    for d in sentiment.DISASTERS:
        entry = scores.get(d.slug) or {}
        rows.append(
            {
                "slug": d.slug,
                "name": d.name,
                "image": d.image,
                "monetary_impact": d.monetary_impact,
                "score": entry.get("score"),
                "n_posts": entry.get("n_posts"),
            }
        )
    return rows


def _page_context() -> dict:
    plot_path = get_settings().generated_dir / "sentiment" / "impact_vs_sentiment.png"
    return {
        "rows": _rows(),
        "plot": "generated/sentiment/impact_vs_sentiment.png" if plot_path.exists() else None,
        "meta": registry.metadata(sentiment.SCORES),
        "trained": registry.is_trained(sentiment.SCORES),
        "llm_enabled": llm.is_configured(),
        "triage_examples": TRIAGE_EXAMPLES,
        "triage_max_chars": triage.MAX_POST_CHARS,
        "sample_datasets": [{"slug": d.slug, "name": d.name} for d in sentiment.DISASTERS],
    }


@router.get("", name="sentiment_page")
def sentiment_page(request: Request):
    return render(request, "sentiment/page.html", **_page_context())


@router.get("/score", include_in_schema=False)
def sentiment_score_get(request: Request):
    return render(request, "sentiment/_text_score.html", text="", score=None, label=None)


@router.post("/score", name="sentiment_score_text")
def sentiment_score_text(request: Request, text: str = Form("")):
    try:
        clean = parse_text(text, "Text", 5000)
    except FormError as exc:
        return render(request, "sentiment/_text_score.html", text=text, score=None, error=str(exc))
    score = sentiment.score_text(clean)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return render(request, "sentiment/_text_score.html", text=clean, score=score, label=label)


_NOT_CONFIGURED = "not_configured"

# Each triage call costs real money on the owner's key: cap it per client address.
triage_limiter = RateLimiter(20, 600.0)  # single posts
sample_limiter = RateLimiter(4, 600.0)  # each click makes up to five calls


def _too_many(limiter: RateLimiter, request: Request) -> str | None:
    allowed, retry = limiter.hit(request.client.host if request.client else "unknown")
    return (
        None
        if allowed
        else f"Too many requests from this address. Try again in {int(retry) + 1} seconds."
    )


@router.post("/triage", name="sentiment_triage")
def sentiment_triage(request: Request, text: str = Form("")):
    """AI triage suggestion for one pasted post. Needs Claude; a person must verify the result."""

    def respond(**result):
        return page_or_partial(
            request, "sentiment/page.html", "sentiment/_triage.html", _page_context, **result
        )

    if not llm.is_configured():
        return respond(triage_state=_NOT_CONFIGURED, triage_text=text)
    if refused := _too_many(triage_limiter, request):
        return respond(triage_error=refused, triage_text=text)
    try:
        clean = parse_text(text, "Post", triage.MAX_POST_CHARS)
        result = triage.triage_post(clean)
    except (FormError, ServiceError) as exc:
        return respond(triage_error=str(exc), triage_text=text)
    return respond(triage=result, triage_text=clean)


@router.post("/triage/sample", name="sentiment_triage_sample")
def sentiment_triage_sample(request: Request, dataset: str = Form("")):
    """Triage up to 5 random posts from one disaster's dataset; each row succeeds or fails alone."""

    def respond(**result):
        return page_or_partial(
            request, "sentiment/page.html", "sentiment/_triage_sample.html", _page_context, **result
        )

    if not llm.is_configured():
        return respond(sample_state=_NOT_CONFIGURED)
    slug = dataset.strip() or sentiment.DISASTERS[0].slug
    if refused := _too_many(sample_limiter, request):
        return respond(sample_error=refused, sample_slug=slug)
    try:
        pairs = triage.triage_sample(slug, triage.MAX_SAMPLE)
    except ServiceError as exc:
        return respond(sample_error=str(exc), sample_slug=slug)
    rows = [
        {
            "text": post,
            "triage": None if isinstance(outcome, triage.TriageFailure) else outcome,
            "error": outcome.message if isinstance(outcome, triage.TriageFailure) else None,
        }
        for post, outcome in pairs
    ]
    name = next((d.name for d in sentiment.DISASTERS if d.slug == slug), slug)
    return respond(sample_rows=rows, sample_slug=slug, sample_name=name)


@router.get("/{slug}", name="sentiment_score")
def sentiment_score(request: Request, slug: str):
    row = next((r for r in _rows() if r["slug"] == slug), None)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown disaster")
    if row["score"] is None:
        # Not precomputed yet (ndp-train not run): compute on demand, once, and cache it.
        disaster = next(d for d in sentiment.DISASTERS if d.slug == slug)
        scored = _computed.get(slug)
        if scored is None:
            try:
                scored = _computed[slug] = sentiment.score_dataset(disaster)
            except DataMissing as exc:
                return render(request, "sentiment/_score.html", row=row, error=str(exc))
        row.update(score=scored["score"], n_posts=scored["n_posts"])
    return render(request, "sentiment/_score.html", row=row)
