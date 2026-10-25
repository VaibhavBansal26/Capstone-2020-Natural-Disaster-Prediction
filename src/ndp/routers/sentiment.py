from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request

from ndp.config import get_settings
from ndp.forms import FormError, parse_text
from ndp.ml import registry, sentiment
from ndp.web import render

router = APIRouter(prefix="/sentiment")


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
                "monetary_impact_busd": d.monetary_impact_busd,
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


@router.get("/{slug}", name="sentiment_score")
def sentiment_score(request: Request, slug: str):
    row = next((r for r in _rows() if r["slug"] == slug), None)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown disaster")
    if row["score"] is None:
        # Not precomputed yet (ndp-train not run): compute on demand, once, and cache it.
        disaster = next(d for d in sentiment.DISASTERS if d.slug == slug)
        scored = sentiment.score_dataset(disaster)
        row.update(score=scored["score"], n_posts=scored["n_posts"])
    return render(request, "sentiment/_score.html", row=row)
