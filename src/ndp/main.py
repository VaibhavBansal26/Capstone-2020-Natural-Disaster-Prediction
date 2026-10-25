"""FastAPI application factory and `ndp-serve` entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ndp import __version__
from ndp.config import PACKAGE_DIR, get_settings
from ndp.ml.registry import ModelNotTrained
from ndp.web import render


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings().generated_dir.mkdir(parents=True, exist_ok=True)
    _warm_up()
    from ndp.services import watcher

    await watcher.start(app)
    try:
        yield
    finally:
        await watcher.stop(app)


def _warm_up() -> None:
    """Load the heavier artifacts and the NLTK tagger once at startup so the first request
    does not pay for them. Missing models or corpora are fine here; pages report them."""
    import logging

    from ndp.ml import compare, registry, sentiment

    log = logging.getLogger("ndp")
    for hazard in compare.HAZARDS:
        name = compare.artifact_name(hazard)
        if registry.is_trained(name):
            registry.load(name)
    try:
        sentiment.ensure_corpora()
    except Exception as exc:  # network down: the sentiment page will retry lazily
        log.warning("could not prepare NLTK corpora at startup: %s", exc)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Natural Disaster Prediction",
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    from ndp.routers import (
        alerts,
        api,
        assistant,
        earthquake,
        flood,
        hurricane,
        live,
        pages,
        sentiment,
        storm,
        tsunami,
        weather,
    )

    for router in (
        pages,
        earthquake,
        tsunami,
        flood,
        hurricane,
        storm,
        weather,
        alerts,
        sentiment,
        live,
        assistant,
        api,
    ):
        app.include_router(router.router)
    if settings.enable_covid:
        from ndp.routers import covid

        app.include_router(covid.router)

    @app.exception_handler(ModelNotTrained)
    async def _model_not_trained(request: Request, exc: ModelNotTrained):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": str(exc), "model": exc.name}, status_code=503)
        return render(request, "errors/not_trained.html", model_name=exc.name, status_code=503)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()


def serve() -> None:
    settings = get_settings()
    uvicorn.run("ndp.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
