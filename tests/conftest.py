import os
from pathlib import Path

import pytest

# Keep tests hermetic: models and generated plots go to a temp dir, never the real ones.
_tmp_root = Path(os.environ.get("PYTEST_NDP_TMP", "/tmp")) / "ndp-tests"


@pytest.fixture(autouse=True, scope="session")
def _isolated_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("ndp")
    os.environ["NDP_MODELS_DIR"] = str(base / "models")
    os.environ["NDP_GENERATED_DIR"] = str(base / "generated")
    from ndp.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Per-client limiters are module state; a test must not inherit another test's hits."""
    from ndp.routers import alerts
    from ndp.routers import sentiment as sentiment_router
    from ndp.services import assistant

    for limiter in (
        alerts.send_limiter,
        alerts.translate_limiter,
        sentiment_router.triage_limiter,
        sentiment_router.sample_limiter,
        assistant.rate_limiter,
    ):
        limiter.reset()
    yield


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings are cached; tests that change NDP_* variables must not leak into the next test."""
    from ndp.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
