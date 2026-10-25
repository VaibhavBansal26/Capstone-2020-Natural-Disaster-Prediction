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
    os.environ.setdefault("NDP_ENABLE_COVID", "false")
    from ndp.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
