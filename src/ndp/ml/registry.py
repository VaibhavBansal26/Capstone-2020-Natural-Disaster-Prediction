"""Save and load trained artifacts with a small JSON sidecar of metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from ndp.config import get_settings


class ModelNotTrained(RuntimeError):
    """Raised when a route needs a model that `ndp-train` has not produced yet."""

    def __init__(self, name: str):
        super().__init__(f"model '{name}' is not trained yet; run `uv run ndp-train`")
        self.name = name


def _paths(name: str) -> tuple[Path, Path]:
    models_dir = get_settings().models_dir
    return models_dir / f"{name}.joblib", models_dir / f"{name}.json"


def save(name: str, artifact: Any, metadata: dict[str, Any] | None = None) -> Path:
    model_path, meta_path = _paths(name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    meta = {"name": name, "trained_at": datetime.now(UTC).isoformat(timespec="seconds")}
    meta.update(metadata or {})
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    load.cache_clear()
    return model_path


@lru_cache(maxsize=32)
def load(name: str) -> Any:
    model_path, _ = _paths(name)
    if not model_path.exists():
        raise ModelNotTrained(name)
    return joblib.load(model_path)


def metadata(name: str) -> dict[str, Any] | None:
    _, meta_path = _paths(name)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def is_trained(name: str) -> bool:
    return _paths(name)[0].exists()


def all_metadata() -> list[dict[str, Any]]:
    models_dir = get_settings().models_dir
    if not models_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(models_dir.glob("*.json"))]
