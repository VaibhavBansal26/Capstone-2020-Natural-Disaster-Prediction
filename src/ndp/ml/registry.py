"""Save and load trained artifacts with a small JSON sidecar of metadata."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
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


def _atomic_write(target: Path, write: Callable[[Any], None]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.chmod(tmp_name, 0o644)  # mkstemp creates 0600; match a normal file write
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save(name: str, artifact: Any, metadata: dict[str, Any] | None = None) -> Path:
    model_path, meta_path = _paths(name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "trained_at": datetime.now(UTC).isoformat(timespec="seconds")}
    meta.update(metadata or {})
    # Write next to the target and rename, so a reader never sees a half-written file.
    _atomic_write(model_path, lambda handle: joblib.dump(artifact, handle))
    _atomic_write(
        meta_path, lambda handle: handle.write(json.dumps(meta, indent=2, default=str).encode())
    )
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
