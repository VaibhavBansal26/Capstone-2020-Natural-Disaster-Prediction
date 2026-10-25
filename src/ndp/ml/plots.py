"""Helpers for the plots that are produced once at train time and served as static files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ndp.config import get_settings  # noqa: E402


def generated_dir(sub: str) -> Path:
    path = get_settings().generated_dir / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_current(sub: str, filename: str) -> str:
    """Save the current matplotlib figure under static/generated/<sub>/ and return its URL path."""
    target = generated_dir(sub) / filename
    plt.tight_layout()
    plt.savefig(target, dpi=110)
    plt.close("all")
    return f"generated/{sub}/{filename}"


def bar_comparison(
    title: str, values: dict[str, float], sub: str, filename: str, ylabel: str
) -> str:
    plt.figure(figsize=(6, 4))
    plt.bar(list(values.keys()), list(values.values()), color="#2563eb", width=0.5)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15)
    return save_current(sub, filename)
