"""Small form-parsing helpers that turn bad input into a friendly message instead of a 422."""

from __future__ import annotations

from datetime import date


class FormError(ValueError):
    """Raised for invalid user input; routers render it as `error` in the result partial."""


def parse_float(
    value: str | None, label: str, lo: float | None = None, hi: float | None = None
) -> float:
    if value is None or str(value).strip() == "":
        raise FormError(f"{label} is required.")
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise FormError(f"{label} must be a number (got {value!r}).") from exc
    if number != number:  # NaN
        raise FormError(f"{label} must be a number.")
    if lo is not None and number < lo:
        raise FormError(f"{label} must be at least {lo:g}.")
    if hi is not None and number > hi:
        raise FormError(f"{label} must be at most {hi:g}.")
    return number


def parse_date(value: str | None, label: str = "Date") -> date:
    if value is None or str(value).strip() == "":
        raise FormError(f"{label} is required.")
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise FormError(f"{label} must look like 2026-10-25 (got {value!r}).")


def parse_text(value: str | None, label: str, max_len: int = 2000) -> str:
    text = (value or "").strip()
    if not text:
        raise FormError(f"{label} is required.")
    if len(text) > max_len:
        raise FormError(f"{label} is too long (max {max_len} characters).")
    return text
