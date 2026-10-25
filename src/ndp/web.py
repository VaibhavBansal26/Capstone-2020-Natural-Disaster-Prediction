"""Jinja2 templating shared by all routers."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates

from ndp.config import PACKAGE_DIR, get_settings

templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


def render(request: Request, name: str, *, status_code: int = 200, **context: Any):
    """Render a template with the settings object always available as `settings`."""
    context.setdefault("settings", get_settings())
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def page_or_partial(request: Request, page: str, partial: str, page_context, **result):
    """POST handlers call this: HTMX gets the partial, a plain browser gets the full page
    with the partial's context merged in (so the no-JS fallback still shows the result)."""
    if is_htmx(request):
        return render(request, partial, **result)
    context = page_context()
    context.update(result)
    return render(request, page, **context)
