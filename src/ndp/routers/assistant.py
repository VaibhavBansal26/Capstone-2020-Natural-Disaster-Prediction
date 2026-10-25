"""Claude-powered assistant page and API.

``GET /assistant`` shows the chat, ``POST /assistant/ask`` answers one message (an HTMX partial
with the user's message, the answer, tool chips and sources) and ``POST /assistant/reset`` starts
over. Everything the user can trigger here ends in a friendly message in the chat, never a 500.

The session cookie is a random 128-bit token (``HttpOnly``, ``SameSite=Lax``) and is only ever set
by these three endpoints.
"""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response

from ndp.config import get_settings
from ndp.services import ServiceError, assistant
from ndp.web import is_htmx, templates

log = logging.getLogger("ndp.assistant")

router = APIRouter(prefix="/assistant")

COOKIE_NAME = "ndp_assistant"
COOKIE_MAX_AGE = 6 * 3600

EXAMPLES = (
    "Is there a tsunami risk after a magnitude 7 near Japan?",
    "What should I do during an earthquake?",
    "Weather outlook for Chennai this week",
    "Which earthquakes were strongest this week?",
    "Where is the nearest hospital to Kathmandu?",
)

templates.env.filters["assistant_md"] = assistant.safe_markdown
templates.env.filters["assistant_plain"] = assistant.plain_text


def render(request: Request, name: str, status_code: int = 200, **context):
    """Like ``ndp.web.render`` but with a response status (it is not a template variable)."""
    context.setdefault("settings", get_settings())
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def _session(request: Request) -> tuple[str, bool]:
    """(session id, is_new). An invalid or missing cookie gets a fresh token."""
    cookie = request.cookies.get(COOKIE_NAME)
    if assistant.valid_session_id(cookie):
        return cookie, False  # type: ignore[return-value]
    return assistant.new_session_id(), True


def _with_cookie(response: Response, request: Request, session_id: str, is_new: bool) -> Response:
    if is_new:
        response.set_cookie(
            COOKIE_NAME,
            session_id,
            max_age=COOKIE_MAX_AGE,
            path="/assistant",
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
    return response


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _page_context(session_id: str | None = None) -> dict:
    settings = get_settings()
    return {
        "configured": settings.llm_configured,
        "examples": EXAMPLES,
        "turns": assistant.history_for_page(session_id),
        "max_chars": assistant.MAX_MESSAGE_CHARS,
        "model": settings.anthropic_model,
    }


@router.get("", name="assistant_page")
def assistant_page(request: Request):
    session_id, is_new = _session(request)
    response = render(request, "assistant/page.html", **_page_context(session_id))
    return _with_cookie(response, request, session_id, is_new)


def _error_turn(
    request: Request,
    session_id: str,
    is_new: bool,
    question: str,
    message: str,
    status_code: int,
    headers: dict[str, str] | None = None,
):
    """A friendly error as a chat bubble. HTMX gets the partial; a plain POST gets the full page
    so the form still works without JavaScript."""
    if is_htmx(request):
        response = render(
            request,
            "assistant/_turn.html",
            status_code=status_code,
            question=question,
            error=message,
        )
    else:
        context = _page_context(session_id)
        context.update(pending_question=question, pending_error=message)
        response = render(request, "assistant/page.html", status_code=status_code, **context)
    for key, value in (headers or {}).items():
        response.headers[key] = value
    return _with_cookie(response, request, session_id, is_new)


@router.post("/ask", name="assistant_ask")
def assistant_ask(request: Request, message: str = Form("")):
    session_id, is_new = _session(request)
    question = (message or "").strip()[: assistant.MAX_MESSAGE_CHARS + 1]

    def fail(text: str, status: int, headers: dict[str, str] | None = None):
        return _error_turn(request, session_id, is_new, question, text, status, headers)

    if not get_settings().llm_configured:
        return fail(
            "The assistant is not set up on this server yet. Set ANTHROPIC_API_KEY and restart.",
            503,
        )
    allowed, retry_after = assistant.rate_limiter.hit(_client_key(request))
    if not allowed:
        wait = max(1, math.ceil(retry_after))
        return fail(
            f"You are sending messages faster than I can keep up with. "
            f"Please try again in {wait} seconds.",
            429,
            {"Retry-After": str(wait)},
        )
    try:
        turn = assistant.ask(session_id, message)
    except assistant.AssistantError as exc:
        return fail(str(exc), 400)
    except ServiceError as exc:
        return fail(str(exc), 503)
    except Exception:
        log.exception("assistant failed")
        return fail("Something went wrong on my side. Please try again in a moment.", 503)

    if is_htmx(request):
        response = render(request, "assistant/_turn.html", question=question, turn=turn)
    else:
        context = _page_context(session_id)
        response = render(request, "assistant/page.html", **context)
    return _with_cookie(response, request, session_id, is_new)


@router.post("/reset", name="assistant_reset")
def assistant_reset(request: Request):
    session_id, is_new = _session(request)
    assistant.sessions.reset(session_id)
    if is_htmx(request):
        response: Response = render(request, "assistant/_welcome.html", **_page_context())
    else:
        response = RedirectResponse(request.url_for("assistant_page"), status_code=303)
    return _with_cookie(response, request, session_id, is_new)
