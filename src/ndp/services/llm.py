"""Shared Claude client for the assistant, post triage and alert translation.

Everything that talks to the Anthropic API goes through here so the safe defaults live in one
place:

* One cached client built from settings (``NDP_ANTHROPIC_API_KEY`` or ``ANTHROPIC_API_KEY``).
* The model comes from ``NDP_ANTHROPIC_MODEL`` (default ``claude-opus-5-5``). Sonnet 5.5 or Haiku
  5.5 are cheaper choices for a public deployment.
* No sampling parameters (temperature, top_p, top_k are rejected by current models), no assistant
  prefill, no forced ``tool_choice`` (rejected by Opus 5.5): steer with the prompt instead.
* ``stop_reason == "refusal"`` is a normal 200 response; callers check it via ``refused()``.
  Server-side refusal fallbacks are requested by ``create_message`` for the models that support
  them; ``parse`` (structured output, used by triage) calls the SDK directly and does not.
* Anthropic SDK errors become ``ServiceError`` with a message safe to show in the UI.

Tests replace ``get_client`` with a fake; nothing here runs without a key.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import anthropic
from pydantic import BaseModel

from ndp.config import get_settings
from ndp.services import ServiceError

log = logging.getLogger("ndp.llm")

_usage: dict[str, int] = {
    "requests": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}
_usage_lock = threading.Lock()


def _record_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    with _usage_lock:
        _usage["requests"] += 1
        for key in tuple(_usage)[1:]:
            _usage[key] += int(getattr(usage, key, 0) or 0)


def usage_totals() -> dict[str, int]:
    """Cumulative token counters for this process (used by the eval runners to report cost)."""
    with _usage_lock:
        return dict(_usage)


FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept `fallbacks: "default"` on the Claude API. Others (e.g. Haiku 5.5) reject it.
FALLBACK_MODELS = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
)


class LLMNotConfigured(ServiceError):
    """Raised when a Claude-powered feature is used without an API key."""

    def __init__(self) -> None:
        super().__init__(
            "Claude is not configured on this server. Set ANTHROPIC_API_KEY "
            "(or NDP_ANTHROPIC_API_KEY) to enable this feature."
        )


@lru_cache(maxsize=4)
def _client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=2)


def is_configured() -> bool:
    return get_settings().llm_configured


def get_client() -> anthropic.Anthropic:
    key = get_settings().anthropic_api_key
    if not key:
        raise LLMNotConfigured
    return _client(key)


def model_name() -> str:
    return get_settings().anthropic_model


@contextmanager
def friendly_errors() -> Iterator[None]:
    """Translate Anthropic SDK exceptions into ServiceError messages the UI can show."""
    try:
        yield
    except ServiceError:
        raise
    except anthropic.AuthenticationError as exc:
        log.error("Anthropic rejected the API key: %s", exc)
        raise ServiceError("The Anthropic API key was rejected. Check the server's key.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise ServiceError("The Anthropic API key lacks permission for this model.") from exc
    except anthropic.NotFoundError as exc:
        raise ServiceError(
            f"The model {model_name()!r} was not found. Check NDP_ANTHROPIC_MODEL."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise ServiceError("Claude is rate limited right now; please retry in a minute.") from exc
    except anthropic.APITimeoutError as exc:
        raise ServiceError("Claude took too long to answer; please try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise ServiceError("Could not reach the Anthropic API; check the network.") from exc
    except anthropic.APIStatusError as exc:
        log.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        if exc.status_code >= 500:
            raise ServiceError("Claude is temporarily unavailable; please retry.") from exc
        raise ServiceError(f"Claude rejected the request ({exc.status_code}).") from exc


def create_message(**kwargs: Any) -> Any:
    """`messages.create` with the app's model and refusal fallbacks. Pass `system`, `messages`,
    `max_tokens`, `tools` etc. Returns the SDK message; check `refused()` before reading content."""
    client = get_client()
    model = kwargs.pop("model", None) or model_name()
    with friendly_errors():
        if model in FALLBACK_MODELS:
            response = client.beta.messages.create(
                model=model, betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = client.messages.create(model=model, **kwargs)
    _record_usage(response)
    return response


def refused(response: Any) -> bool:
    return getattr(response, "stop_reason", None) == "refusal"


def text_of(response: Any) -> str:
    """Concatenated text blocks of a response (thinking and tool blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


def parse[T: BaseModel](schema: type[T], system: str, user: str, max_tokens: int = 2000) -> T:
    """One structured-output call validated against a Pydantic model."""
    client = get_client()
    with friendly_errors():
        response = client.messages.parse(
            model=model_name(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
    _record_usage(response)
    if refused(response):
        raise ServiceError("Claude declined to process this text.")
    if response.parsed_output is None:
        raise ServiceError("Claude returned an answer that could not be parsed; please retry.")
    return response.parsed_output
