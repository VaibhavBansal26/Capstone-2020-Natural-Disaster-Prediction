import anthropic
import pytest

from ndp.config import get_settings
from ndp.services import ServiceError, llm


class FakeBetaMessages:
    def __init__(self, sink):
        self.sink = sink

    def create(self, **kwargs):
        self.sink.append(("beta", kwargs))
        return "beta-response"


class FakeMessages:
    def __init__(self, sink):
        self.sink = sink

    def create(self, **kwargs):
        self.sink.append(("plain", kwargs))
        return "plain-response"


class FakeClient:
    def __init__(self):
        self.calls = []
        self.beta = type("B", (), {"messages": FakeBetaMessages(self.calls)})()
        self.messages = FakeMessages(self.calls)


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


def test_not_configured_raises_friendly_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NDP_ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    assert not llm.is_configured()
    with pytest.raises(llm.LLMNotConfigured, match="ANTHROPIC_API_KEY"):
        llm.get_client()


def test_standard_env_var_is_accepted(env):
    env(ANTHROPIC_API_KEY="sk-test")
    assert llm.is_configured()


def test_default_model_uses_refusal_fallbacks_and_no_sampling_params(env, monkeypatch):
    env(ANTHROPIC_API_KEY="sk-test", NDP_ANTHROPIC_MODEL="claude-opus-5-5")
    fake = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    llm.create_message(max_tokens=100, messages=[{"role": "user", "content": "hi"}])
    kind, kwargs = fake.calls[0]
    assert kind == "beta" and kwargs["model"] == "claude-opus-5-5"
    assert kwargs["betas"] == [llm.FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    for banned in ("temperature", "top_p", "top_k", "tool_choice"):
        assert banned not in kwargs


def test_models_without_fallback_support_skip_the_parameter(env, monkeypatch):
    env(ANTHROPIC_API_KEY="sk-test", NDP_ANTHROPIC_MODEL="claude-haiku-5-5")
    fake = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    llm.create_message(max_tokens=100, messages=[])
    kind, kwargs = fake.calls[0]
    assert kind == "plain" and "fallbacks" not in kwargs


def test_sdk_errors_become_service_errors():
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    with pytest.raises(ServiceError, match="rate limited"):
        with llm.friendly_errors():
            raise anthropic.RateLimitError("slow down", response=response, body=None)
    with pytest.raises(ServiceError, match="Could not reach"):
        with llm.friendly_errors():
            raise anthropic.APIConnectionError(request=request)


def test_text_of_skips_non_text_blocks():
    block = lambda t, **kw: type("Block", (), {"type": t, **kw})()  # noqa: E731
    response = type(
        "R", (), {"content": [block("thinking"), block("text", text=" hello "), block("tool_use")]}
    )()
    assert llm.text_of(response) == "hello"
