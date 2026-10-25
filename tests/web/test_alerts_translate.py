"""Alert translation UI and sending: preview endpoint, review-then-send, SMS length handling.

The model is faked; no real translation is produced here."""

import re

import pytest
from fastapi.testclient import TestClient

from ndp.config import get_settings
from ndp.main import create_app
from ndp.routers import alerts as alerts_router
from ndp.services import ServiceError, alerting, geo, llm, translate
from ndp.services.geo import NearbyPlace, Place

HX = {"HX-Request": "true"}
PLACES = [NearbyPlace("City Hospital", "Hospital", 28.61, 77.21, "Main Rd", "+91 11 2222")]
MESSAGE = (
    "Emergency alert for Delhi (28.610, 77.209).\n"
    "Nearest hospitals: City Hospital (+91 11 2222).\n"
    "Please follow official guidance and move to safety."
)
HINDI = "दिल्ली के लिए आपातकालीन चेतावनी (28.610, 77.209)।"


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setenv(f"NDP_{key.upper()}", str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(geo, "reverse_geocode", lambda lat, lon: Place("Delhi, India", lat, lon))
    monkeypatch.setattr(geo, "nearby_services", lambda lat, lon, radius, limit=40: PLACES)
    return TestClient(create_app())


@pytest.fixture
def llm_on(monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: True)


@pytest.fixture
def llm_off(monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: False)


def nearby(client):
    return client.post("/alerts/nearby", headers=HX, data={"lat": "28.6", "lon": "77.2"}).text


# ---------------------------------------------------------------- the send form


def test_form_offers_languages_and_preview_when_configured(client, llm_on):
    html = nearby(client)
    for code, name in translate.LANGUAGES.items():
        assert f'name="languages" value="{code}"' in html and name in html
    assert "Preview translations" in html
    assert 'data-voice-command="preview translations"' in html
    # the preview button must not trigger the form's "send this alert?" confirm
    assert re.search(r'hx-post="[^"]*/alerts/translate"', html) and 'hx-confirm="unset"' in html
    assert html.index('id="translations"') > html.index('<form method="post"')
    assert html.index('id="translations"') < html.index("Send alert\n")


def test_existing_send_form_contract_is_intact_with_translation_controls(client, llm_on):
    html = nearby(client)
    assert 'hx-confirm="Send this alert' in html
    assert 'data-voice-command="send alert"' in html and "data-voice-confirm" in html
    assert "EARTHQUAKE" not in html and "Emergency alert for Delhi, India" in html


def test_translation_controls_are_replaced_by_a_note_without_claude(client, llm_off):
    html = nearby(client)
    assert "Translation is off on this server" in html and "ANTHROPIC_API_KEY" in html
    assert "Preview translations" not in html and 'name="languages"' not in html
    assert 'hx-confirm="Send this alert' in html  # sending still works


# ---------------------------------------------------------------- /alerts/translate


@pytest.fixture
def fake_translate(monkeypatch):
    seen = {}

    def many(message, languages):
        seen["message"], seen["languages"] = message, list(languages)
        out = {}
        for code in languages:
            name = translate.LANGUAGES[code]
            if code == "ta":
                out[code] = translate.Translation(code, name, error="Rejected: numbers missing.")
            else:
                out[code] = translate.Translation(
                    code, name, text=f"{HINDI}\n\n{translate.MT_NOTE}"
                )
        return out

    monkeypatch.setattr(translate, "translate_many", many)
    return seen


def test_translate_returns_editable_textareas_with_rtl_for_arabic(client, llm_on, fake_translate):
    resp = client.post(
        "/alerts/translate",
        headers=HX,
        data={"message": MESSAGE, "languages": ["ar", "hi", "ta"]},
    )
    html = resp.text
    assert resp.status_code == 200 and "<html" not in html
    assert '<textarea name="translation_hi" lang="hi"' in html
    assert 'name="translation_ar" lang="ar" dir="rtl"' in html
    assert 'name="translation_hi" lang="hi" dir=' not in html
    assert "<span" in html and ">Hindi<" in html and ">Arabic<" in html
    assert "Read and edit each one before sending" in html
    assert "Tamil</strong> was not drafted" in html and "numbers missing" in html
    assert 'name="translation_ta"' not in html  # failures get no textarea
    assert fake_translate["languages"] == ["hi", "ta", "ar"]  # picker order, not click order
    assert translate.MT_NOTE in html


def test_translate_validates_and_reports_problems(client, llm_on, fake_translate):
    def post(**data):
        return client.post("/alerts/translate", headers=HX, data=data).text

    assert "Choose at least one language" in post(message=MESSAGE)
    assert "Unsupported language: xx" in post(message=MESSAGE, languages=["xx"])
    assert "Message is required" in post(message=" ", languages=["hi"])
    assert "too long" in post(message="x" * 1201, languages=["hi"])
    assert fake_translate == {}


def test_translate_without_claude_and_service_errors(client, monkeypatch, fake_translate):
    monkeypatch.setattr(llm, "is_configured", lambda: False)
    html = client.post(
        "/alerts/translate", headers=HX, data={"message": MESSAGE, "languages": ["hi"]}
    ).text
    assert "not configured" in html and "translation_hi" not in html

    monkeypatch.setattr(llm, "is_configured", lambda: True)

    def boom(message, languages):
        raise ServiceError("Claude is rate limited right now; please retry in a minute.")

    monkeypatch.setattr(translate, "translate_many", boom)
    html = client.post(
        "/alerts/translate", headers=HX, data={"message": MESSAGE, "languages": ["hi"]}
    ).text
    assert "Could not translate" in html and "rate limited" in html


# ---------------------------------------------------------------- /alerts/send


@pytest.fixture
def sent(monkeypatch):
    """Capture what reaches dispatch (new signature with sms_body)."""
    captured = {}

    def dispatch(place_name, body, phone=None, sms_body=None):
        captured.update(place=place_name, body=body, phone=phone, sms_body=sms_body)
        return alerting.DispatchResult(channels=["email to ops@example.com"])

    monkeypatch.setattr(alerting, "dispatch", dispatch)
    return captured


def test_send_appends_non_empty_translations_under_headings_english_first(client, sent):
    resp = client.post(
        "/alerts/send",
        headers=HX,
        data={
            "place": "Delhi",
            "message": MESSAGE,
            "translation_hi": HINDI,
            "translation_ar": "   ",
            "translation_fr": "Alerte d'urgence pour Delhi (28,610, 77,209).",
            "translation_xx": "ignored: not a supported language",
        },
    )
    body = sent["body"]
    assert body.startswith(MESSAGE)
    assert body.index("=== Hindi ===") < body.index("=== French ===")
    assert HINDI in body and "Alerte d'urgence" in body
    assert "Arabic" not in body and "ignored" not in body
    html = resp.text
    assert "Languages included:" in html and "English, Hindi, French" in html
    assert "Show the translations sent" in html and HINDI in html


def test_send_without_translations_behaves_as_before(client, sent):
    html = client.post("/alerts/send", headers=HX, data={"place": "Delhi", "message": MESSAGE}).text
    assert sent["body"] == MESSAGE and sent["sms_body"] is None
    assert "Languages included:" in html and "English" in html
    assert "Show the translations sent" not in html and "SMS carries" not in html


def test_edited_translation_is_what_gets_sent(client, sent):
    client.post(
        "/alerts/send",
        headers=HX,
        data={
            "place": "Delhi",
            "message": MESSAGE,
            "translation_es": "Texto corregido por una persona",
        },
    )
    assert "Texto corregido por una persona" in sent["body"]


def test_dry_run_send_with_translation_still_says_nothing_was_sent(client, env):
    env(alerts_dry_run="true")
    html = client.post(
        "/alerts/send",
        headers=HX,
        data={"place": "Delhi", "message": MESSAGE, "translation_hi": HINDI},
    ).text
    assert "Dry run, nothing was sent" in html and "Hindi" in html


def test_send_rejects_an_overlong_translation(client, sent):
    html = client.post(
        "/alerts/send",
        headers=HX,
        data={"place": "Delhi", "message": MESSAGE, "translation_hi": "क" * 4001},
    ).text
    assert "Hindi translation is too long" in html and sent == {}


def test_full_page_fallback_for_send(client, sent):
    html = client.post(
        "/alerts/send", data={"place": "Delhi", "message": MESSAGE, "translation_hi": HINDI}
    ).text
    assert "<html" in html and "Languages included:" in html


# ---------------------------------------------------------------- SMS length


def test_compose_short_message_goes_whole_to_both_channels():
    email, sms, english_only = alerts_router.compose_bodies("Hello", {"hi": "नमस्ते"})
    assert email == sms == "Hello\n\n=== Hindi ===\nनमस्ते" and not english_only


def test_compose_without_translations_is_the_message_itself():
    assert alerts_router.compose_bodies("Hello", {}) == ("Hello", "Hello", False)


def test_compose_long_message_sends_sms_english_only_plus_note():
    english = "A" * 1200
    email, sms, english_only = alerts_router.compose_bodies(
        english, {"hi": "क" * 900, "ta": "த" * 900}
    )
    assert english_only and len(email) > 1500
    assert email.startswith(english) and "=== Tamil ===" in email
    assert sms.startswith(english[:1000]) and sms.endswith(alerts_router.SMS_TRANSLATIONS_NOTE)
    assert len(sms) <= alerts_router.SMS_LIMIT
    assert "=== Hindi ===" not in sms and "क" not in sms


def test_send_long_message_uses_separate_sms_body_and_says_so(client, sent):
    html = client.post(
        "/alerts/send",
        headers=HX,
        data={
            "place": "Delhi",
            "message": "Short English alert",
            "translation_hi": "क" * 1800,
            "translation_ta": "த" * 1800,
        },
    ).text
    assert sent["sms_body"].startswith("Short English alert")
    assert sent["sms_body"].endswith(alerts_router.SMS_TRANSLATIONS_NOTE)
    assert len(sent["body"]) > 1500 and "த" in sent["body"]
    assert "SMS carries the English text only" in html
