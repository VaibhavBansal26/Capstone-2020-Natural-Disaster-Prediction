"""Application settings, loaded from environment variables or a local .env file.

Every secret or key is optional. Features that need one report themselves as
"not configured" instead of failing, so the app always starts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NDP_", env_file=".env", extra="ignore")

    debug: bool = False
    enable_covid: bool = True

    data_dir: Path = PROJECT_DIR / "data" / "raw"
    models_dir: Path = PROJECT_DIR / "models"
    generated_dir: Path = PACKAGE_DIR / "static" / "generated"

    openweather_api_key: str = ""

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    alert_from_email: str = ""
    alert_to_email: str = ""

    # Claude-powered features (assistant, post triage, alert translation). Optional: without a key
    # the pages say "not configured" and everything else works. The standard ANTHROPIC_API_KEY
    # variable is accepted as well as NDP_ANTHROPIC_API_KEY.
    anthropic_api_key: str = Field(
        default="", validation_alias=AliasChoices("NDP_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    anthropic_model: str = "claude-opus-5-5"
    assistant_max_tool_rounds: int = 6
    guidance_dir: Path = PROJECT_DIR / "data" / "guidance"

    # Real-event watcher: polls USGS and alerts the configured contacts for new strong events.
    watch_quakes: bool = False
    watch_min_magnitude: float = 6.0
    watch_interval_seconds: int = 300

    # Alert behaviour. Automatic alerts come from the real-event watcher (watch_quakes) and are
    # opt-in; manual alerts always need a click and a confirm.
    auto_alert: bool = False
    alerts_dry_run: bool = False
    alert_radius_km: float = 15.0

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    alert_to_number: str = ""

    @property
    def llm_configured(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def email_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password and self.alert_to_email)

    @property
    def sms_configured(self) -> bool:
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_from_number
            and self.alert_to_number
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
