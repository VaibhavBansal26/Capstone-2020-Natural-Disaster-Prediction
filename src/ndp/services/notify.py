"""Outbound alerts: email over SMTP and SMS over Twilio. Both are optional and configured
through environment variables; nothing is hard-coded."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

from ndp.config import get_settings
from ndp.services import ServiceError


def send_email(subject: str, body: str) -> str:
    s = get_settings()
    if not s.email_configured:
        raise ServiceError("Email is not configured.")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = s.alert_from_email or s.smtp_user
    msg["To"] = s.alert_to_email
    msg.set_content(body)
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, context=context, timeout=20) as server:
            server.login(s.smtp_user, s.smtp_password)
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        raise ServiceError(f"Email failed: {exc}") from exc
    return f"email to {s.alert_to_email}"


def send_sms(body: str, to_number: str | None = None) -> str:
    s = get_settings()
    if not s.sms_configured and not (to_number and s.twilio_account_sid and s.twilio_auth_token):
        raise ServiceError("SMS is not configured.")
    try:
        from twilio.rest import Client  # optional dependency: `uv sync --extra alerts`
    except ImportError as exc:
        raise ServiceError("Twilio is not installed; run `uv sync --extra alerts`.") from exc
    target = to_number or s.alert_to_number
    try:
        client = Client(s.twilio_account_sid, s.twilio_auth_token)
        client.messages.create(body=body, from_=s.twilio_from_number, to=target)
    except Exception as exc:  # twilio raises its own hierarchy
        raise ServiceError(f"SMS failed: {exc}") from exc
    return f"sms to {target}"
