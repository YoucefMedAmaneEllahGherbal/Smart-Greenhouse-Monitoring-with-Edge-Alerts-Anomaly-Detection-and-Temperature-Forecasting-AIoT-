"""Application configuration and security utilities for Smart Greenhouse AIoT.

Provides centralized constants, secure credentials retrieval with fail-safes,
sanitized redaction filters, and protected error logging.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any
import streamlit as st

# -----------------------------------------------------------------------------
# Telemetry & Runtime Constants (Immutable defaults, no sidebar controls)
# -----------------------------------------------------------------------------
REFRESH_SECONDS: int = 20
N_READINGS: int = 600       # Historical points fetched (capped at 800)
MAX_RESULTS: int = 800      # Hard ceiling for ThingSpeak results parameter
DISPLAY_TZ: str = "Africa/Algiers"
STALE_THRESHOLD_MINUTES: float = 5.0
HISTORY_WINDOW_MINUTES: int = 800
REQUEST_TIMEOUT_SEC: int = 10

# Exact message required when credentials are missing or invalid
NOT_CONFIGURED_MESSAGE: str = (
    "Set THINGSPEAK_CHANNEL_ID and THINGSPEAK_READ_KEY in .streamlit/secrets.toml "
    "(or as environment variables), then reload."
)

# Strict regex pattern for numeric ThingSpeak channel ID (1 to 12 digits)
CHANNEL_ID_PATTERN = re.compile(r"^\d{1,12}$")


# -----------------------------------------------------------------------------
# Secure Credential Retrieval & Redaction
# -----------------------------------------------------------------------------
def get_secret(name: str) -> str:
    """Retrieve secret from st.secrets, fallback to environment variable, or return empty string.

    Never raises an exception, never exposes traceback or secret value to UI.
    """
    try:
        if name in st.secrets:
            val = st.secrets[name]
            if val is not None:
                return str(val).strip()
    except Exception:
        pass

    try:
        val = os.environ.get(name, "")
        if val is not None:
            return str(val).strip()
    except Exception:
        pass

    return ""


def redact(text: str) -> str:
    """Sanitize read key, channel ID, tokens, and api_key= query parameters with '***'."""
    if not text:
        return ""
    sanitized = str(text)

    # Redact any configured secrets if they appear in text
    for secret_name in ("THINGSPEAK_READ_KEY", "THINGSPEAK_CHANNEL_ID", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        secret_val = get_secret(secret_name)
        if secret_val and len(secret_val) >= 3:
            sanitized = sanitized.replace(secret_val, "***")

    # Generic credential patterns
    sanitized = re.sub(r"(?i)api_key=[^&\s'\"]+", "api_key=***", sanitized)
    sanitized = re.sub(r"(?i)bot\d+:[A-Za-z0-9_-]+", "bot***", sanitized)
    sanitized = re.sub(r"(?i)read_key=[^&\s'\"]+", "read_key=***", sanitized)
    return sanitized


def log_error(exc_or_msg: Any) -> None:
    """Safely log error messages without leaking URLs, credentials, or tracebacks."""
    if isinstance(exc_or_msg, BaseException):
        err_type = type(exc_or_msg).__name__
        sys.stderr.write(f"[ERROR] {err_type}\n")
    else:
        sys.stderr.write(f"[ERROR] {redact(str(exc_or_msg))}\n")
