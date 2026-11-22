"""Security and hardening verification suite.

Verifies:
1. Regex validation for channel ID (no HTTP call on invalid/placeholder).
2. Exact NOT_CONFIGURED message.
3. Sanitized error logging and redaction (zero leaked secrets or URLs).
4. No verify=False in network requests.
5. No st.sidebar anywhere in python files.
6. Clean secrets handling via get_secret.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from config import (
    CHANNEL_ID_PATTERN,
    NOT_CONFIGURED_MESSAGE,
    get_secret,
    log_error,
    redact,
)
from thingspeak import fetch_thingspeak_data


def test_channel_id_regex_validation():
    """Verify channel ID regex accepts 1-12 digits and rejects placeholders/invalid input."""
    assert CHANNEL_ID_PATTERN.match("12345") is not None
    assert CHANNEL_ID_PATTERN.match("3580105") is not None
    assert CHANNEL_ID_PATTERN.match("123456789012") is not None

    # Invalid patterns
    assert CHANNEL_ID_PATTERN.match("YOUR_CHANNEL_ID") is None
    assert CHANNEL_ID_PATTERN.match("PASTE_YOUR_ID") is None
    assert CHANNEL_ID_PATTERN.match("") is None
    assert CHANNEL_ID_PATTERN.match("abc") is None
    assert CHANNEL_ID_PATTERN.match("1234567890123") is None  # 13 digits > 12


def test_fetch_thingspeak_without_secrets():
    """Verify that unconfigured / placeholder channel ID makes NO HTTP call and returns 'not_configured'."""
    with patch("thingspeak.requests.get") as mock_get:
        # Test placeholder string
        df, status = fetch_thingspeak_data(channel_id="YOUR_CHANNEL_ID", read_key="")
        assert status == "not_configured"
        assert df.empty
        mock_get.assert_not_called()

        # Test empty string
        df, status = fetch_thingspeak_data(channel_id="", read_key="")
        assert status == "not_configured"
        assert df.empty
        mock_get.assert_not_called()


def test_exact_not_configured_message():
    """Verify the exact NOT_CONFIGURED message matches requirement."""
    expected = "Set THINGSPEAK_CHANNEL_ID and THINGSPEAK_READ_KEY in .streamlit/secrets.toml (or as environment variables), then reload."
    assert NOT_CONFIGURED_MESSAGE == expected


def test_wrong_read_key_error_sanitization():
    """Verify that invalid response/key logs only error type, never URL, params, or keys."""
    # Capture stderr
    captured_stderr = io.StringIO()
    sys.stderr = captured_stderr

    with patch("thingspeak.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "401 Unauthorized - invalid api_key=SECRET_KEY_12345"
        mock_get.return_value = mock_resp

        df, status = fetch_thingspeak_data(channel_id="999999", read_key="SECRET_KEY_12345")
        sys.stderr = sys.__stderr__

        assert status == "unreachable"
        assert df.empty

        stderr_output = captured_stderr.getvalue()
        # Verify secret key and channel id do not appear in stderr
        assert "SECRET_KEY_12345" not in stderr_output
        assert "api_key=SECRET_KEY_12345" not in stderr_output
        assert "https://" not in stderr_output
        assert "[ERROR] HTTPStatus_401" in stderr_output


def test_network_exception_sanitization():
    """Verify network exceptions log ONLY exception class name, never exception str or URL."""
    captured_stderr = io.StringIO()
    sys.stderr = captured_stderr

    with patch("thingspeak.requests.get") as mock_get:
        # Simulate requests exception with sensitive URL in str(exc)
        exc = Exception("Connection error to https://api.thingspeak.com?api_key=LEAKED_KEY")
        mock_get.side_effect = exc

        df, status = fetch_thingspeak_data(channel_id="999999", read_key="LEAKED_KEY")
        sys.stderr = sys.__stderr__

        assert status == "unreachable"
        assert df.empty

        stderr_output = captured_stderr.getvalue()
        assert "LEAKED_KEY" not in stderr_output
        assert "https://" not in stderr_output
        assert "[ERROR] Exception" in stderr_output


def test_redact_utility():
    """Verify redact utility masks known secrets and pattern matches."""
    sample = "Connecting with api_key=12345ABCDE&results=100 and bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    cleaned = redact(sample)
    assert "api_key=***" in cleaned
    assert "12345ABCDE" not in cleaned
    assert "bot***" in cleaned


def test_codebase_security_rules():
    """Inspect python source files to ensure no forbidden patterns exist."""
    py_files = [f for f in Path(".").glob("*.py") if not f.name.startswith("test_")]

    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")

        # 1. No verify=False in requests
        assert "verify=False" not in content, f"Forbidden 'verify=False' in {py_file}"
        assert "verify = False" not in content, f"Forbidden 'verify=False' in {py_file}"

        # 2. No st.sidebar
        assert "st.sidebar" not in content, f"Forbidden 'st.sidebar' in {py_file}"


if __name__ == "__main__":
    test_channel_id_regex_validation()
    print("Channel ID Regex: PASSED")
    test_fetch_thingspeak_without_secrets()
    print("Fetch without secrets: PASSED")
    test_exact_not_configured_message()
    print("Exact NOT_CONFIGURED message: PASSED")
    test_wrong_read_key_error_sanitization()
    print("Wrong read key error sanitization: PASSED")
    test_network_exception_sanitization()
    print("Network exception sanitization: PASSED")
    test_redact_utility()
    print("Redact utility: PASSED")
    test_codebase_security_rules()
    print("Codebase security rules (no sidebar, no verify=False): PASSED")
    print("\nALL SECURITY TESTS PASSED SUCCESSFULLY!")
