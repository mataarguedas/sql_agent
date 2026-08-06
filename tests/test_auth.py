"""Unit tests for api/auth.py: credential verification, rate limiting, session gating.

End-to-end login/logout behavior through the FastAPI app lives in
test_api.py; this file exercises the auth primitives in isolation.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main as main
from api import auth
from tests.conftest import (
    TEST_AUTH_PASSWORD,
    TEST_AUTH_PASSWORD_HASH_B64,
    TEST_AUTH_USERNAME,
    current_totp_code,
)

# --------------------------------------------------------------------------- #
# Regression: Docker Compose interpolates "$" in .env files (both the
# project-root .env and anything loaded via env_file:) and silently blanks
# out anything matching "$name" -- near-certain to appear somewhere in a
# bcrypt hash's random salt/hash bytes. That corrupted every login with an
# unhandled ValueError from bcrypt, surfaced to users as a 500. Storing the
# hash base64-encoded (AUTH_PASSWORD_HASH_B64) sidesteps it: base64's
# alphabet has no "$". These tests pin that invariant so it can't regress.
# --------------------------------------------------------------------------- #

_REAL_BCRYPT_HASH_THAT_TRIGGERED_THE_BUG = (
    "$2b$12$mEb.cTUQvD2qAG83d7GJC.1sUqP2xdmXGuFN1jSK9nCHRc3UF8nTy"
)


def _simulate_compose_dollar_interpolation(value: str) -> str:
    """Approximate Compose's env-file interpolation: blank out $name references."""
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", "", value)


def test_a_raw_bcrypt_hash_is_corrupted_by_compose_style_interpolation() -> None:
    mangled = _simulate_compose_dollar_interpolation(_REAL_BCRYPT_HASH_THAT_TRIGGERED_THE_BUG)
    assert mangled != _REAL_BCRYPT_HASH_THAT_TRIGGERED_THE_BUG


def test_the_base64_encoded_form_survives_the_same_interpolation_untouched() -> None:
    encoded = base64.b64encode(_REAL_BCRYPT_HASH_THAT_TRIGGERED_THE_BUG.encode()).decode()
    assert _simulate_compose_dollar_interpolation(encoded) == encoded


def test_password_hash_env_var_never_contains_a_dollar_sign() -> None:
    assert "$" not in os.environ["AUTH_PASSWORD_HASH_B64"]
    assert "$" not in TEST_AUTH_PASSWORD_HASH_B64


# --------------------------------------------------------------------------- #
# verify_credentials
# --------------------------------------------------------------------------- #


def test_verify_credentials_accepts_the_right_username_password_and_code() -> None:
    assert auth.verify_credentials(TEST_AUTH_USERNAME, TEST_AUTH_PASSWORD, current_totp_code())


def test_verify_credentials_rejects_wrong_username() -> None:
    assert not auth.verify_credentials("someone-else", TEST_AUTH_PASSWORD, current_totp_code())


def test_verify_credentials_rejects_wrong_password() -> None:
    assert not auth.verify_credentials(TEST_AUTH_USERNAME, "wrong password", current_totp_code())


def test_verify_credentials_rejects_wrong_totp_code() -> None:
    assert not auth.verify_credentials(TEST_AUTH_USERNAME, TEST_AUTH_PASSWORD, "000000")


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def test_rate_limiting_kicks_in_after_max_attempts() -> None:
    key = "203.0.113.5"
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        assert not auth.rate_limited(key)
        auth.record_failed_attempt(key)
    assert auth.rate_limited(key)


def test_clear_attempts_resets_the_limiter() -> None:
    key = "203.0.113.9"
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        auth.record_failed_attempt(key)
    assert auth.rate_limited(key)

    auth.clear_attempts(key)

    assert not auth.rate_limited(key)


def test_rate_limiting_is_scoped_per_client_key() -> None:
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        auth.record_failed_attempt("198.51.100.1")
    assert auth.rate_limited("198.51.100.1")
    assert not auth.rate_limited("198.51.100.2")


# --------------------------------------------------------------------------- #
# Session dependency
# --------------------------------------------------------------------------- #


class _FakeRequest:
    """Minimal stand-in for ``fastapi.Request`` exposing only ``.session``."""

    def __init__(self, session: dict[str, Any] | None = None) -> None:
        self.session = session if session is not None else {}


def test_require_session_rejects_a_signed_out_request() -> None:
    with pytest.raises(HTTPException) as exc_info:
        auth.require_session(_FakeRequest())  # type: ignore[arg-type]
    assert exc_info.value.status_code == 401


def test_require_session_allows_an_authenticated_request() -> None:
    auth.require_session(_FakeRequest({"authenticated": True}))  # type: ignore[arg-type]


def test_log_in_and_log_out_toggle_is_authenticated() -> None:
    request = _FakeRequest()
    assert not auth.is_authenticated(request)  # type: ignore[arg-type]

    auth.log_in(request)  # type: ignore[arg-type]
    assert auth.is_authenticated(request)  # type: ignore[arg-type]

    auth.log_out(request)  # type: ignore[arg-type]
    assert not auth.is_authenticated(request)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# End-to-end login/logout through the app
# --------------------------------------------------------------------------- #


def _login(client: TestClient, *, password: str = TEST_AUTH_PASSWORD, code: str | None = None):
    return client.post(
        "/login",
        data={
            "username": TEST_AUTH_USERNAME,
            "password": password,
            "code": code if code is not None else current_totp_code(),
        },
        follow_redirects=False,
    )


def test_login_page_is_reachable_when_signed_out() -> None:
    response = TestClient(main.app).get("/login")
    assert response.status_code == 200
    assert "form" in response.text


def test_wrong_password_redirects_back_to_login_with_an_error() -> None:
    response = _login(TestClient(main.app), password="not the password")
    assert response.status_code == 303
    assert response.headers["location"] == "/login?error=invalid"


def test_wrong_totp_code_redirects_back_to_login_with_an_error() -> None:
    response = _login(TestClient(main.app), code="000000")
    assert response.status_code == 303
    assert response.headers["location"] == "/login?error=invalid"


def test_correct_credentials_start_a_session_that_can_reach_the_app() -> None:
    client = TestClient(main.app)
    login_response = _login(client)
    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/"

    assert client.get("/", follow_redirects=False).status_code == 200


def test_repeated_failures_lock_out_further_attempts() -> None:
    client = TestClient(main.app)
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        _login(client, password="wrong")

    response = _login(client)  # correct creds, but the client key is now rate-limited
    assert response.status_code == 303
    assert response.headers["location"] == "/login?error=rate_limited"


def test_logout_ends_the_session() -> None:
    client = TestClient(main.app)
    _login(client)
    assert client.get("/", follow_redirects=False).status_code == 200

    client.post("/logout")

    assert client.get("/", follow_redirects=False).status_code == 303
