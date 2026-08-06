"""Shared pytest fixtures.

``db.connection`` calls ``load_dotenv()`` at import time, so a developer's real
``.env`` leaks into every test process. With tracing enabled there, simply
running the suite exports junk runs to the production LangSmith project —
noise in the exact dashboard the tracing feature exists to make readable.
Tests should never talk to LangSmith, so tracing is forced off for all of them.
"""

from __future__ import annotations

import base64
import os

import bcrypt
import pyotp
import pytest

# api.main builds its SessionMiddleware (secret key, cookie Secure flag) at
# import time, before any per-test fixture runs. TestClient talks plain
# http, so a Secure-flagged cookie would never be echoed back on later
# requests. These must be real env vars, set before that first import.
os.environ.setdefault("SESSION_SECRET_KEY", "test-session-secret")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

from api import auth  # noqa: E402 - must follow the os.environ.setdefault calls above

_TRACING_ENV_VARS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",  # deprecated v1 flag; left set, it breaks langchain-core
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)


@pytest.fixture(autouse=True)
def _disable_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every test to run untraced, whatever the ambient ``.env`` says."""
    for var in _TRACING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# Auth fixtures — a fixed, valid credential set for every test process.
# --------------------------------------------------------------------------- #

TEST_AUTH_USERNAME = "mataarguedas"
TEST_AUTH_PASSWORD = "correct horse battery staple"
TEST_AUTH_PASSWORD_HASH_B64 = base64.b64encode(
    bcrypt.hashpw(TEST_AUTH_PASSWORD.encode("utf-8"), bcrypt.gensalt())
).decode("utf-8")
TEST_AUTH_TOTP_SECRET = pyotp.random_base32()


def current_totp_code() -> str:
    """A currently-valid TOTP code for ``TEST_AUTH_TOTP_SECRET``."""
    return pyotp.TOTP(TEST_AUTH_TOTP_SECRET).now()


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire the fixed test credentials into env, and reset the login rate limiter.

    The rate limiter is module-level state in ``api.auth`` keyed by client
    host; ``TestClient`` always reports the same host, so failed-login tests
    would otherwise bleed into unrelated tests running later in the process.
    """
    monkeypatch.setenv("AUTH_USERNAME", TEST_AUTH_USERNAME)
    monkeypatch.setenv("AUTH_PASSWORD_HASH_B64", TEST_AUTH_PASSWORD_HASH_B64)
    monkeypatch.setenv("AUTH_TOTP_SECRET", TEST_AUTH_TOTP_SECRET)
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-session-secret")
    auth._login_attempts.clear()
    yield
    auth._login_attempts.clear()
