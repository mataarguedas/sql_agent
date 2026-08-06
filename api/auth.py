"""Single-user auth: password + TOTP second factor, session-cookie gated.

Two independent factors are required to open a session: a bcrypt-hashed
password and a TOTP code from an authenticator app (e.g. Google
Authenticator). Both live only in env vars — see ``scripts/setup_auth.py``
to provision them. There is no user table; this site has exactly one user.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import secrets
import time

import bcrypt
import pyotp
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED

load_dotenv()

logger = logging.getLogger(__name__)

_SESSION_KEY = "authenticated"
_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60  # 12 hours

_MAX_LOGIN_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300  # 5 minutes
_login_attempts: dict[str, list[float]] = {}

_ephemeral_session_secret: str | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Run `python -m scripts.setup_auth` and add the "
            "printed values to your .env."
        )
    return value


def session_secret_key() -> str:
    """The key sessions are signed with.

    Read lazily (not required at import time, unlike the credential env vars)
    so the app — and the test suite that imports it — can still start before
    ``scripts/setup_auth.py`` has been run. Falls back to a key generated
    once per process: sessions from that process won't survive a restart, so
    real deployments should set ``SESSION_SECRET_KEY`` in ``.env``.
    """
    global _ephemeral_session_secret
    configured = os.getenv("SESSION_SECRET_KEY")
    if configured:
        return configured
    if _ephemeral_session_secret is None:
        _ephemeral_session_secret = secrets.token_urlsafe(32)
        logger.warning(
            "SESSION_SECRET_KEY not set; using a one-off key for this process. "
            "Sessions will not survive a restart. Run `python -m scripts.setup_auth` "
            "and set SESSION_SECRET_KEY in .env for real deployments."
        )
    return _ephemeral_session_secret


def cookie_is_secure() -> bool:
    """Whether the session cookie requires HTTPS; disable only for local http testing."""
    return os.getenv("AUTH_COOKIE_SECURE", "true").strip().lower() not in ("0", "false", "no")


def session_max_age_seconds() -> int:
    return _SESSION_MAX_AGE_SECONDS


def rate_limited(client_key: str) -> bool:
    """True if ``client_key`` has hit the login attempt ceiling within the window."""
    now = time.monotonic()
    attempts = [t for t in _login_attempts.get(client_key, []) if now - t < _LOGIN_WINDOW_SECONDS]
    _login_attempts[client_key] = attempts
    return len(attempts) >= _MAX_LOGIN_ATTEMPTS


def record_failed_attempt(client_key: str) -> None:
    _login_attempts.setdefault(client_key, []).append(time.monotonic())


def clear_attempts(client_key: str) -> None:
    _login_attempts.pop(client_key, None)


def _decode_password_hash(encoded: str) -> bytes:
    """Base64-decode ``AUTH_PASSWORD_HASH_B64`` back into a raw bcrypt hash.

    The hash is stored base64-encoded, not as the raw ``$2b$12$...`` string,
    because Docker Compose interpolates ``$`` in env files (both the
    project-root ``.env`` and anything loaded via ``env_file:``) and silently
    blanks out anything that looks like ``$name`` — a near-certainty in a
    bcrypt hash's random salt/hash bytes. That corrupted every login with an
    unhandled ValueError from bcrypt (surfaced as a 500). Base64 has no ``$``
    in its alphabet, so it survives Compose untouched either way.
    """
    try:
        return base64.b64decode(encoded.encode("utf-8"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            "AUTH_PASSWORD_HASH_B64 is not valid base64. Re-run "
            "`python -m scripts.setup_auth` and paste its output as-is."
        ) from exc


def verify_credentials(username: str, password: str, totp_code: str) -> bool:
    """Check username, bcrypt password, and TOTP code. All three must match."""
    expected_username = _require_env("AUTH_USERNAME")
    password_hash = _decode_password_hash(_require_env("AUTH_PASSWORD_HASH_B64"))
    totp_secret = _require_env("AUTH_TOTP_SECRET")

    if not hmac.compare_digest(username, expected_username):
        return False
    if not bcrypt.checkpw(password.encode("utf-8"), password_hash):
        return False
    return pyotp.TOTP(totp_secret).verify(totp_code, valid_window=1)


def is_authenticated(request: Request) -> bool:
    return bool(request.session.get(_SESSION_KEY))


def log_in(request: Request) -> None:
    request.session[_SESSION_KEY] = True


def log_out(request: Request) -> None:
    request.session.clear()


def require_session(request: Request) -> None:
    """FastAPI dependency for JSON/API routes: 401 if there's no authenticated session."""
    if not is_authenticated(request):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Not authenticated")
