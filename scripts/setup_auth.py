"""One-time interactive provisioning for the site login (see api/auth.py).

Run this yourself, locally: ``python -m scripts.setup_auth``. It prompts for
a new password (hidden input, never printed or logged) and generates a fresh
TOTP secret for Google Authenticator — there is no way to read an existing
authenticator app's code from here or anywhere else; TOTP secrets never
leave the device they were enrolled on. This script only ever creates a new
one, which you enroll by scanning the printed QR code.

It prints the values to put in your .env; it does not write .env itself, so
it can never clobber unrelated settings already in that file.
"""

from __future__ import annotations

import base64
import getpass
import secrets
import sys

import bcrypt
import pyotp
import qrcode

USERNAME = "mataarguedas"
ISSUER = "SQL Analyst Agent"
ACCOUNT_LABEL = "mataarguedass@gmail.com"
QR_PATH = "auth_qr_code.png"
MIN_PASSWORD_LENGTH = 12


def main() -> None:
    password = getpass.getpass("Choose a login password (min 12 chars): ")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password != getpass.getpass("Confirm password: "):
        sys.exit("Passwords did not match.")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    # Base64-encoded, not the raw "$2b$12$..." string: Docker Compose
    # interpolates "$" in .env files (both the project-root .env and
    # anything loaded via env_file:) and silently blanks out anything that
    # looks like "$name" -- near-certain to appear in a bcrypt hash's random
    # bytes. Base64's alphabet has no "$", so it survives untouched.
    password_hash_b64 = base64.b64encode(password_hash).decode("utf-8")
    totp_secret = pyotp.random_base32()
    session_secret = secrets.token_urlsafe(32)

    uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(name=ACCOUNT_LABEL, issuer_name=ISSUER)
    qrcode.make(uri).save(QR_PATH)

    print(f"\nScan {QR_PATH} with Google Authenticator (Add account -> Scan QR code),")
    print("then delete the file - it's a plaintext copy of your TOTP secret.")
    print(f"Manual entry key, if you'd rather type it in: {totp_secret}")

    print("\nAdd these lines to your .env (do not commit them):\n")
    print(f"AUTH_USERNAME={USERNAME}")
    print(f"AUTH_PASSWORD_HASH_B64={password_hash_b64}")
    print(f"AUTH_TOTP_SECRET={totp_secret}")
    print(f"SESSION_SECRET_KEY={session_secret}")
    print("AUTH_COOKIE_SECURE=true  # set to false only for local http testing")


if __name__ == "__main__":
    main()
