#!/usr/bin/env python3
"""
One-time OAuth2 bootstrap for the Whoop API.

Runs the authorization-code flow locally: opens the Whoop consent page in
your browser, catches the redirect on localhost, exchanges the code for an
access + refresh token, and saves them to .whoop_tokens.json (gitignored).

Requires WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET, either exported in the
shell or set in a .env file in the project root.

Usage:
    python scripts/whoop_auth.py
"""

import http.server
import json
import secrets
import sys
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
REDIRECT_URI = "http://localhost:8000/auth/whoop/callback"
SCOPES = "offline read:cycles read:recovery read:sleep read:workout read:body_measurement read:profile"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_ROOT / ".whoop_tokens.json"
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_credentials() -> tuple[str, str]:
    import os

    env = load_env_file(ENV_FILE)
    client_id = os.environ.get("WHOOP_CLIENT_ID") or env.get("WHOOP_CLIENT_ID")
    client_secret = os.environ.get("WHOOP_CLIENT_SECRET") or env.get("WHOOP_CLIENT_SECRET")

    if not client_id or not client_secret:
        sys.exit(
            "Missing WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET.\n"
            f"Set them in the shell or in {ENV_FILE}."
        )
    return client_id, client_secret


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    # populated by run_local_server before serving
    expected_state = None
    result = {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/whoop/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]

        if error:
            CallbackHandler.result["error"] = error
            body = f"Authorization failed: {error}. You can close this tab."
        elif state != CallbackHandler.expected_state:
            CallbackHandler.result["error"] = "state_mismatch"
            body = "State mismatch — possible CSRF, aborting. You can close this tab."
        elif not code:
            CallbackHandler.result["error"] = "no_code"
            body = "No authorization code received. You can close this tab."
        else:
            CallbackHandler.result["code"] = code
            body = "Authorization received. You can close this tab and return to the terminal."

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass  # quiet


def run_local_server(expected_state: str) -> dict:
    CallbackHandler.expected_state = expected_state
    CallbackHandler.result = {}
    server = http.server.HTTPServer(("localhost", 8000), CallbackHandler)
    # Handle exactly one request (the redirect), then stop.
    server.timeout = 120
    server.handle_request()
    return CallbackHandler.result


def exchange_code(code: str, client_id: str, client_secret: str) -> dict:
    data = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode()

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"Token exchange failed ({e.code}): {e.read().decode()}")


def main():
    client_id, client_secret = get_credentials()
    state = secrets.token_urlsafe(16)

    auth_params = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "state": state,
        }
    )
    auth_url = f"{AUTH_URL}?{auth_params}"

    print("Opening browser for Whoop authorization...")
    print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    print("Waiting for redirect on http://localhost:8000/auth/whoop/callback ...")
    result = run_local_server(state)

    if "error" in result:
        sys.exit(f"Authorization failed: {result['error']}")
    if "code" not in result:
        sys.exit("Timed out waiting for authorization redirect.")

    print("Exchanging code for tokens...")
    tokens = exchange_code(result["code"], client_id, client_secret)

    TOKEN_FILE.write_text(json.dumps(tokens, indent=2))
    TOKEN_FILE.chmod(0o600)

    print(f"Saved tokens to {TOKEN_FILE} (gitignored).")
    print(f"Access token expires in {tokens.get('expires_in')}s; refresh token has no fixed expiry (offline scope).")


if __name__ == "__main__":
    main()
