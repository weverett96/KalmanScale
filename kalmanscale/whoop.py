"""
Manual Whoop sync for the MVP: fetch the most recent physiological cycle's
energy expenditure. No cron/scheduling here — this is called on demand from
the "Sync from Whoop" button (see plans/whoop_api_notes.md for the full
API research; the real Milestone 5 cron job is a separate future task).
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_ROOT / ".whoop_tokens.json"
ENV_FILE = PROJECT_ROOT / ".env"

TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
CYCLE_URL = "https://api.prod.whoop.com/developer/v2/cycle"

# Whoop's API sits behind Cloudflare, which blocks the default urllib
# User-Agent as a bot fingerprint (403 / error code 1010).
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

KJ_PER_KCAL = 4.184


class WhoopAuthError(Exception):
    pass


def _load_env_file(path: Path) -> dict:
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


def _client_credentials() -> tuple[str, str]:
    env = _load_env_file(ENV_FILE)
    client_id = os.environ.get("WHOOP_CLIENT_ID") or env.get("WHOOP_CLIENT_ID")
    client_secret = os.environ.get("WHOOP_CLIENT_SECRET") or env.get("WHOOP_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise WhoopAuthError(
            "Missing WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET (shell env or .env)."
        )
    return client_id, client_secret


def _load_tokens() -> dict:
    if not TOKEN_FILE.exists():
        raise WhoopAuthError(
            f"No {TOKEN_FILE.name} found — run scripts/whoop_auth.py first."
        )
    return json.loads(TOKEN_FILE.read_text())


def _save_tokens(tokens: dict) -> None:
    TOKEN_FILE.write_text(json.dumps(tokens, indent=2))
    TOKEN_FILE.chmod(0o600)


def _refresh(tokens: dict) -> dict:
    client_id, client_secret = _client_credentials()
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode()
    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req) as resp:
            new_tokens = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise WhoopAuthError(
            f"Refresh failed ({e.code}): {e.read().decode()}. "
            "The refresh token may be stale — re-run scripts/whoop_auth.py."
        ) from e
    # Whoop rotates refresh tokens: the old one is invalidated the moment
    # this call succeeds, so the new one must be persisted immediately.
    _save_tokens(new_tokens)
    return new_tokens


def _get(url: str, access_token: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def fetch_latest_cycle_kcal() -> dict:
    """
    Returns {"kcal": float|None, "score_state": str, "date": "YYYY-MM-DD"}
    for the most recent physiological cycle. kcal is None if the cycle
    isn't fully scored yet (score_state != "SCORED") — caller should not
    treat that as a hard failure, just tell the user to retry later.
    """
    tokens = _load_tokens()
    url = f"{CYCLE_URL}?limit=1"
    try:
        data = _get(url, tokens["access_token"])
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        tokens = _refresh(tokens)
        data = _get(url, tokens["access_token"])

    records = data.get("records", [])
    if not records:
        raise WhoopAuthError("No cycles returned from Whoop API.")

    cycle = records[0]
    score_state = cycle.get("score_state")
    cycle_date = cycle["start"][:10]

    if score_state != "SCORED":
        return {"kcal": None, "score_state": score_state, "date": cycle_date}

    kcal = cycle["score"]["kilojoule"] / KJ_PER_KCAL
    return {"kcal": kcal, "score_state": score_state, "date": cycle_date}
