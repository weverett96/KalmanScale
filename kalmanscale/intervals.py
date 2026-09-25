"""
intervals.icu sync: Garmin Index weight + body fat (wellness) and cycling
ride kcal (activities), via intervals.icu's official personal API key.

Auth is HTTP Basic with username "API_KEY" and the key as password; athlete
id "0" resolves to the key's own athlete.
"""

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

BASE_URL = "https://intervals.icu/api/v1/athlete/0"

# intervals.icu sits behind Cloudflare, which blocks urllib's default
# User-Agent as a bot fingerprint (403 / error code 1010).
USER_AGENT = "KalmanScale/0.1 (personal weight-trend app)"

LB_PER_KG = 2.20462
CM_PER_IN = 2.54

# With a power meter, ride kJ ≈ kcal burned: the 4.184 kJ/kcal conversion
# roughly cancels ~24% gross cycling efficiency, so mechanical work in kJ is
# used directly as kcal when available.
RIDE_TYPES = {
    "Ride",
    "VirtualRide",
    "GravelRide",
    "MountainBikeRide",
    "EBikeRide",
    "EMountainBikeRide",
    "TrackRide",
    "Velomobile",
}


class IntervalsError(Exception):
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


def _api_key() -> str:
    key = os.environ.get("INTERVALS_API_KEY") or _load_env_file(ENV_FILE).get(
        "INTERVALS_API_KEY"
    )
    if not key:
        raise IntervalsError("Missing INTERVALS_API_KEY (shell env or .env).")
    return key


def _get(path: str, params: dict) -> list:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    token = base64.b64encode(f"API_KEY:{_api_key()}".encode()).decode()
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise IntervalsError(f"GET {path} failed ({e.code}): {e.read().decode()}") from e


def parse_wellness(records: list[dict]) -> dict[str, tuple[float, float | None]]:
    """{"YYYY-MM-DD": (weight_lb, body_fat_pct | None)} for days with a
    weight. Garmin sometimes sends bodyFat = 0 on days without a real
    reading, so 0 is treated as missing."""
    out = {}
    for rec in records:
        weight_kg = rec.get("weight")
        if not weight_kg:
            continue
        body_fat = rec.get("bodyFat") or None
        out[rec["id"]] = (round(weight_kg * LB_PER_KG, 1), body_fat)
    return out


def parse_abdomen(records: list[dict]) -> dict[str, float]:
    """{"YYYY-MM-DD": abdomen_in}. intervals.icu returns abdomen in cm
    regardless of display units (verified: 102.87 logged as 40.5 in)."""
    return {
        rec["id"]: round(rec["abdomen"] / CM_PER_IN, 2)
        for rec in records
        if rec.get("abdomen")
    }


def _ride_kcal(activity: dict) -> float | None:
    joules = activity.get("icu_joules")
    if joules:
        return joules / 1000.0
    return activity.get("calories")


def parse_rides(activities: list[dict]) -> dict[str, float]:
    """{"YYYY-MM-DD": total ride kcal} summed over cycling activities by
    local start date. Rides with neither work nor calories are skipped."""
    out = defaultdict(float)
    for act in activities:
        if act.get("type") not in RIDE_TYPES:
            continue
        kcal = _ride_kcal(act)
        if kcal is None:
            continue
        out[act["start_date_local"][:10]] += kcal
    return {day: round(kcal) for day, kcal in out.items()}


def fetch_wellness(oldest: Date, newest: Date) -> list[dict]:
    """Raw wellness records; parse with parse_wellness / parse_abdomen."""
    return _get("/wellness", {"oldest": oldest.isoformat(), "newest": newest.isoformat()})


def fetch_rides(oldest: Date, newest: Date) -> dict[str, float]:
    return parse_rides(
        _get("/activities", {"oldest": oldest.isoformat(), "newest": newest.isoformat()})
    )
