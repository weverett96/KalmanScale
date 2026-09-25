from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, intervals
from .filter import (
    FilterParams,
    forecast_ride_kcal,
    navy_body_fat_pct,
    projected_trend,
    run_filter,
)

app = FastAPI(title="KalmanScale")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SettingsIn(BaseModel):
    height_in: float
    neck_in: float


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "logo.png")


@app.get("/api/entries")
def get_entries():
    return db.list_entries()


@app.get("/api/rides")
def get_rides():
    return db.list_rides()


@app.get("/api/tape")
def get_tape():
    return db.list_tape()


@app.get("/api/settings")
def get_settings():
    return db.get_settings()


@app.put("/api/settings")
def put_settings(settings: SettingsIn):
    db.set_settings(settings.model_dump())
    return db.get_settings()


@app.get("/api/filter")
def get_filter():
    settings = db.get_settings()
    tape = db.list_tape()
    entries = []
    for r in db.list_entries():
        abdomen = tape.get(r["date"])
        entries.append(
            {
                "date": Date.fromisoformat(r["date"]),
                "weight": r["weight"],
                "body_fat_pct": r["body_fat_pct"],
                "tape_bf_pct": navy_body_fat_pct(
                    abdomen, settings["neck_in"], settings["height_in"]
                )
                if abdomen is not None
                else None,
            }
        )
    rides = {Date.fromisoformat(d): kcal for d, kcal in db.list_rides().items()}
    results = run_filter(entries, rides, FilterParams())
    if not results:
        return {"trajectory": [], "latest": None}

    # Forecast from the rides the latest weigh-in has already seen (through
    # the day before it), over the span the filter has data for.
    latest = dict(results[-1])
    ride_kcal = forecast_ride_kcal(
        rides, entries[0]["date"], entries[-1]["date"] - timedelta(days=1)
    )
    if ride_kcal is not None:
        latest["ride_kcal_forecast"] = ride_kcal
        for part in ("", "_fat", "_lean"):
            trend, se_trend = projected_trend(latest, ride_kcal, part)
            latest[f"trend{part}"] = trend
            latest[f"se_trend{part}"] = se_trend
    return {"trajectory": results, "latest": latest}


@app.post("/api/intervals/sync")
def intervals_sync(days: int = 30):
    """
    Pulls the last `days` days (through today) from intervals.icu.
    Wellness weight/body fat is upserted into entries — intervals.icu is
    the source of truth for any date it returns. Ride rows in the range
    are replaced wholesale.
    """
    newest = Date.today()
    oldest = newest - timedelta(days=days - 1)
    try:
        records = intervals.fetch_wellness(oldest, newest)
        rides = intervals.fetch_rides(oldest, newest)
    except intervals.IntervalsError as e:
        raise HTTPException(status_code=400, detail=str(e))

    wellness = intervals.parse_wellness(records)
    tape = intervals.parse_abdomen(records)
    for day, (weight, body_fat_pct) in wellness.items():
        db.upsert_entry(day, weight, body_fat_pct)
    db.replace_rides(oldest.isoformat(), newest.isoformat(), rides)
    db.replace_tape(oldest.isoformat(), newest.isoformat(), tape)

    return {
        "oldest": oldest.isoformat(),
        "newest": newest.isoformat(),
        "weigh_ins": len(wellness),
        "ride_days": len(rides),
        "tape_days": len(tape),
    }
