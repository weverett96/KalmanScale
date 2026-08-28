from datetime import date as Date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, whoop
from .filter import FilterParams, run_filter

app = FastAPI(title="KalmanScale")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class EntryIn(BaseModel):
    date: str  # ISO 8601
    weight: float
    cal_in: float | None = None
    cal_out: float | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "logo.png")


@app.get("/api/entries")
def get_entries():
    return db.list_entries()


@app.post("/api/entries")
def upsert_entry(entry: EntryIn):
    db.upsert_entry(entry.date, entry.weight, entry.cal_in, entry.cal_out)
    return {"ok": True}


@app.delete("/api/entries/{date_str}")
def delete_entry(date_str: str):
    db.delete_entry(date_str)
    return {"ok": True}


@app.get("/api/filter")
def get_filter():
    rows = db.list_entries()
    entries = [
        {
            "date": Date.fromisoformat(r["date"]),
            "weight": r["weight"],
            "cal_in": r["cal_in"],
            "cal_out": r["cal_out"],
        }
        for r in rows
    ]
    results = run_filter(entries, FilterParams())
    return {"trajectory": results, "latest": results[-1] if results else None}


@app.post("/api/whoop/sync")
def whoop_sync():
    """
    Backfills cal_out for existing entries that are missing it, using
    recent *closed* Whoop cycles only (never today's still-in-progress
    cycle — see whoop.fetch_recent_closed_cycles). Never creates rows and
    never overwrites a cal_out that's already set.
    """
    try:
        by_date = whoop.fetch_recent_closed_cycles()
    except whoop.WhoopAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))

    updated = []
    for row in db.list_entries():
        if row["cal_out"] is not None:
            continue
        kcal = by_date.get(row["date"])
        if kcal is None:
            continue
        if db.backfill_cal_out(row["date"], kcal):
            updated.append({"date": row["date"], "cal_out": kcal})

    return {"updated": updated}
