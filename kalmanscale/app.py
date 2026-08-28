from datetime import date as Date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db, whoop
from .filter import FilterParams, run_filter

app = FastAPI(title="KalmanScale")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class EntryIn(BaseModel):
    date: str  # ISO 8601
    weight: float
    cal_in: float | None = None
    cal_out: float | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


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
    try:
        return whoop.fetch_latest_cycle_kcal()
    except whoop.WhoopAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
