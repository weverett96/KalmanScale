import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "kalmanscale.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    date TEXT PRIMARY KEY,      -- ISO 8601
    weight REAL NOT NULL,       -- lb
    cal_in REAL,                -- nullable
    cal_out REAL                -- nullable, Whoop
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def upsert_entry(date_str: str, weight: float, cal_in: float | None, cal_out: float | None) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO entries (date, weight, cal_in, cal_out) VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                weight = excluded.weight,
                cal_in = excluded.cal_in,
                cal_out = excluded.cal_out
            """,
            (date_str, weight, cal_in, cal_out),
        )


def delete_entry(date_str: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM entries WHERE date = ?", (date_str,))


def list_entries() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT date, weight, cal_in, cal_out FROM entries ORDER BY date ASC"
        ).fetchall()
    return [
        {"date": r[0], "weight": r[1], "cal_in": r[2], "cal_out": r[3]} for r in rows
    ]
