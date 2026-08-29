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


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
    if "body_fat_pct" not in columns:
        conn.execute("ALTER TABLE entries ADD COLUMN body_fat_pct REAL")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    _migrate(conn)
    return conn


def upsert_entry(
    date_str: str,
    weight: float,
    cal_in: float | None,
    cal_out: float | None,
    body_fat_pct: float | None = None,
) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO entries (date, weight, cal_in, cal_out, body_fat_pct)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                weight = excluded.weight,
                cal_in = excluded.cal_in,
                cal_out = excluded.cal_out,
                body_fat_pct = excluded.body_fat_pct
            """,
            (date_str, weight, cal_in, cal_out, body_fat_pct),
        )


def backfill_cal_out(date_str: str, cal_out: float) -> bool:
    """Fill cal_out only for an existing row where it's currently NULL.
    Never overwrites an already-set value, never creates a new row.
    Returns True if a row was actually updated."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE entries SET cal_out = ? WHERE date = ? AND cal_out IS NULL",
            (cal_out, date_str),
        )
        return cur.rowcount > 0


def delete_entry(date_str: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM entries WHERE date = ?", (date_str,))


def list_entries() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT date, weight, cal_in, cal_out, body_fat_pct FROM entries ORDER BY date ASC"
        ).fetchall()
    return [
        {
            "date": r[0],
            "weight": r[1],
            "cal_in": r[2],
            "cal_out": r[3],
            "body_fat_pct": r[4],
        }
        for r in rows
    ]
