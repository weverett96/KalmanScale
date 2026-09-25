import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "kalmanscale.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    date TEXT PRIMARY KEY,      -- ISO 8601
    weight REAL NOT NULL,       -- lb
    body_fat_pct REAL           -- nullable, Garmin Index
);
CREATE TABLE IF NOT EXISTS rides (
    date TEXT PRIMARY KEY,      -- ISO 8601, local start date
    kcal REAL NOT NULL          -- summed over that day's rides
);
CREATE TABLE IF NOT EXISTS tape (
    date TEXT PRIMARY KEY,      -- ISO 8601
    abdomen_in REAL NOT NULL    -- at the navel
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,       -- no history: one current value per key
    value REAL NOT NULL
);
INSERT OR IGNORE INTO settings (key, value) VALUES ('height_in', 64.5), ('neck_in', 16.5);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
    if "body_fat_pct" not in columns:
        conn.execute("ALTER TABLE entries ADD COLUMN body_fat_pct REAL")
    # Intake/Whoop expenditure were dropped in favor of intervals.icu ride kcal.
    for dropped in ("cal_in", "cal_out"):
        if dropped in columns:
            conn.execute(f"ALTER TABLE entries DROP COLUMN {dropped}")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def upsert_entry(date_str: str, weight: float, body_fat_pct: float | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO entries (date, weight, body_fat_pct)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                weight = excluded.weight,
                body_fat_pct = excluded.body_fat_pct
            """,
            (date_str, weight, body_fat_pct),
        )


def list_entries() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT date, weight, body_fat_pct FROM entries ORDER BY date ASC"
        ).fetchall()
    return [{"date": r[0], "weight": r[1], "body_fat_pct": r[2]} for r in rows]


def _replace_range(table: str, column: str, oldest: str, newest: str, by_date: dict) -> None:
    """Replace all rows of `table` in [oldest, newest] with by_date, so
    values deleted or edited upstream are corrected on the next sync."""
    with _conn() as conn:
        conn.execute(f"DELETE FROM {table} WHERE date BETWEEN ? AND ?", (oldest, newest))
        conn.executemany(
            f"INSERT INTO {table} (date, {column}) VALUES (?, ?)", sorted(by_date.items())
        )


def _list_by_date(table: str, column: str) -> dict[str, float]:
    with _conn() as conn:
        rows = conn.execute(f"SELECT date, {column} FROM {table} ORDER BY date ASC").fetchall()
    return {r[0]: r[1] for r in rows}


def replace_rides(oldest: str, newest: str, kcal_by_date: dict[str, float]) -> None:
    _replace_range("rides", "kcal", oldest, newest, kcal_by_date)


def list_rides() -> dict[str, float]:
    return _list_by_date("rides", "kcal")


def replace_tape(oldest: str, newest: str, abdomen_by_date: dict[str, float]) -> None:
    _replace_range("tape", "abdomen_in", oldest, newest, abdomen_by_date)


def list_tape() -> dict[str, float]:
    return _list_by_date("tape", "abdomen_in")


def get_settings() -> dict[str, float]:
    with _conn() as conn:
        return dict(conn.execute("SELECT key, value FROM settings").fetchall())


def set_settings(values: dict[str, float]) -> None:
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            values.items(),
        )
