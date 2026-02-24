"""
SQLite trade store for incremental ingestion.
Handles insert with dedup, ingestion logging, and full reads.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "trades.db"

# All trade dict keys that map to DB columns (excluding source_file/ingested_at metadata)
TRADE_COLUMNS = [
    "id", "instrument", "instrumentFull", "strategy", "subStrategy",
    "direction", "qty", "entryPrice", "exitPrice", "entryTime", "exitTime",
    "entryName", "exitName", "profit", "commission", "mae", "mfe", "etd",
    "bars", "holdingMinutes", "entryHour", "entryHalfHour", "entryDayOfWeek", "entryDate",
]

# Dedup key columns — same grouping key used by read_trades()
DEDUP_COLUMNS = ("subStrategy", "direction", "entryTime", "exitTime")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER,
            instrument      TEXT NOT NULL,
            instrumentFull  TEXT,
            strategy        TEXT NOT NULL,
            subStrategy     TEXT NOT NULL,
            direction       TEXT NOT NULL,
            qty             INTEGER,
            entryPrice      REAL,
            exitPrice       REAL,
            entryTime       TEXT NOT NULL,
            exitTime        TEXT NOT NULL,
            entryName       TEXT,
            exitName        TEXT,
            profit          REAL,
            commission      REAL,
            mae             REAL,
            mfe             REAL,
            etd             REAL,
            bars            INTEGER,
            holdingMinutes  REAL,
            entryHour       INTEGER,
            entryHalfHour   TEXT,
            entryDayOfWeek  INTEGER,
            entryDate       TEXT,
            source_file     TEXT,
            ingested_at     TEXT,
            UNIQUE(subStrategy, direction, entryTime, exitTime)
        );

        CREATE TABLE IF NOT EXISTS ingestion_log (
            file_path   TEXT PRIMARY KEY,
            file_name   TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            rows_raw    INTEGER DEFAULT 0,
            rows_sim    INTEGER DEFAULT 0,
            trades_new  INTEGER DEFAULT 0,
            trades_dup  INTEGER DEFAULT 0
        );
    """)
    # Migrate: add entryHalfHour column if missing (added after initial schema)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "entryHalfHour" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN entryHalfHour TEXT")
        conn.commit()


def insert_trades(conn: sqlite3.Connection, trades: list[dict],
                  source_file: str = "") -> tuple[int, int]:
    """Insert trades with INSERT OR IGNORE dedup.
    Returns (new_count, dup_count).
    """
    now = datetime.now().isoformat()
    cols = TRADE_COLUMNS + ["source_file", "ingested_at"]
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO trades ({col_names}) VALUES ({placeholders})"

    new_count = 0
    dup_count = 0
    for t in trades:
        values = [t.get(c) for c in TRADE_COLUMNS] + [source_file, now]
        cursor = conn.execute(sql, values)
        if cursor.rowcount > 0:
            new_count += 1
        else:
            dup_count += 1

    conn.commit()
    return new_count, dup_count


def read_all_trades(conn: sqlite3.Connection) -> list[dict]:
    """Read all trades from the DB, returning list of dicts matching the
    format produced by process_trades.read_trades().
    """
    col_names = ", ".join(TRADE_COLUMNS)
    cursor = conn.execute(f"SELECT {col_names} FROM trades ORDER BY exitTime")
    rows = cursor.fetchall()
    trades = []
    for row in rows:
        t = dict(zip(TRADE_COLUMNS, row))
        # Ensure correct types for numeric fields
        for key in ("id", "qty", "bars", "entryHour", "entryDayOfWeek"):
            if t[key] is not None:
                t[key] = int(t[key])
        for key in ("entryPrice", "exitPrice", "profit", "commission",
                     "mae", "mfe", "etd", "holdingMinutes"):
            if t[key] is not None:
                t[key] = float(t[key])
        # Backfill entryHalfHour from entryHour if missing (old DB records)
        if not t.get("entryHalfHour") and t.get("entryHour") is not None:
            t["entryHalfHour"] = f"{t['entryHour']:02d}:00"
        trades.append(t)
    return trades


def log_ingestion(conn: sqlite3.Connection, file_path: str, file_name: str,
                  rows_raw: int, rows_sim: int,
                  trades_new: int, trades_dup: int) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO ingestion_log
            (file_path, file_name, ingested_at, rows_raw, rows_sim, trades_new, trades_dup)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (file_path, file_name, datetime.now().isoformat(),
          rows_raw, rows_sim, trades_new, trades_dup))
    conn.commit()


def get_ingestion_log(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.execute(
        "SELECT file_path, file_name, ingested_at, rows_raw, rows_sim, trades_new, trades_dup "
        "FROM ingestion_log ORDER BY ingested_at DESC"
    )
    cols = ["file_path", "file_name", "ingested_at", "rows_raw", "rows_sim",
            "trades_new", "trades_dup"]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def is_file_ingested(conn: sqlite3.Connection, file_path: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM ingestion_log WHERE file_path = ?", (file_path,)
    )
    return cursor.fetchone() is not None


def get_trade_count(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("SELECT COUNT(*) FROM trades")
    return cursor.fetchone()[0]
