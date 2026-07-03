from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS requests (
    id              INTEGER PRIMARY KEY,
    ts              REAL    NOT NULL,
    route           TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    error_detail    TEXT,
    layers_applied  TEXT    NOT NULL DEFAULT '[]',
    tokens_in_raw   INTEGER,
    tokens_in_final INTEGER,
    tokens_out      INTEGER,
    model_used      TEXT,
    cache_kind      TEXT    NOT NULL DEFAULT 'none',
    escalated       INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER,
    est_cost_usd    REAL,
    est_saved_usd   REAL    NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);

CREATE TABLE IF NOT EXISTS cache_exact (
    cache_key  TEXT PRIMARY KEY,
    expires_at REAL NOT NULL,
    body_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cache_exact_exp ON cache_exact(expires_at);

CREATE TABLE IF NOT EXISTS cache_semantic (
    cache_key  TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,
    body_json  TEXT NOT NULL,
    ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cache_semantic_ts ON cache_semantic(ts);

CREATE TABLE IF NOT EXISTS cache_summary (
    turns_hash   TEXT PRIMARY KEY,
    parent_hash  TEXT,
    summary      TEXT NOT NULL,
    pinned_facts TEXT NOT NULL DEFAULT '[]',
    expires_at   REAL NOT NULL,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cache_summary_parent ON cache_summary(parent_hash);

CREATE TABLE IF NOT EXISTS distiller_turns (
    turn_hash    TEXT PRIMARY KEY,
    turn_text    TEXT NOT NULL,
    embedding    BLOB,
    expires_at   REAL NOT NULL,
    ts           REAL NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(_SCHEMA)
    con.commit()
    con.close()


def evict_expired(db_path: Path) -> None:
    """Delete rows past their TTL from all time-bounded tables."""
    now = time.time()
    con = sqlite3.connect(db_path)
    con.execute("DELETE FROM cache_exact WHERE expires_at < ?", (now,))
    con.execute("DELETE FROM cache_summary WHERE expires_at < ?", (now,))
    con.execute("DELETE FROM distiller_turns WHERE expires_at < ?", (now,))
    con.commit()
    con.close()


def write_row(
    db_path: Path,
    *,
    ts: float,
    route: str,
    status: str,
    error_detail: str | None = None,
    layers_applied: list | None = None,
    tokens_in_raw: int | None = None,
    tokens_in_final: int | None = None,
    tokens_out: int | None = None,
    model_used: str | None = None,
    cache_kind: str = "none",
    escalated: int = 0,
    latency_ms: int | None = None,
    est_cost_usd: float | None = None,
    est_saved_usd: float = 0.0,
) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO requests
           (ts, route, status, error_detail, layers_applied,
            tokens_in_raw, tokens_in_final, tokens_out, model_used,
            cache_kind, escalated, latency_ms, est_cost_usd, est_saved_usd)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ts, route, status, error_detail,
            json.dumps(layers_applied or []),
            tokens_in_raw, tokens_in_final, tokens_out, model_used,
            cache_kind, escalated, latency_ms, est_cost_usd, est_saved_usd,
        ),
    )
    con.commit()
    con.close()
