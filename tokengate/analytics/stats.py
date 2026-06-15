from __future__ import annotations
import sqlite3
from pathlib import Path


def get_stats(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    totals = dict(con.execute("""
        SELECT
            COUNT(*) AS total_requests,
            COALESCE(SUM(tokens_in_final), 0) AS total_tokens_in,
            COALESCE(SUM(tokens_out), 0) AS total_tokens_out,
            SUM(est_cost_usd) AS total_est_cost_usd,
            COALESCE(SUM(est_saved_usd), 0.0) AS total_est_saved_usd,
            CAST(SUM(CASE WHEN cache_kind != 'none' THEN 1 ELSE 0 END) AS REAL)
              / NULLIF(COUNT(*), 0) AS cache_hit_rate
        FROM requests
    """).fetchone())

    by_status = {
        row["status"]: row["cnt"]
        for row in con.execute(
            "SELECT status, COUNT(*) AS cnt FROM requests GROUP BY status"
        ).fetchall()
    }

    daily = [
        dict(row)
        for row in con.execute("""
            SELECT
                DATE(ts, 'unixepoch') AS date,
                COUNT(*) AS requests,
                COALESCE(SUM(tokens_in_final), 0) AS tokens_in,
                COALESCE(SUM(tokens_out), 0) AS tokens_out,
                SUM(est_cost_usd) AS est_cost_usd
            FROM requests
            GROUP BY DATE(ts, 'unixepoch')
            ORDER BY date DESC
            LIMIT 30
        """).fetchall()
    ]

    cache_by_kind = {
        row["cache_kind"]: {
            "count": row["cnt"],
            "saved_usd": row["saved"],
        }
        for row in con.execute("""
            SELECT cache_kind,
                   COUNT(*) AS cnt,
                   COALESCE(SUM(est_saved_usd), 0.0) AS saved
            FROM requests
            WHERE cache_kind != 'none'
            GROUP BY cache_kind
        """).fetchall()
    }

    con.close()

    return {
        "total_requests": totals["total_requests"],
        "total_tokens_in": totals["total_tokens_in"],
        "total_tokens_out": totals["total_tokens_out"],
        "total_est_cost_usd": totals["total_est_cost_usd"],
        "total_est_saved_usd": totals["total_est_saved_usd"],
        "cache_hit_rate": totals["cache_hit_rate"] or 0.0,
        "requests_by_status": by_status,
        "daily": daily,
        "cache_by_kind": cache_by_kind,
    }


def get_recent(db_path: Path, limit: int = 50) -> list[dict]:
    """Return the last `limit` request rows, newest first."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT ts, route, status, model_used, tokens_in_raw,
                      tokens_out, latency_ms, est_cost_usd
               FROM requests
               ORDER BY ts DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
