import sqlite3
import time
import time as _time
import pytest
from tokengate.analytics.db import init_db, write_row
from tokengate.analytics.stats import get_stats


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "tokengate.db"
    init_db(p)
    return p


def test_init_creates_table(db_path):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [r[0] for r in rows]
    assert "requests" in tables
    con.close()


def test_init_wal_mode(db_path):
    con = sqlite3.connect(db_path)
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    con.close()


def test_write_row_ok(db_path):
    ts = time.time()
    write_row(db_path, ts=ts, route="openai", status="ok",
              tokens_in_raw=10, tokens_in_final=10, tokens_out=5,
              model_used="gpt-4o", latency_ms=123, est_cost_usd=0.001)
    con = sqlite3.connect(db_path)
    row = con.execute("SELECT * FROM requests").fetchone()
    con.close()
    assert row is not None


def test_write_row_columns(db_path):
    ts = time.time()
    write_row(db_path, ts=ts, route="anthropic", status="ok",
              tokens_in_raw=12, tokens_in_final=12, tokens_out=8,
              model_used="claude-sonnet-4-6", cache_kind="none",
              latency_ms=200, est_cost_usd=0.00016, est_saved_usd=0.0)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests").fetchone())
    con.close()
    assert row["route"] == "anthropic"
    assert row["status"] == "ok"
    assert row["tokens_in_raw"] == 12
    assert row["tokens_out"] == 8
    assert row["model_used"] == "claude-sonnet-4-6"
    assert row["cache_kind"] == "none"
    assert row["error_detail"] is None
    assert row["est_saved_usd"] == 0.0


def test_write_row_unknown_model_null_cost(db_path):
    write_row(db_path, ts=time.time(), route="openai", status="ok",
              tokens_in_raw=10, tokens_in_final=10, tokens_out=5,
              model_used="unknown-model-xyz", est_cost_usd=None)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests").fetchone())
    con.close()
    assert row["est_cost_usd"] is None


def test_write_row_upstream_error(db_path):
    write_row(db_path, ts=time.time(), route="openai", status="upstream_error",
              error_detail="503 Service Unavailable")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests").fetchone())
    con.close()
    assert row["status"] == "upstream_error"
    assert row["error_detail"] == "503 Service Unavailable"


def test_index_on_ts_exists(db_path):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    con.close()
    assert any("requests_ts" in r[0] for r in rows)


def test_get_stats_empty(db_path):
    stats = get_stats(db_path)
    assert stats["total_requests"] == 0
    assert stats["cache_hit_rate"] == 0.0
    assert stats["daily"] == []


def test_get_stats_totals(db_path):
    ts = _time.time()
    write_row(db_path, ts=ts, route="openai", status="ok",
              tokens_in_raw=100, tokens_in_final=100, tokens_out=50,
              model_used="gpt-4o", est_cost_usd=0.001)
    write_row(db_path, ts=ts, route="anthropic", status="ok",
              tokens_in_raw=200, tokens_in_final=200, tokens_out=100,
              model_used="claude-sonnet-4-6", est_cost_usd=0.002)
    stats = get_stats(db_path)
    assert stats["total_requests"] == 2
    assert stats["total_tokens_in"] == 300
    assert stats["total_tokens_out"] == 150
    assert stats["requests_by_status"]["ok"] == 2


def test_get_stats_null_cost_excluded_from_total(db_path):
    ts = _time.time()
    write_row(db_path, ts=ts, route="openai", status="ok",
              tokens_in_raw=10, tokens_in_final=10, tokens_out=5,
              model_used="unknown-xyz", est_cost_usd=None)
    stats = get_stats(db_path)
    # SUM of NULL values is NULL — total_est_cost_usd should be None, not 0
    assert stats["total_est_cost_usd"] is None
