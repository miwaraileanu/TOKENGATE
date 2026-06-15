"""Tests for Phase 2 caching — exact cache, semantic cache, and infrastructure."""
from __future__ import annotations
import json
import sqlite3
import pytest
from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayRequest, GatewayResponse
from tokengate.core.config import Settings
from tokengate.analytics.db import init_db


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_settings(tmp_path) -> Settings:
    import os
    old = os.environ.get("TOKENGATE_DATA_DIR")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    init_db(s.db_path)
    if old is None:
        del os.environ["TOKENGATE_DATA_DIR"]
    else:
        os.environ["TOKENGATE_DATA_DIR"] = old
    return s


def make_request(**overrides) -> GatewayRequest:
    defaults = dict(
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-haiku-4-5-20251001",
        stream=False,
        max_tokens=100,
        temperature=0.0,
        tools=[],
        route="anthropic",
        raw_headers={},
    )
    defaults.update(overrides)
    return GatewayRequest(**defaults)


def make_response() -> GatewayResponse:
    return GatewayResponse(
        content="Test response",
        model="claude-haiku-4-5-20251001",
        tokens_in=10,
        tokens_out=5,
        stop_reason="end_turn",
        raw_body={"id": "msg_test"},
    )


# ── Task 1: LayerContext infrastructure ─────────────────────────────────────

def test_layer_context_has_settings_field():
    req = make_request()
    ctx = LayerContext(request=req)
    assert ctx.settings is None  # default


def test_layer_context_settings_can_be_set(tmp_path):
    s = make_settings(tmp_path)
    req = make_request()
    ctx = LayerContext(request=req, settings=s)
    assert ctx.settings is s


def test_layer_context_cache_writers_default_empty():
    req = make_request()
    ctx = LayerContext(request=req)
    assert ctx.cache_writers == []


def test_layer_context_cache_writers_can_hold_callables():
    req = make_request()
    ctx = LayerContext(request=req)
    called = []

    async def writer(resp):
        called.append(resp)

    ctx.cache_writers.append(writer)
    assert len(ctx.cache_writers) == 1


# ── DB schema ────────────────────────────────────────────────────────────────

def test_db_has_cache_exact_table(tmp_path):
    s = make_settings(tmp_path)
    con = sqlite3.connect(s.db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()
    assert "cache_exact" in tables


def test_db_cache_exact_has_expected_columns(tmp_path):
    s = make_settings(tmp_path)
    con = sqlite3.connect(s.db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(cache_exact)").fetchall()}
    con.close()
    assert {"cache_key", "expires_at", "body_json"} <= cols


def test_db_has_cache_semantic_table(tmp_path):
    s = make_settings(tmp_path)
    con = sqlite3.connect(s.db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()
    assert "cache_semantic" in tables


def test_db_cache_semantic_has_expected_columns(tmp_path):
    s = make_settings(tmp_path)
    con = sqlite3.connect(s.db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(cache_semantic)").fetchall()}
    con.close()
    assert {"cache_key", "embedding", "body_json", "ts"} <= cols


# ── Task 2: Settings cache fields ────────────────────────────────────────────

def test_settings_cache_exact_ttl_default(tmp_path):
    s = make_settings(tmp_path)
    assert s.cache_exact_ttl == 86400


def test_settings_cache_semantic_threshold_default(tmp_path):
    s = make_settings(tmp_path)
    assert s.cache_semantic_threshold == 0.93


def test_settings_cache_max_entries_default(tmp_path):
    s = make_settings(tmp_path)
    assert s.cache_max_entries == 50000


def test_settings_cache_blocklist_default(tmp_path):
    s = make_settings(tmp_path)
    assert isinstance(s.cache_blocklist, list)
    assert len(s.cache_blocklist) > 0


def test_settings_cache_serve_unverified_default_false(tmp_path):
    s = make_settings(tmp_path)
    assert s.cache_serve_unverified is False


def test_settings_cache_fields_read_from_yaml(tmp_path):
    import yaml, os
    yaml_path = tmp_path / "tokengate.yaml"
    yaml_path.write_text(yaml.dump({
        "cache": {
            "exact_ttl_seconds": 3600,
            "semantic_threshold": 0.95,
            "max_entries": 1000,
            "serve_unverified": True,
            "blocklist_patterns": [r"\btest\b"],
        }
    }))
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings(config_path=yaml_path)
    assert s.cache_exact_ttl == 3600
    assert s.cache_semantic_threshold == 0.95
    assert s.cache_max_entries == 1000
    assert s.cache_serve_unverified is True
    assert s.cache_blocklist == [r"\btest\b"]
