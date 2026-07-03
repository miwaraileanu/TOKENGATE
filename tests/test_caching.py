"""Tests for Phase 2 caching — exact cache, semantic cache, and infrastructure."""
from __future__ import annotations
import json
import sqlite3
import time
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
    import yaml
    import os
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
    old = os.environ.get("TOKENGATE_DATA_DIR")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    try:
        s = Settings(config_path=yaml_path)
        assert s.cache_exact_ttl == 3600
        assert s.cache_semantic_threshold == 0.95
        assert s.cache_max_entries == 1000
        assert s.cache_serve_unverified is True
        assert s.cache_blocklist == [r"\btest\b"]
    finally:
        if old is None:
            del os.environ["TOKENGATE_DATA_DIR"]
        else:
            os.environ["TOKENGATE_DATA_DIR"] = old


# ── Task 3: L1 Exact Cache ───────────────────────────────────────────────────

import tokengate.layers.exact_cache as exact_cache


@pytest.mark.asyncio
async def test_exact_cache_first_request_is_miss(tmp_path):
    s = make_settings(tmp_path)
    ctx = LayerContext(request=make_request(), settings=s)
    ctx = await exact_cache.apply(ctx)
    assert ctx.response is None
    assert any(d.action == "miss" for d in ctx.decisions)
    assert len(ctx.cache_writers) == 1


@pytest.mark.asyncio
async def test_exact_cache_hit_after_write(tmp_path):
    s = make_settings(tmp_path)
    req = make_request()

    ctx1 = LayerContext(request=req, settings=s)
    ctx1 = await exact_cache.apply(ctx1)
    assert ctx1.response is None
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(request=req, settings=s)
    ctx2 = await exact_cache.apply(ctx2)
    assert ctx2.response is not None
    assert ctx2.response.content == "Test response"
    assert ctx2.response.tokens_in == 10
    assert ctx2.response.tokens_out == 5
    assert any(d.action == "hit" for d in ctx2.decisions)
    assert len(ctx2.cache_writers) == 0


@pytest.mark.asyncio
async def test_exact_cache_skips_streaming_read(tmp_path):
    s = make_settings(tmp_path)
    ctx = LayerContext(request=make_request(stream=True), settings=s)
    ctx = await exact_cache.apply(ctx)
    assert ctx.response is None
    assert len(ctx.cache_writers) == 0
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "streaming"


@pytest.mark.asyncio
async def test_exact_cache_skips_high_temperature(tmp_path):
    s = make_settings(tmp_path)
    ctx = LayerContext(request=make_request(temperature=0.9), settings=s)
    ctx = await exact_cache.apply(ctx)
    assert ctx.response is None
    assert len(ctx.cache_writers) == 0
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "temperature_too_high"


@pytest.mark.asyncio
async def test_exact_cache_writes_on_opt_in_header(tmp_path):
    s = make_settings(tmp_path)
    ctx = LayerContext(
        request=make_request(temperature=0.9, raw_headers={"x-tokengate-cache-write": "always"}),
        settings=s,
    )
    ctx = await exact_cache.apply(ctx)
    assert len(ctx.cache_writers) == 1


@pytest.mark.asyncio
async def test_exact_cache_different_system_prompts_no_collision(tmp_path):
    s = make_settings(tmp_path)

    req_pirate = make_request(messages=[
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Hello"},
    ])
    req_chef = make_request(messages=[
        {"role": "system", "content": "You are a chef."},
        {"role": "user", "content": "Hello"},
    ])

    ctx_a = LayerContext(request=req_pirate, settings=s)
    ctx_a = await exact_cache.apply(ctx_a)
    await ctx_a.cache_writers[0](make_response())

    ctx_b = LayerContext(request=req_chef, settings=s)
    ctx_b = await exact_cache.apply(ctx_b)
    assert ctx_b.response is None
    assert any(d.action == "miss" for d in ctx_b.decisions)

    con = sqlite3.connect(s.db_path)
    rows = con.execute("SELECT cache_key FROM cache_exact").fetchall()
    con.close()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_exact_cache_expired_entry_is_miss(tmp_path):
    s = make_settings(tmp_path)
    req = make_request()

    ctx = LayerContext(request=req, settings=s)
    ctx = await exact_cache.apply(ctx)
    await ctx.cache_writers[0](make_response())

    con = sqlite3.connect(s.db_path)
    con.execute("UPDATE cache_exact SET expires_at = ?", (time.time() - 10,))
    con.commit()
    con.close()

    ctx2 = LayerContext(request=req, settings=s)
    ctx2 = await exact_cache.apply(ctx2)
    assert ctx2.response is None
    assert any(d.action == "miss" for d in ctx2.decisions)


# ── Task 4: L2 Semantic Cache ────────────────────────────────────────────────

import numpy as np
import tokengate.layers.semantic_cache as sem_cache

# 2-D unit vectors with known dot products
_EMB_A = np.array([1.0, 0.0], dtype=np.float32)
_EMB_B = np.array([0.98, float(np.sqrt(1 - 0.98 ** 2))], dtype=np.float32)   # dot=0.98 >=0.97
_EMB_C = np.array([0.94, float(np.sqrt(1 - 0.94 ** 2))], dtype=np.float32)   # dot=0.94 in [0.93,0.97)
_EMB_D = np.array([0.5, float(np.sqrt(0.75))], dtype=np.float32)              # dot=0.5 <0.93

_EMB_MAP = {
    "original": _EMB_A,
    "paraphrase": _EMB_B,
    "slight variant": _EMB_C,
    "unrelated": _EMB_D,
}


def _fake_embed(text: str):
    lower = text.lower()
    for key, emb in _EMB_MAP.items():
        if key in lower:
            return emb.copy()
    return np.array([0.0, 1.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_semantic_state():
    sem_cache._index.clear()
    sem_cache.set_embedder(None)
    yield
    sem_cache._index.clear()
    sem_cache.set_embedder(None)


@pytest.mark.asyncio
async def test_semantic_cache_first_request_is_miss(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)
    ctx = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx = await sem_cache.apply(ctx)
    assert ctx.response is None
    assert any(d.action == "miss" for d in ctx.decisions)
    assert len(ctx.cache_writers) == 1


@pytest.mark.asyncio
async def test_semantic_cache_high_confidence_hit(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(request=make_request(messages=[{"role": "user", "content": "paraphrase query"}]), settings=s)
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is not None
    hit = next(d for d in ctx2.decisions if d.action == "hit")
    assert hit.detail["score"] >= 0.97
    assert hit.detail["verified"] is True


@pytest.mark.asyncio
async def test_semantic_cache_unverified_blocked_by_default(tmp_path):
    s = make_settings(tmp_path)
    assert s.cache_serve_unverified is False
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(request=make_request(messages=[{"role": "user", "content": "slight variant query"}]), settings=s)
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is None
    miss = next(d for d in ctx2.decisions if d.action == "miss")
    assert miss.detail.get("reason") == "unverified_blocked"


@pytest.mark.asyncio
async def test_semantic_cache_unverified_served_when_opted_in(tmp_path):
    s = make_settings(tmp_path)
    s.cache_serve_unverified = True
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(request=make_request(messages=[{"role": "user", "content": "slight variant query"}]), settings=s)
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is not None
    hit = next(d for d in ctx2.decisions if d.action == "hit")
    assert 0.93 <= hit.detail["score"] < 0.97
    assert hit.detail["verified"] is True


@pytest.mark.asyncio
async def test_semantic_cache_low_similarity_is_miss(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(request=make_request(messages=[{"role": "user", "content": "unrelated query"}]), settings=s)
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is None
    assert len(ctx2.cache_writers) == 1


@pytest.mark.asyncio
async def test_semantic_cache_blocklist_skips(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)
    ctx = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "what is the price today?"}]),
        settings=s,
    )
    ctx = await sem_cache.apply(ctx)
    assert ctx.response is None
    assert len(ctx.cache_writers) == 0
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "blocklisted"


@pytest.mark.asyncio
async def test_semantic_cache_skips_tool_calls(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)
    ctx = LayerContext(
        request=make_request(tools=[{"name": "search", "description": "Search the web"}]),
        settings=s,
    )
    ctx = await sem_cache.apply(ctx)
    assert ctx.response is None
    assert len(ctx.cache_writers) == 0
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "has_tools"


@pytest.mark.asyncio
async def test_semantic_cache_skips_streaming_read(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)
    ctx = LayerContext(request=make_request(stream=True), settings=s)
    ctx = await sem_cache.apply(ctx)
    assert ctx.response is None
    assert len(ctx.cache_writers) == 0
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "streaming"


@pytest.mark.asyncio
async def test_semantic_cache_no_embedder_skips_gracefully(tmp_path):
    s = make_settings(tmp_path)
    # sem_cache.set_embedder(None) already called by reset_semantic_state fixture
    if sem_cache._HAVE_EMBEDDER:
        pytest.skip("sentence-transformers installed; no_embedder path not testable")
    ctx = LayerContext(request=make_request(), settings=s)
    ctx = await sem_cache.apply(ctx)
    assert ctx.response is None
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "no_embedder"


@pytest.mark.asyncio
async def test_semantic_cache_index_persists_to_sqlite(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx = await sem_cache.apply(ctx)
    await ctx.cache_writers[0](make_response())

    con = sqlite3.connect(s.db_path)
    row = con.execute("SELECT cache_key, embedding, body_json FROM cache_semantic").fetchone()
    con.close()
    assert row is not None
    key, emb_bytes, body_json = row
    assert len(emb_bytes) > 0
    body = json.loads(body_json)
    assert body["tokens_in"] == 10
    assert body["tokens_out"] == 5


@pytest.mark.asyncio
async def test_semantic_cache_index_reloads_from_sqlite(tmp_path):
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx = LayerContext(request=make_request(messages=[{"role": "user", "content": "original query"}]), settings=s)
    ctx = await sem_cache.apply(ctx)
    await ctx.cache_writers[0](make_response())

    sem_cache._index.clear()
    assert len(sem_cache._index) == 0

    sem_cache.load_index(s.db_path, s.cache_max_entries)
    assert len(sem_cache._index) == 1

    ctx2 = LayerContext(request=make_request(messages=[{"role": "user", "content": "paraphrase query"}]), settings=s)
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is not None


# ── Task 5: Integration tests (server wiring) ────────────────────────────────

import tokengate.proxy.server as _sv
from tokengate.core.mock_provider import MockTransport
from fastapi.testclient import TestClient


@pytest.fixture
def cache_client(tmp_path, monkeypatch):
    import tokengate.layers.router as _l_router
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.prices["mock"] = (1.0, 5.0)   # give mock model a price → est_saved_usd > 0
    s.router_escalation_enabled = False  # keep router to 1 upstream call so transport.requests count is predictable
    _sv._settings = s
    init_db(s.db_path)
    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    monkeypatch.setattr(_l_router, "_transport", transport)
    sem_cache._index.clear()
    sem_cache.set_embedder(None)
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, transport, s
    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_router, "_transport", None)
    sem_cache._index.clear()
    sem_cache.set_embedder(None)


_ANT_BODY = {
    "model": "claude-haiku-4-5-20251001",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
}


def test_integration_exact_cache_hit(cache_client):
    client, transport, _ = cache_client
    r1 = client.post("/v1/messages", json=_ANT_BODY)
    assert r1.status_code == 200
    assert r1.headers["x-tokengate-cache"] == "none"

    r2 = client.post("/v1/messages", json=_ANT_BODY)
    assert r2.status_code == 200
    assert r2.headers["x-tokengate-cache"] == "exact"
    assert r2.headers["x-tokengate-saved-tokens"] != "0"
    assert len(transport.requests) == 1


def test_integration_exact_cache_miss_on_high_temperature(cache_client):
    client, transport, _ = cache_client
    body = {**_ANT_BODY, "temperature": 0.9}
    client.post("/v1/messages", json=body)
    client.post("/v1/messages", json=body)
    assert len(transport.requests) == 2


def test_integration_exact_cache_est_saved_usd_in_db(cache_client):
    client, transport, s = cache_client
    client.post("/v1/messages", json=_ANT_BODY)
    client.post("/v1/messages", json=_ANT_BODY)

    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT cache_kind, est_saved_usd, est_cost_usd FROM requests WHERE cache_kind != 'none'"
    ).fetchall()
    con.close()

    assert len(rows) == 1
    row = dict(rows[0])
    assert row["cache_kind"] == "exact"
    assert row["est_saved_usd"] > 0
    assert row["est_cost_usd"] == 0.0


def test_integration_semantic_cache_hit(cache_client):
    client, transport, s = cache_client
    sem_cache.set_embedder(_fake_embed)

    r1 = client.post("/v1/messages", json={
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "original query"}],
        "max_tokens": 100,
    })
    assert r1.status_code == 200

    r2 = client.post("/v1/messages", json={
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "paraphrase query"}],
        "max_tokens": 100,
    })
    assert r2.status_code == 200
    assert r2.headers["x-tokengate-cache"] == "semantic"
    assert len(transport.requests) == 1


def test_integration_semantic_unverified_blocked_by_default(cache_client):
    client, transport, s = cache_client
    assert s.cache_serve_unverified is False
    sem_cache.set_embedder(_fake_embed)

    client.post("/v1/messages", json={
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "original query"}],
        "max_tokens": 100,
    })
    r2 = client.post("/v1/messages", json={
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "slight variant query"}],
        "max_tokens": 100,
    })
    assert r2.headers["x-tokengate-cache"] == "none"
    assert len(transport.requests) == 2


def test_integration_streaming_bypasses_cache(cache_client):
    client, transport, _ = cache_client
    body = {**_ANT_BODY, "stream": True}
    client.post("/v1/messages", json=body)
    client.post("/v1/messages", json=body)
    assert len(transport.requests) == 2


def test_integration_stats_cache_breakdown(cache_client):
    client, transport, s = cache_client
    client.post("/v1/messages", json=_ANT_BODY)  # miss
    client.post("/v1/messages", json=_ANT_BODY)  # exact hit

    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "cache_by_kind" in data
    assert "exact" in data["cache_by_kind"]
    assert data["cache_by_kind"]["exact"]["count"] == 1
    assert data["cache_by_kind"]["exact"]["saved_usd"] > 0
