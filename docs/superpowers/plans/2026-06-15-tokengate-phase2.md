# TokenGate Phase 2 — Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement L1 exact caching and L2 semantic caching in the TokenGate gateway, including two infrastructure fixes that Phase 1 left as stubs, with full offline test coverage.

**Architecture:** Fix `LayerContext` (add `settings` + `cache_writers` fields) and the DB schema (add `cache_exact` + `cache_semantic` tables) first. Exact cache keys on the full normalized message array (including system prompt and all history) via SHA-256. Semantic cache uses an injectable embedder backed by `sentence-transformers`, an in-memory LRU `OrderedDict` index that survives restarts via SQLite blob persistence, and a `verify_hit()` stub wired at the call site for Phase 4. The server passes settings into every `LayerContext`, calls cache writers after a successful upstream response, and records real `cache_kind` + `est_saved_usd` in analytics.

**Tech Stack:** Python 3.12, FastAPI, SQLite3 (stdlib), numpy>=1.26, sentence-transformers>=3 (optional), pytest, pytest-asyncio

---

## File Map

```
tokengate/core/context.py           modify  — add settings + cache_writers fields
tokengate/core/config.py            modify  — add cache_* Settings fields
tokengate/analytics/db.py           modify  — add cache_exact + cache_semantic tables to _SCHEMA
tokengate/analytics/stats.py        modify  — add cache_by_kind breakdown to get_stats()
tokengate/layers/exact_cache.py     rewrite — full L1 implementation (was no-op stub)
tokengate/layers/semantic_cache.py  rewrite — full L2 implementation (was no-op stub)
tokengate/proxy/server.py           modify  — settings in ctx, call cache_writers, headers, analytics
tokengate.yaml                      modify  — add serve_unverified: false under cache:
pyproject.toml                      modify  — numpy in dev deps; semantic optional extra
tests/test_caching.py               create  — all Phase 2 cache unit + integration tests
```

---

## Task 1: Infrastructure — LayerContext fields + DB schema

**Files:**
- Modify: `tokengate/core/context.py`
- Modify: `tokengate/analytics/db.py`

- [ ] **Step 1: Write failing tests for `LayerContext` new fields**

Add these tests to a new file `tests/test_caching.py`:

```python
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
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    init_db(s.db_path)
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
    import os
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
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
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_caching.py::test_layer_context_has_settings_field -v
```

Expected: `FAILED` — `LayerContext.__init__() got an unexpected keyword argument 'settings'`

- [ ] **Step 3: Update `tokengate/core/context.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokengate.core.normalize import GatewayRequest, GatewayResponse
    from tokengate.core.config import Settings


@dataclass
class LayerDecision:
    layer: str
    action: str  # "hit" | "miss" | "skip" | "applied" | "escalated"
    detail: dict = field(default_factory=dict)


@dataclass
class LayerContext:
    request: "GatewayRequest"
    response: "GatewayResponse | None" = None
    decisions: list[LayerDecision] = field(default_factory=list)
    settings: "Settings | None" = None
    cache_writers: list = field(default_factory=list)  # list[Callable[[GatewayResponse], Awaitable[None]]]
```

- [ ] **Step 4: Run context tests — confirm pass**

```
pytest tests/test_caching.py::test_layer_context_has_settings_field tests/test_caching.py::test_layer_context_settings_can_be_set tests/test_caching.py::test_layer_context_cache_writers_default_empty tests/test_caching.py::test_layer_context_cache_writers_can_hold_callables -v
```

Expected: `4 passed`

- [ ] **Step 5: Write failing DB schema tests — append to `tests/test_caching.py`**

```python
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
```

- [ ] **Step 6: Run DB schema tests — confirm failure**

```
pytest tests/test_caching.py::test_db_has_cache_exact_table tests/test_caching.py::test_db_has_cache_semantic_table -v
```

Expected: `FAILED` — tables don't exist yet

- [ ] **Step 7: Update `tokengate/analytics/db.py` — add cache tables to `_SCHEMA`**

Replace the existing `_SCHEMA` string:

```python
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
"""
```

- [ ] **Step 8: Run all Task 1 tests — confirm pass**

```
pytest tests/test_caching.py -k "layer_context or db_" -v
```

Expected: `8 passed`

- [ ] **Step 9: Run full existing test suite — confirm no regressions**

```
pytest tests/ -v --ignore=tests/test_caching.py
```

Expected: all existing tests pass

- [ ] **Step 10: Commit**

```bash
git add tokengate/core/context.py tokengate/analytics/db.py tests/test_caching.py
git commit -m "feat: add settings+cache_writers to LayerContext; add cache tables to DB schema"
```

---

## Task 2: Settings cache fields

**Files:**
- Modify: `tokengate/core/config.py`
- Modify: `tokengate.yaml`

- [ ] **Step 1: Write failing settings tests — append to `tests/test_caching.py`**

```python
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
```

- [ ] **Step 2: Run — confirm failure**

```
pytest tests/test_caching.py -k "settings_cache" -v
```

Expected: `FAILED` — `Settings` has no attribute `cache_exact_ttl`

- [ ] **Step 3: Update `tokengate/core/config.py` — add cache fields**

Append inside `Settings.__init__`, after the `self.db_path` / `self.pid_path` / `self.log_path` lines:

```python
        _c = raw.get("cache", {})
        self.cache_exact_ttl: int = int(_c.get("exact_ttl_seconds", 86400))
        self.cache_semantic_threshold: float = float(_c.get("semantic_threshold", 0.93))
        self.cache_max_entries: int = int(_c.get("max_entries", 50000))
        self.cache_blocklist: list = list(_c.get("blocklist_patterns", [
            r"\btoday\b", r"\bnow\b", r"\blatest\b", r"\bprice\b",
        ]))
        self.cache_serve_unverified: bool = bool(_c.get("serve_unverified", False))
```

- [ ] **Step 4: Update `tokengate.yaml` — add `serve_unverified: false` under `cache:`**

The existing `cache:` block already has `exact_ttl_seconds`, `semantic_threshold`, `max_entries`, `blocklist_patterns`. Add one line:

```yaml
cache:
  exact_ttl_seconds: 86400
  semantic_threshold: 0.93
  max_entries: 50000
  serve_unverified: false
  blocklist_patterns:
    - "\\btoday\\b"
    - "\\bnow\\b"
    - "\\blatest\\b"
    - "\\bprice\\b"
```

- [ ] **Step 5: Run settings tests — confirm pass**

```
pytest tests/test_caching.py -k "settings_cache" -v
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add tokengate/core/config.py tokengate.yaml tests/test_caching.py
git commit -m "feat: add cache config fields to Settings; serve_unverified default false"
```

---

## Task 3: L1 Exact Cache

**Files:**
- Modify: `tokengate/layers/exact_cache.py`

- [ ] **Step 1: Write failing exact cache tests — append to `tests/test_caching.py`**

```python
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
    """Header x-tokengate-cache-write: always overrides temperature check."""
    s = make_settings(tmp_path)
    ctx = LayerContext(
        request=make_request(temperature=0.9, raw_headers={"x-tokengate-cache-write": "always"}),
        settings=s,
    )
    ctx = await exact_cache.apply(ctx)
    assert len(ctx.cache_writers) == 1  # write-back registered despite high temp


@pytest.mark.asyncio
async def test_exact_cache_different_system_prompts_no_collision(tmp_path):
    """Two requests with same last message but different system prompts must NOT collide."""
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
    assert ctx_b.response is None, "different system prompts must not produce a cache hit"
    assert any(d.action == "miss" for d in ctx_b.decisions)

    con = sqlite3.connect(s.db_path)
    rows = con.execute("SELECT cache_key FROM cache_exact").fetchall()
    con.close()
    assert len(rows) == 1  # only pirate written so far; chef was a miss


@pytest.mark.asyncio
async def test_exact_cache_expired_entry_is_miss(tmp_path):
    import time
    s = make_settings(tmp_path)
    req = make_request()

    ctx = LayerContext(request=req, settings=s)
    ctx = await exact_cache.apply(ctx)
    await ctx.cache_writers[0](make_response())

    # Manually expire the entry
    con = sqlite3.connect(s.db_path)
    con.execute("UPDATE cache_exact SET expires_at = ?", (time.time() - 10,))
    con.commit()
    con.close()

    ctx2 = LayerContext(request=req, settings=s)
    ctx2 = await exact_cache.apply(ctx2)
    assert ctx2.response is None
    assert any(d.action == "miss" for d in ctx2.decisions)
```

- [ ] **Step 2: Run — confirm failure**

```
pytest tests/test_caching.py -k "exact_cache" -v
```

Expected: all `FAILED` — layer is still the no-op stub

- [ ] **Step 3: Rewrite `tokengate/layers/exact_cache.py`**

```python
from __future__ import annotations
import hashlib
import json
import sqlite3
import time

from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayResponse


def _compute_key(req) -> str:
    """SHA-256 of the full message array + model + temperature bucket + tools.

    req.messages already contains the system prompt (normalize_anthropic prepends
    it as {"role": "system", ...}) so two requests differing only in system prompt
    or any earlier turn produce different keys.
    """
    temp_bucket = round(req.temperature, 1) if req.temperature is not None else None
    payload = {
        "messages": req.messages,
        "model": req.model,
        "temp": temp_bucket,
        "tools": req.tools,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def _write_eligible(req) -> bool:
    if req.raw_headers.get("x-tokengate-cache-write", "").lower() == "always":
        return True
    return req.temperature is None or req.temperature <= 0.3


async def apply(ctx: LayerContext) -> LayerContext:
    if ctx.settings is None:
        return ctx

    req = ctx.request

    if req.stream:
        ctx.decisions.append(LayerDecision("exact_cache", "skip", {"reason": "streaming"}))
        return ctx

    if not _write_eligible(req):
        ctx.decisions.append(LayerDecision("exact_cache", "skip", {"reason": "temperature_too_high"}))
        return ctx

    db_path = ctx.settings.db_path
    ttl = ctx.settings.cache_exact_ttl
    cache_key = _compute_key(req)
    now = time.time()

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT body_json FROM cache_exact WHERE cache_key=? AND expires_at>?",
        (cache_key, now),
    ).fetchone()
    con.close()

    if row:
        body = json.loads(row[0])
        ctx.response = GatewayResponse(**body)
        ctx.decisions.append(LayerDecision("exact_cache", "hit", {"key": cache_key}))
        return ctx

    ctx.decisions.append(LayerDecision("exact_cache", "miss", {"key": cache_key}))

    async def _write(response: GatewayResponse) -> None:
        body_json = json.dumps({
            "content": response.content,
            "model": response.model,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "stop_reason": response.stop_reason,
            "raw_body": response.raw_body,
        })
        expires_at = time.time() + ttl
        _con = sqlite3.connect(db_path)
        _con.execute(
            "INSERT OR REPLACE INTO cache_exact (cache_key, expires_at, body_json) VALUES (?,?,?)",
            (cache_key, expires_at, body_json),
        )
        _con.commit()
        _con.close()

    ctx.cache_writers.append(_write)
    return ctx
```

- [ ] **Step 4: Run exact cache tests — confirm pass**

```
pytest tests/test_caching.py -k "exact_cache" -v
```

Expected: `7 passed`

- [ ] **Step 5: Run full suite — confirm no regressions**

```
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add tokengate/layers/exact_cache.py tests/test_caching.py
git commit -m "feat: implement L1 exact cache with SHA-256 keying and TTL"
```

---

## Task 4: L2 Semantic Cache

**Files:**
- Modify: `tokengate/layers/semantic_cache.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add numpy to dev deps in `pyproject.toml`**

```toml
[project.optional-dependencies]
semantic = ["sentence-transformers>=3.0", "numpy>=1.26"]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
    "anyio[trio]>=4",
    "numpy>=1.26",
]
```

- [ ] **Step 2: Install updated dev deps**

```
pip install -e ".[dev]"
```

Expected: numpy installs (sentence-transformers is NOT required for the tests)

- [ ] **Step 3: Write failing semantic cache tests — append to `tests/test_caching.py`**

```python
# ── Task 4: L2 Semantic Cache ────────────────────────────────────────────────

import numpy as np
import tokengate.layers.semantic_cache as sem_cache

# 2-D unit vectors with known dot products (cosine similarity = dot product for L2-normalised vecs)
_EMB_A = np.array([1.0, 0.0], dtype=np.float32)
# dot(A, B) = 0.98  →  high-confidence hit (≥ 0.97)
_EMB_B = np.array([0.98, float(np.sqrt(1 - 0.98 ** 2))], dtype=np.float32)
# dot(A, C) ≈ 0.94  →  unverified band (0.93 ≤ score < 0.97)
_EMB_C = np.array([0.94, float(np.sqrt(1 - 0.94 ** 2))], dtype=np.float32)
# dot(A, D) = 0.50  →  miss (< 0.93)
_EMB_D = np.array([0.5, float(np.sqrt(0.75))], dtype=np.float32)

_EMB_MAP = {
    "original": _EMB_A,
    "paraphrase": _EMB_B,
    "slight variant": _EMB_C,
    "unrelated": _EMB_D,
}


def _fake_embed(text: str) -> np.ndarray:
    lower = text.lower()
    for key, emb in _EMB_MAP.items():
        if key in lower:
            return emb.copy()
    return np.array([0.0, 1.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_semantic_state():
    """Clear module-level index and embedder between every test in this file."""
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
    """Similarity ≥ 0.97 → cache_kind 'semantic', verified=True."""
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "original query"}]),
        settings=s,
    )
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "paraphrase query"}]),
        settings=s,
    )
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is not None
    hit = next(d for d in ctx2.decisions if d.action == "hit")
    assert hit.detail["score"] >= 0.97
    assert hit.detail["verified"] is True


@pytest.mark.asyncio
async def test_semantic_cache_unverified_blocked_by_default(tmp_path):
    """0.93 ≤ similarity < 0.97 with serve_unverified=False → miss."""
    s = make_settings(tmp_path)
    assert s.cache_serve_unverified is False
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "original query"}]),
        settings=s,
    )
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "slight variant query"}]),
        settings=s,
    )
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is None
    miss = next(d for d in ctx2.decisions if d.action == "miss")
    assert miss.detail.get("reason") == "unverified_blocked"


@pytest.mark.asyncio
async def test_semantic_cache_unverified_served_when_opted_in(tmp_path):
    """0.93 ≤ similarity < 0.97 with serve_unverified=True → hit, cache_kind 'semantic-unverified'."""
    s = make_settings(tmp_path)
    s.cache_serve_unverified = True
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "original query"}]),
        settings=s,
    )
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "slight variant query"}]),
        settings=s,
    )
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is not None
    hit = next(d for d in ctx2.decisions if d.action == "hit")
    assert 0.93 <= hit.detail["score"] < 0.97
    assert hit.detail["verified"] is True  # stub always returns True


@pytest.mark.asyncio
async def test_semantic_cache_low_similarity_is_miss(tmp_path):
    """similarity < 0.93 → miss, write-back registered."""
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx1 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "original query"}]),
        settings=s,
    )
    ctx1 = await sem_cache.apply(ctx1)
    await ctx1.cache_writers[0](make_response())

    ctx2 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "unrelated query"}]),
        settings=s,
    )
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is None
    assert len(ctx2.cache_writers) == 1  # write-back registered for the new entry


@pytest.mark.asyncio
async def test_semantic_cache_blocklist_skips(tmp_path):
    """Queries matching blocklist patterns are never cached (skip on read path too)."""
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
    """Requests with tools defined are never served from semantic cache."""
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
    """Streaming requests are bypassed on the READ path — no cached body returned."""
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
    """When embedder is not available, layer skips transparently."""
    s = make_settings(tmp_path)
    # Do NOT call set_embedder — leave it as None
    # If sentence-transformers is installed this test is less meaningful, but still valid
    if sem_cache._HAVE_EMBEDDER:
        pytest.skip("sentence-transformers is installed; no_embedder path untestable")
    ctx = LayerContext(request=make_request(), settings=s)
    ctx = await sem_cache.apply(ctx)
    assert ctx.response is None
    skip = next(d for d in ctx.decisions if d.action == "skip")
    assert skip.detail["reason"] == "no_embedder"


@pytest.mark.asyncio
async def test_semantic_cache_index_persists_to_sqlite(tmp_path):
    """write-back stores embedding + body in cache_semantic table."""
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "original query"}]),
        settings=s,
    )
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
    """After clearing the in-memory index, load_index() rebuilds it from SQLite."""
    s = make_settings(tmp_path)
    sem_cache.set_embedder(_fake_embed)

    ctx = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "original query"}]),
        settings=s,
    )
    ctx = await sem_cache.apply(ctx)
    await ctx.cache_writers[0](make_response())

    # Simulate server restart: clear in-memory index
    sem_cache._index.clear()
    assert len(sem_cache._index) == 0

    # Reload from SQLite
    sem_cache.load_index(s.db_path, s.cache_max_entries)
    assert len(sem_cache._index) == 1

    # A paraphrase should now hit the reloaded entry
    ctx2 = LayerContext(
        request=make_request(messages=[{"role": "user", "content": "paraphrase query"}]),
        settings=s,
    )
    ctx2 = await sem_cache.apply(ctx2)
    assert ctx2.response is not None
```

- [ ] **Step 4: Run — confirm all semantic tests fail**

```
pytest tests/test_caching.py -k "semantic_cache" -v
```

Expected: all `FAILED` — layer is still the stub

- [ ] **Step 5: Rewrite `tokengate/layers/semantic_cache.py`**

```python
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayResponse

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

_HAVE_EMBEDDER: bool = _HAVE_NUMPY and _HAVE_ST  # real model available

_model = None
_embed_fn = None  # injectable for tests; set via set_embedder()
_index: OrderedDict = OrderedDict()  # cache_key → (emb: np.ndarray, body_json: str)


def set_embedder(fn) -> None:
    """Inject a custom embed function (tests only). Pass None to reset."""
    global _embed_fn
    _embed_fn = fn


def can_embed() -> bool:
    return _HAVE_EMBEDDER or _embed_fn is not None


def _embed(text: str):
    if _embed_fn is not None:
        return _embed_fn(text)
    if not _HAVE_EMBEDDER:
        raise RuntimeError("sentence-transformers not installed and no embed_fn injected")
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode([text], normalize_embeddings=True)[0]


def _query_text(req) -> str:
    """Last user message + first 100 chars of system prompt as fingerprint."""
    system_fp = ""
    last_user = ""
    for msg in req.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_fp = (content if isinstance(content, str) else str(content))[:100]
        elif role == "user":
            if isinstance(content, list):
                last_user = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                last_user = str(content)
    return f"{system_fp}\n{last_user}".strip()


def _is_blocked(text: str, patterns: list) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


async def verify_hit(query: str, cached_answer: str) -> bool:
    """Phase 2: no-op stub — always returns True.
    Phase 4: replace this body with a cheap-model micro-prompt that rates whether
    the cached answer adequately answers the new query.
    """
    return True


def load_index(db_path: Path, max_entries: int) -> None:
    """Rebuild the in-memory index from SQLite after a server restart.
    Called from server lifespan when can_embed() is True.
    Requires numpy to deserialise embedding blobs.
    """
    if not _HAVE_NUMPY:
        return
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT cache_key, embedding, body_json FROM cache_semantic ORDER BY ts ASC"
    ).fetchall()
    con.close()
    for key, emb_bytes, body_json in rows[-max_entries:]:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        _index[key] = (emb, body_json)


async def apply(ctx: LayerContext) -> LayerContext:
    if not can_embed():
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "no_embedder"}))
        return ctx

    if ctx.settings is None:
        return ctx

    req = ctx.request

    if req.stream:
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "streaming"}))
        return ctx

    if req.tools:
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "has_tools"}))
        return ctx

    settings = ctx.settings
    query_text = _query_text(req)

    if _is_blocked(query_text, settings.cache_blocklist):
        ctx.decisions.append(LayerDecision("semantic_cache", "skip", {"reason": "blocklisted"}))
        return ctx

    query_emb = _embed(query_text)

    # ── Search in-memory index ──────────────────────────────────────────────
    best_key: str | None = None
    best_score: float = -1.0
    for key, (emb, _) in _index.items():
        score = float(np.dot(query_emb, emb))
        if score > best_score:
            best_score = score
            best_key = key

    threshold = settings.cache_semantic_threshold  # default 0.93

    if best_key is not None and best_score >= threshold:
        _, body_json = _index[best_key]
        body = json.loads(body_json)

        if best_score >= 0.97:
            ctx.response = GatewayResponse(**body)
            ctx.decisions.append(LayerDecision("semantic_cache", "hit", {
                "score": round(best_score, 4),
                "key": best_key,
                "verified": True,
            }))
            _index.move_to_end(best_key)
            return ctx

        # 0.93 ≤ score < 0.97 — unverified band
        verified = await verify_hit(query_text, body["content"])
        if settings.cache_serve_unverified and verified:
            ctx.response = GatewayResponse(**body)
            ctx.decisions.append(LayerDecision("semantic_cache", "hit", {
                "score": round(best_score, 4),
                "key": best_key,
                "verified": verified,
            }))
            _index.move_to_end(best_key)
            return ctx

        ctx.decisions.append(LayerDecision("semantic_cache", "miss", {
            "score": round(best_score, 4),
            "reason": "unverified_blocked",
        }))
    else:
        ctx.decisions.append(LayerDecision("semantic_cache", "miss", {
            "best_score": round(best_score, 4) if best_score >= 0 else None,
        }))

    # ── Register write-back callback ────────────────────────────────────────
    db_path = settings.db_path
    max_entries = settings.cache_max_entries
    # Capture query_emb in the closure (already a copy from _embed)
    _qemb = query_emb

    async def _write(response: GatewayResponse) -> None:
        key = hashlib.sha256(query_text.encode()).hexdigest()[:32]
        body_json = json.dumps({
            "content": response.content,
            "model": response.model,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "stop_reason": response.stop_reason,
            "raw_body": response.raw_body,
        })
        emb_bytes = _qemb.astype(np.float32).tobytes()
        ts = time.time()

        # Persist to SQLite
        _con = sqlite3.connect(db_path)
        _con.execute(
            "INSERT OR REPLACE INTO cache_semantic (cache_key, embedding, body_json, ts) "
            "VALUES (?,?,?,?)",
            (key, emb_bytes, body_json, ts),
        )
        _con.commit()
        _con.close()

        # Update in-memory index (LRU eviction)
        if key not in _index and len(_index) >= max_entries:
            _index.popitem(last=False)
        _index[key] = (_qemb, body_json)
        _index.move_to_end(key)

    ctx.cache_writers.append(_write)
    return ctx
```

- [ ] **Step 6: Run semantic cache tests — confirm pass**

```
pytest tests/test_caching.py -k "semantic_cache" -v
```

Expected: all pass (except `test_semantic_cache_no_embedder_skips_gracefully` which skips if ST is installed)

- [ ] **Step 7: Run full suite — confirm no regressions**

```
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add tokengate/layers/semantic_cache.py pyproject.toml tests/test_caching.py
git commit -m "feat: implement L2 semantic cache with LRU index, SQLite persistence, and verify_hit stub"
```

---

## Task 5: Server wiring + analytics + integration tests

**Files:**
- Modify: `tokengate/proxy/server.py`
- Modify: `tokengate/analytics/stats.py`

- [ ] **Step 1: Write failing integration tests — append to `tests/test_caching.py`**

```python
# ── Task 5: Integration tests (server wiring) ────────────────────────────────

import tokengate.proxy.server as _sv
from tokengate.core.mock_provider import MockTransport
from fastapi.testclient import TestClient


@pytest.fixture
def cache_client(tmp_path, monkeypatch):
    """TestClient with mock transport, temp data dir, and semantic index cleared."""
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.prices["mock"] = (1.0, 5.0)   # give the mock model a price so est_saved_usd is non-zero
    _sv._settings = s
    init_db(s.db_path)
    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    sem_cache._index.clear()
    sem_cache.set_embedder(None)
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, transport, s
    _sv._settings = None
    _sv._transport = None
    sem_cache._index.clear()
    sem_cache.set_embedder(None)


_ANT_BODY = {
    "model": "claude-haiku-4-5-20251001",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
}


def test_integration_exact_cache_hit(cache_client):
    """Identical non-streaming request × 2: only one upstream call, second has cache header."""
    client, transport, _ = cache_client

    r1 = client.post("/v1/messages", json=_ANT_BODY)
    assert r1.status_code == 200
    assert r1.headers["x-tokengate-cache"] == "none"

    r2 = client.post("/v1/messages", json=_ANT_BODY)
    assert r2.status_code == 200
    assert r2.headers["x-tokengate-cache"] == "exact"
    assert r2.headers["x-tokengate-saved-tokens"] != "0"

    assert len(transport.requests) == 1  # upstream called exactly once


def test_integration_exact_cache_miss_on_high_temperature(cache_client):
    """temperature=0.9 without opt-in header → never cached → transport called twice."""
    client, transport, _ = cache_client
    body = {**_ANT_BODY, "temperature": 0.9}

    client.post("/v1/messages", json=body)
    client.post("/v1/messages", json=body)

    assert len(transport.requests) == 2


def test_integration_exact_cache_est_saved_usd_in_db(cache_client):
    """On an exact cache hit the DB row records cache_kind='exact' and est_saved_usd > 0."""
    client, transport, s = cache_client

    client.post("/v1/messages", json=_ANT_BODY)  # miss → writes cache
    client.post("/v1/messages", json=_ANT_BODY)  # hit

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
    assert row["est_cost_usd"] == 0.0  # no upstream cost on a cache hit


def test_integration_semantic_cache_hit(cache_client, monkeypatch):
    """Semantic hit (score ≥ 0.97): second request served from index, header = 'semantic'."""
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
    assert len(transport.requests) == 1  # only original went to upstream


def test_integration_semantic_unverified_blocked_by_default(cache_client):
    """Similarity in unverified band with default serve_unverified=False → miss."""
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
    """Streaming requests are never served from cache."""
    client, transport, _ = cache_client

    body = {**_ANT_BODY, "stream": True}
    client.post("/v1/messages", json=body)
    client.post("/v1/messages", json=body)

    assert len(transport.requests) == 2


def test_integration_stats_cache_breakdown(cache_client):
    """/stats returns cache_by_kind with count and saved_usd for each kind used."""
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
```

- [ ] **Step 2: Run — confirm failure**

```
pytest tests/test_caching.py -k "integration" -v
```

Expected: `FAILED` — server doesn't pass settings to ctx, doesn't call writers, headers always "none"

- [ ] **Step 3: Update `tokengate/proxy/server.py`**

**3a.** Add import at the top (after existing layer imports):

```python
import tokengate.layers.semantic_cache as _l_semantic
```

(Already imported as `_l_semantic` — just confirm it's there. The existing import line is `import tokengate.layers.semantic_cache as _l_semantic`.)

**3b.** Add a helper after the `_PIPELINE` list:

```python
def _determine_cache_kind(decisions: list) -> str:
    """Read the first 'hit' decision and return the cache_kind string."""
    for d in decisions:
        if d.action == "hit":
            if d.layer == "exact_cache":
                return "exact"
            if d.layer == "semantic_cache":
                score = d.detail.get("score", 1.0)
                return "semantic" if score >= 0.97 else "semantic-unverified"
    return "none"
```

**3c.** Update `lifespan` to pass settings to index reload:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    check_startup(s)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(s.db_path)
    if _l_semantic.can_embed():
        _l_semantic.load_index(s.db_path, s.cache_max_entries)
    yield
```

**3d.** Update both endpoint handlers to pass `settings` into `LayerContext`:

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    s = get_settings()
    start_ts = time.time()
    body = await request.json()
    req = normalize_openai(body, dict(request.headers))
    ctx = LayerContext(request=req, settings=s)
    ctx = await _run_pipeline(ctx)
    return await _handle_request(req, ctx, s, start_ts)


@app.post("/v1/messages")
async def messages(request: Request):
    s = get_settings()
    start_ts = time.time()
    body = await request.json()
    req = normalize_anthropic(body, dict(request.headers))
    ctx = LayerContext(request=req, settings=s)
    ctx = await _run_pipeline(ctx)
    return await _handle_request(req, ctx, s, start_ts)
```

**3e.** Rewrite `_non_streaming_response` to call cache writers, fix headers, and record accurate analytics:

```python
async def _non_streaming_response(req, ctx: LayerContext, s: Settings, start_ts: float):
    status = "ok"
    error_detail = None
    resp_body: dict = {}
    tokens_in = tokens_out = 0
    model = req.model

    cache_kind = _determine_cache_kind(ctx.decisions)

    if ctx.response is not None:
        # Cache hit — restore from ctx.response, no upstream call
        tokens_in = ctx.response.tokens_in
        tokens_out = ctx.response.tokens_out
        model = ctx.response.model
        resp_body = ctx.response.raw_body
        est_cost = 0.0
        est_saved = compute_cost(model, tokens_in, tokens_out, s) or 0.0
    else:
        try:
            upstream_resp = await call_upstream(req, s, transport=_transport)
            tokens_in = upstream_resp.tokens_in
            tokens_out = upstream_resp.tokens_out
            model = upstream_resp.model
            resp_body = upstream_resp.raw_body
            # Call cache write-back callbacks
            for writer in ctx.cache_writers:
                await writer(upstream_resp)
        except UpstreamError as e:
            status = "upstream_error"
            error_detail = str(e)
            resp_body = e.body
        est_cost = compute_cost(model, tokens_in, tokens_out, s)
        est_saved = 0.0

    latency_ms = int((time.time() - start_ts) * 1000)
    write_row(
        s.db_path,
        ts=start_ts, route=req.route, status=status, error_detail=error_detail,
        layers_applied=[asdict(d) for d in ctx.decisions],
        tokens_in_raw=tokens_in or None, tokens_in_final=tokens_in or None,
        tokens_out=tokens_out or None, model_used=model,
        cache_kind=cache_kind,
        latency_ms=latency_ms,
        est_cost_usd=est_cost,
        est_saved_usd=est_saved,
    )

    saved_tokens = (tokens_in + tokens_out) if cache_kind != "none" else 0
    tg_headers = {
        "x-tokengate-cache": cache_kind,
        "x-tokengate-model": model,
        "x-tokengate-saved-tokens": str(saved_tokens),
    }
    http_status = 200 if status == "ok" else 502
    return JSONResponse(content=resp_body, status_code=http_status, headers=tg_headers)
```

- [ ] **Step 4: Update `tokengate/analytics/stats.py` — add `cache_by_kind` to `get_stats()`**

Inside `get_stats()`, after the `daily` list and before `con.close()`, add:

```python
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
```

And add `"cache_by_kind": cache_by_kind` to the returned dict:

```python
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
```

- [ ] **Step 5: Run integration tests — confirm pass**

```
pytest tests/test_caching.py -k "integration" -v
```

Expected: all pass

- [ ] **Step 6: Run full test suite — confirm no regressions**

```
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add tokengate/proxy/server.py tokengate/analytics/stats.py tests/test_caching.py
git commit -m "feat: wire cache layers into server pipeline; accurate cache_kind/est_saved_usd analytics"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Task that covers it |
|---|---|
| `LayerContext.settings` + `cache_writers` fields | Task 1 |
| `cache_exact` + `cache_semantic` DB tables | Task 1 |
| `body_json` includes `tokens_in`/`tokens_out` | Task 3 step 3, Task 4 step 5 |
| Settings cache fields + `serve_unverified` | Task 2 |
| L1 key includes full message array (system prompt + history) | Task 3 step 3 + collision test |
| L1 streaming bypass on READ | Task 3 test + impl |
| L1 temperature gate + opt-in header | Task 3 |
| L1 TTL expiry | Task 3 |
| L2 `_HAVE_EMBEDDER` guard, graceful skip | Task 4 |
| L2 `set_embedder()` for test injection | Task 4 |
| L2 `verify_hit()` stub at call site | Task 4 step 5 |
| L2 hit tiers: exact/semantic/semantic-unverified + score in detail | Task 4 |
| L2 blocklist regex | Task 4 |
| L2 tools bypass | Task 4 |
| L2 streaming bypass on READ | Task 4 |
| L2 write-back: in-memory + SQLite | Task 4 |
| L2 embedding stored as blob | Task 4 |
| L2 `load_index()` on startup | Task 4 + Task 5 lifespan |
| Server passes `settings` into `LayerContext` | Task 5 |
| Cache writers called after upstream success | Task 5 |
| `x-tokengate-cache` / `x-tokengate-saved-tokens` headers | Task 5 |
| `est_cost_usd=0` on hit, `est_saved_usd>0` on hit | Task 5 |
| `cache_by_kind` in `/stats` | Task 5 |
| Single-process limitation documented | In spec — no code needed |
| Streaming never served from cache (read bypass) | Both layers + integration test |

**All spec requirements covered. No placeholders. No contradictions.**
