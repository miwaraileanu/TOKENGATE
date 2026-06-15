# TokenGate Phase 2 — Caching Design

**Date:** 2026-06-15
**Phase:** 2 of 5
**Goal:** Replace the no-op `exact_cache` and `semantic_cache` layer stubs with working implementations, ship two infrastructure fixes that Phase 1 left as gaps, and cover all behaviour with offline tests.

---

## 0. Context

Phase 1 ships a passthrough proxy with six stubbed layers. All layer stubs do nothing except return `ctx`. Two structural gaps exist that must be fixed before any caching can work:

1. `LayerContext` has no `settings` field — layers cannot read config.
2. `write_row` always records `cache_kind="none"` and `est_saved_usd=0` — the analytics DB is blind to cache activity.

These are fixed first (Task 1), with tests, before any cache logic is added.

---

## 1. Infrastructure Fixes (Task 1)

### 1.1 `LayerContext` — add `settings` and `cache_writers`

```python
@dataclass
class LayerContext:
    request:       GatewayRequest
    response:      GatewayResponse | None = None
    decisions:     list[LayerDecision]   = field(default_factory=list)
    settings:      Settings | None       = None        # NEW
    cache_writers: list                  = field(default_factory=list)  # NEW
```

`settings` is populated by the server at construction:
```python
ctx = LayerContext(request=req, settings=get_settings())
```

`cache_writers` holds `async (response: GatewayResponse) -> None` callables. Cache layers append a writer on a miss. The server calls all writers after a successful non-streaming upstream response.

### 1.2 DB schema — add cache tables

Both tables are added to `analytics/db.py`'s `_SCHEMA` string so they are created at startup alongside `requests`. No migration needed — `CREATE TABLE IF NOT EXISTS` is idempotent.

```sql
CREATE TABLE IF NOT EXISTS cache_exact (
    cache_key  TEXT PRIMARY KEY,
    expires_at REAL NOT NULL,
    body_json  TEXT NOT NULL        -- JSON-serialised GatewayResponse fields
);
CREATE INDEX IF NOT EXISTS cache_exact_exp ON cache_exact(expires_at);

CREATE TABLE IF NOT EXISTS cache_semantic (
    cache_key  TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,       -- numpy float32 array, ndarray.tobytes()
    body_json  TEXT NOT NULL,       -- JSON-serialised GatewayResponse fields
    ts         REAL NOT NULL        -- unix timestamp, used for LRU rebuild on restart
);
CREATE INDEX IF NOT EXISTS cache_semantic_ts ON cache_semantic(ts);
```

### 1.3 `body_json` contract

Both tables store a JSON object with exactly these keys (derived from `GatewayResponse`):

```json
{
  "content":     "<assistant text>",
  "model":       "<model id>",
  "tokens_in":   10,
  "tokens_out":  5,
  "stop_reason": "end_turn",
  "raw_body":    { ... }
}
```

`tokens_in` and `tokens_out` are the **original upstream usage**. They are read back on cache hit and used both to populate `ctx.response` and to compute `est_saved_usd = compute_cost(model, tokens_in, tokens_out, settings)`.

### 1.4 `write_row` — record real cache activity

Server helper `_determine_cache_kind(decisions)` scans `ctx.decisions` for the first entry with `action="hit"` and maps it:

| `LayerDecision.layer` + `action` | `cache_kind` |
|---|---|
| `"exact_cache"` / `"hit"` | `"exact"` |
| `"semantic_cache"` / `"hit"`, score ≥ 0.97 | `"semantic"` |
| `"semantic_cache"` / `"hit"`, 0.93 ≤ score < 0.97 | `"semantic-unverified"` |
| no hit | `"none"` |

`est_saved_usd` is non-zero only when `cache_kind != "none"` and is computed from the restored response's token counts.

### 1.5 `x-tokengate-cache` and `x-tokengate-saved-tokens` headers

Set in `_non_streaming_response` after cache/upstream resolution:

```
x-tokengate-cache:        exact | semantic | semantic-unverified | none
x-tokengate-saved-tokens: <tokens_in + tokens_out>  (0 on miss)
```

---

## 2. Settings — new cache fields

Added to `Settings.__init__` from the `cache:` block in `tokengate.yaml`:

```python
_c = raw.get("cache", {})
self.cache_exact_ttl:        int   = int(_c.get("exact_ttl_seconds", 86400))
self.cache_semantic_threshold: float = float(_c.get("semantic_threshold", 0.93))
self.cache_max_entries:      int   = int(_c.get("max_entries", 50000))
self.cache_blocklist:        list[str] = _c.get("blocklist_patterns", [
    r"\btoday\b", r"\bnow\b", r"\blatest\b", r"\bprice\b",
])
self.cache_serve_unverified: bool = bool(_c.get("serve_unverified", False))
```

`tokengate.yaml` default config gains `serve_unverified: false` under the `cache:` block.

---

## 3. L1 — Exact Cache (`layers/exact_cache.py`)

### 3.1 Cache key

```python
def _compute_key(req: GatewayRequest) -> str:
    temp_bucket = round(req.temperature, 1) if req.temperature is not None else None
    payload = {
        "messages": req.messages,   # includes system prompt and full history
        "model":    req.model,
        "temp":     temp_bucket,
        "tools":    req.tools,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
```

`req.messages` contains the full turn history. For Anthropic requests `normalize_anthropic` prepends the system prompt as `{"role": "system", "content": ...}`, so two requests differing only in system prompt produce different keys. Same for any differing earlier turn.

### 3.2 Write eligibility

Cache is written only when **all** of these hold:
- `not ctx.request.stream`
- `req.temperature is None or req.temperature <= 0.3` **or** `req.raw_headers.get("x-tokengate-cache-write", "").lower() == "always"`

### 3.3 `apply()` flow

```
stream? → skip (decision: skip/streaming), return
write-eligible? → no → skip (decision: skip/no-write), return
compute key
query DB: SELECT body_json WHERE cache_key=? AND expires_at>now()
  hit → restore GatewayResponse, set ctx.response
        append LayerDecision("exact_cache", "hit", {"key": key})
        return ctx  ← pipeline short-circuits
  miss → append LayerDecision("exact_cache", "miss", {"key": key})
         push write callback onto ctx.cache_writers
         return ctx
```

### 3.4 Write callback

```python
async def _write(response: GatewayResponse) -> None:
    body_json = json.dumps({
        "content": response.content, "model": response.model,
        "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
        "stop_reason": response.stop_reason, "raw_body": response.raw_body,
    })
    expires_at = time.time() + ttl
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT OR REPLACE INTO cache_exact (cache_key, expires_at, body_json) VALUES (?,?,?)",
        (key, expires_at, body_json),
    )
    con.commit(); con.close()
```

No external dependencies.

---

## 4. L2 — Semantic Cache (`layers/semantic_cache.py`)

### 4.1 Dependencies

`sentence-transformers>=3` and `numpy>=1.26` added as `[project.optional-dependencies] semantic = [...]` in `pyproject.toml`. The import is guarded:

```python
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _HAVE_EMBEDDER = True
except ImportError:
    _HAVE_EMBEDDER = False
```

If `_HAVE_EMBEDDER` is `False`, every request gets `LayerDecision("semantic_cache", "skip", {"reason": "no_embedder"})` and the layer is transparent.

### 4.2 Embedder singleton + test injection

```python
_model: SentenceTransformer | None = None
_embed_fn = None  # injectable for tests

def set_embedder(fn) -> None:
    """Tests call this to inject a fake embedding function."""
    global _embed_fn
    _embed_fn = fn

def _embed(text: str) -> np.ndarray:
    global _model, _embed_fn
    if _embed_fn is not None:
        return _embed_fn(text)
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode([text], normalize_embeddings=True)[0]
```

### 4.3 Query text

```python
def _query_text(req: GatewayRequest) -> str:
    system_fp = last_user = ""
    for msg in req.messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            system_fp = (c if isinstance(c, str) else str(c))[:100]
        elif msg.get("role") == "user":
            c = msg.get("content", "")
            last_user = c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if p.get("type") == "text"
            )
    return f"{system_fp}\n{last_user}".strip()
```

### 4.4 In-memory index

```python
_index: OrderedDict[str, tuple[np.ndarray, str]] = OrderedDict()
# key → (l2-normalised embedding, body_json)
```

LRU eviction: when `len(_index) >= max_entries`, `_index.popitem(last=False)` removes the least-recently-used entry before inserting a new one.

Cosine similarity for normalised embeddings = `float(np.dot(a, b))`.

### 4.5 Startup index reload

`semantic_cache.py` exposes:

```python
def load_index(db_path: Path, max_entries: int) -> None:
    """Called by server lifespan after init_db. Rebuilds _index from SQLite."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT cache_key, embedding, body_json FROM cache_semantic ORDER BY ts ASC"
    ).fetchall()
    con.close()
    for key, emb_bytes, body_json in rows[-max_entries:]:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        _index[key] = (emb, body_json)
```

Server `lifespan` calls `load_index(s.db_path, s.cache_max_entries)` after `init_db`. No re-embedding on restart — the stored blob is the authoritative embedding.

### 4.6 Verifier interface (stub in Phase 2, real in Phase 4)

```python
async def verify_hit(query: str, cached_answer: str) -> bool:
    """
    Phase 2: always returns True (no-op stub).
    Phase 4: sends a micro-prompt to the cheap-tier model and returns True iff
    the model rates the cached answer as adequate for the new query.
    """
    return True
```

### 4.7 `apply()` flow

```
not _HAVE_EMBEDDER → skip (no_embedder)
stream?            → skip (streaming)        ← read bypass, not just write
tools non-empty?   → skip (has_tools)
blocklist match?   → skip (blocklisted)

query_text = _query_text(req)
query_emb  = _embed(query_text)

scan _index for best cosine score:
  score >= 0.97:
    ctx.response = restored GatewayResponse
    decision("hit", {"score": score, "key": key, "verified": True})
    LRU: move key to end of _index
    return ctx

  0.93 <= score < 0.97:
    verified = await verify_hit(query_text, cached_body["content"])
    if settings.cache_serve_unverified and verified:
      ctx.response = restored GatewayResponse
      decision("hit", {"score": score, "key": key, "verified": verified})
      LRU: move key to end of _index
      return ctx
    else:
      decision("miss", {"score": score, "reason": "unverified_blocked"})
      # fall through to write-back registration

  score < 0.93 (or no entries):
    decision("miss", {"best_score": score})

push write callback onto ctx.cache_writers
return ctx
```

### 4.8 Write callback

```python
async def _write(response: GatewayResponse) -> None:
    key = hashlib.sha256(query_text.encode()).hexdigest()[:32]
    body_json = json.dumps({...})  # same contract as exact cache
    emb_bytes = query_emb.astype(np.float32).tobytes()
    ts = time.time()

    # 1. Persist to SQLite
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT OR REPLACE INTO cache_semantic (cache_key, embedding, body_json, ts) VALUES (?,?,?,?)",
        (key, emb_bytes, body_json, ts),
    )
    con.commit(); con.close()

    # 2. Populate in-memory index (with LRU eviction)
    if key not in _index and len(_index) >= max_entries:
        _index.popitem(last=False)
    _index[key] = (query_emb, body_json)
    _index.move_to_end(key)
```

---

## 5. Server Changes (`proxy/server.py`)

### 5.1 `LayerContext` construction

```python
ctx = LayerContext(request=req, settings=get_settings())
```

### 5.2 Cache writer invocation

In `_non_streaming_response`, after `upstream_resp` is received:
```python
for writer in ctx.cache_writers:
    await writer(upstream_resp)
```

Writers are **not** called on upstream error or on streaming responses.

### 5.3 `lifespan` — startup index reload

```python
@asynccontextmanager
async def lifespan(app):
    s = get_settings()
    check_startup(s)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(s.db_path)
    if semantic_cache._HAVE_EMBEDDER:             # True only if sentence-transformers installed
        semantic_cache.load_index(s.db_path, s.cache_max_entries)
    yield
```

### 5.4 Header and analytics update

```python
cache_kind = _determine_cache_kind(ctx.decisions)
saved_tokens = (ctx.response.tokens_in + ctx.response.tokens_out
                if ctx.response and cache_kind != "none" else 0)
est_saved = (compute_cost(ctx.response.model,
                          ctx.response.tokens_in,
                          ctx.response.tokens_out, s)
             if cache_kind != "none" else None)

write_row(..., cache_kind=cache_kind, est_saved_usd=est_saved or 0.0)
```

---

## 6. Dashboard (`analytics/dashboard.html`)

The existing stacked-bar "savings by layer" chart gains `cache_kind` breakout. The `/stats` JSON endpoint returns counts for `"exact"`, `"semantic"`, and `"semantic-unverified"` separately so the frontend can render three cache bands.

---

## 7. Test Plan (`tests/test_caching.py`)

All tests use `MockTransport` and `tmp_data_dir` — fully offline.

| # | Test | Assertion |
|---|------|-----------|
| 1 | Identical non-streaming request × 2 | `MockTransport.requests` length == 1; second response `x-tokengate-cache: exact` |
| 2 | Same request, `temperature=0.9`, no opt-in header, × 2 | Transport called twice; no exact hit |
| 3 | Same last message, **different** system prompt × 2 | Transport called twice; distinct DB rows in `cache_exact`; no false hit |
| 4 | Semantic hit, injected similarity = 0.98 | Second response `x-tokengate-cache: semantic`; `decision.detail["score"] == 0.98` |
| 5 | Semantic hit, similarity = 0.94, `serve_unverified=False` (default) | Miss; transport called; no cached response served |
| 6 | Semantic hit, similarity = 0.94, `serve_unverified=True` | Hit; `x-tokengate-cache: semantic-unverified` |
| 7 | Query text contains "today" | Semantic layer skips; decision reason == "blocklisted" |
| 8 | Request with `tools=[...]` | Semantic layer skips; decision reason == "has_tools" |
| 9 | Streaming request | Both exact and semantic decisions have reason == "streaming"; transport called |
| 10 | `est_saved_usd` math | Cache hit → `est_saved_usd == compute_cost(model, tokens_in, tokens_out, settings)` |
| 11 | Semantic index reload | Write entry, clear `_index`, call `load_index`, assert entry in `_index` |
| 12 | `@pytest.mark.slow` paraphrase benchmark (20 pairs) | ≥ 80% hit rate with real embedder; 0 false hits on blocklist set |

---

## 8. `pyproject.toml` additions

```toml
[project.optional-dependencies]
semantic = ["sentence-transformers>=3.0", "numpy>=1.26"]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
    "anyio[trio]>=4",
    "numpy>=1.26",           # needed by test_caching.py fake embedder
]
```

---

## 9. Limitations (Phase 2)

- **Single process per data dir.** The in-memory semantic index is process-local. If two gateway processes share `~/.rait/` (e.g. `rait start` run twice on different ports), their in-memory indexes will diverge from each other and from the SQLite table. Concurrent writers to the semantic index are not supported in Phase 2.
- **Streaming requests are never read from or written to either cache.** Streaming cache write-back is a Phase 5 item.
- **Semantic re-rank (`verify_hit`) is a no-op stub.** Always returns `True`. The real cheap-model call is wired in Phase 4.

---

## 10. Acceptance Criteria

- [ ] `test_caching.py` passes fully offline (no real embedder required for tests 1–11)
- [ ] Second identical request returns in < 20 ms (no transport call)
- [ ] `x-tokengate-cache` header reflects the actual cache kind on every response
- [ ] `est_saved_usd` in the DB is non-zero on every cache hit and equals `compute_cost(model, tokens_in, tokens_out)`
- [ ] A query containing "today" never receives a semantic cache hit
- [ ] A streaming request is never served a cached body
- [ ] Semantic index is non-empty after server restart (loaded from SQLite)
- [ ] `serve_unverified: false` (default) causes 0.93–0.97 hits to fall through to upstream
