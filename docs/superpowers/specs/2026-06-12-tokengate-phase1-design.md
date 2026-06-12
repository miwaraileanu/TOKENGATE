# TokenGate Phase 1 Design Spec
_Date: 2026-06-12_

## Scope

Phase 1 ships a working passthrough gateway with zero optimization. Value: **visibility + one-command setup**.

Deliverables:
- FastAPI proxy supporting both OpenAI and Anthropic API shapes
- Streaming SSE pass-through with real token counts from the final usage event
- Full request logging to SQLite (one row per client request, including errors and cache hits)
- Admin dashboard + `/stats` JSON endpoint
- `rait` CLI: `install`, `start`, `stop`, `status`
- Full offline test suite using a mock provider
- All Phase 2–5 layer stubs wired into the pipeline via a shared `LayerContext`

---

## 1. Request Flow

```
Client
  │
  ▼
POST /v1/chat/completions   ──┐
POST /v1/messages            ─┤
                              ▼
                         normalize()
                              │
                              ▼
                        GatewayRequest
                              │
                    ┌─────────▼──────────┐
                    │  Layer pipeline     │
                    │  (ctx: LayerContext)│
                    │  1. exact_cache     │  ← stub Phase 1
                    │  2. semantic_cache  │  ← stub Phase 1
                    │  3. distiller       │  ← stub Phase 1
                    │  4. compressor      │  ← stub Phase 1
                    │  5. router          │  ← stub Phase 1
                    │  6. budgeter        │  ← stub Phase 1
                    └─────────┬──────────┘
                              │ (if ctx.response is None)
                              ▼
                        provider.py
                       (upstream call)
                              │
                              ▼
                    serialize response
                    append x-tokengate-* headers
                              │
                              ▼
                        analytics log
                              │
                              ▼
                           Client
```

**Auth middleware** runs before the pipeline:
- If bind address is not `127.0.0.1` AND `TOKENGATE_KEY` is unset → server refuses to start (`sys.exit(1)` in lifespan)
- If bind address is not loopback AND `X-TokenGate-Key` header is missing or wrong → `401` immediately, row logged with `status="auth_error"`
- Loopback connections: key check skipped

**Response headers** (always present, set before first byte):
- `x-tokengate-cache: none` (Phase 1 always; Phase 2 sets `exact`, `semantic`, `miss`)
- `x-tokengate-model: <model>` — the model that was actually used
- `x-tokengate-saved-tokens: 0` (Phase 1 always)

---

## 2. Core Data Structures (`core/`)

### `GatewayRequest` (normalize.py)
```python
@dataclass
class GatewayRequest:
    messages: list[dict]       # normalized to OpenAI message shape internally
    model: str
    stream: bool
    max_tokens: int | None
    temperature: float | None
    tools: list[dict]
    route: Literal["openai", "anthropic"]
    raw_headers: dict[str, str]
    extra: dict                # all unrecognized fields (top_p, stop_sequences,
                               # tool_choice, metadata, response_format, etc.)
```

Normalization: Anthropic `messages` + `system` → OpenAI-style with a leading `{"role": "system", ...}` message. Every field not explicitly listed above is captured into `extra` at parse time and re-applied verbatim when serializing the outbound upstream request. Re-serialized back to the client's original shape on the way out.

**Test:** a request containing `top_p=0.9` and `stop_sequences=["END"]` must arrive at the mock provider with both fields present and unchanged.

### `GatewayResponse` (normalize.py)
```python
@dataclass
class GatewayResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    stop_reason: str
    raw_body: dict       # original upstream JSON, relayed unchanged on error
```

### `LayerContext` (core/context.py)
```python
@dataclass
class LayerContext:
    request: GatewayRequest
    response: GatewayResponse | None = None
    decisions: list[LayerDecision] = field(default_factory=list)

@dataclass
class LayerDecision:
    layer: str           # "exact_cache", "semantic_cache", "router", etc.
    action: str          # "hit", "miss", "skip", "applied", "escalated"
    detail: dict         # layer-specific metadata (threshold, model chosen, etc.)
```

**Short-circuit rule:** if `ctx.response is not None` after any layer, the pipeline skips the remaining layers and skips the upstream call. The cache layers use this in Phase 2.

`decisions` is serialized to JSON for the `layers_applied` DB column and drives the `x-tokengate-*` headers.

### Layer stub interface (all Phase 2–5 layers)
```python
async def apply(ctx: LayerContext) -> LayerContext:
    return ctx  # no-op in Phase 1
```

Each layer file imports `LayerContext` from `core.context`. The pipeline calls all six in sequence.

---

## 3. Streaming

For streaming requests (`"stream": true`):

1. Set all `x-tokengate-*` headers in the HTTP response header (before body).
2. Open an `httpx.AsyncClient` stream to upstream.
3. Yield each SSE chunk to the client as it arrives (`StreamingResponse`).
4. On the **final chunk** (OpenAI: `data: [DONE]`; Anthropic: `event: message_stop` with `usage` block) extract token counts.
5. After the last byte is sent, write the analytics row with real token counts.
6. If the stream is interrupted (client disconnect / upstream error mid-stream), write the row with whatever token counts were observed + `status="upstream_error"`.

Token extraction per shape:
- **OpenAI:** last non-`[DONE]` chunk that contains `usage` → `usage.prompt_tokens`, `usage.completion_tokens`
- **Anthropic:** `message_delta` event with `usage.output_tokens` + earlier `message_start` with `usage.input_tokens`

---

## 4. Analytics DB (`analytics/db.py`)

Database path: `~/.rait/tokengate.db`

**Init pragmas:**
```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
```

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY,
    ts            REAL    NOT NULL,          -- Unix timestamp of client request
    route         TEXT    NOT NULL,          -- 'openai' | 'anthropic'
    status        TEXT    NOT NULL,          -- 'ok' | 'upstream_error' | 'auth_error'
    error_detail  TEXT,                      -- NULL on success
    layers_applied TEXT   NOT NULL DEFAULT '[]',  -- JSON array of LayerDecision dicts
    tokens_in_raw INTEGER,                   -- from upstream usage (NULL on cache hit)
    tokens_in_final INTEGER,                 -- after distillation (0 on cache hit)
    tokens_out    INTEGER,
    model_used    TEXT,
    cache_kind    TEXT    NOT NULL DEFAULT 'none',  -- 'none' | 'exact' | 'semantic'
    escalated     INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    est_cost_usd  REAL,                      -- NULL if model price unknown
    est_saved_usd REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
```

**Semantics:** one row per **client request**. Cache hits (Phase 2) still get a row with `tokens_in_raw = NULL`, `tokens_in_final = 0`, `cache_kind` set, `est_cost_usd = 0.0`, `est_saved_usd` reflecting the avoided cost.

---

## 5. Price Table (`core/tokens.py`)

Default price table (per 1M tokens, in USD) lives as a `dict[str, tuple[float, float]]` keyed by model name → `(input_price, output_price)`.

Overrides in `tokengate.yaml`:
```yaml
prices:
  claude-haiku-4-5: [0.80, 4.00]
  my-custom-model:  [1.00, 5.00]
```

On startup, YAML prices are merged over defaults.

**Unknown model rule:** if a model name is not in the merged table, `est_cost_usd = NULL` (not `0`). Dashboard shows `"unknown"` in cost cells where `est_cost_usd IS NULL`. Unit test verifies this.

---

## 6. Dashboard & Stats

**`GET /stats`** — JSON:
```json
{
  "total_requests": 1234,
  "total_tokens_in": 500000,
  "total_tokens_out": 120000,
  "total_est_cost_usd": 4.21,
  "total_est_saved_usd": 0.0,
  "cache_hit_rate": 0.0,
  "requests_by_status": {"ok": 1200, "upstream_error": 34},
  "daily": [{"date": "2026-06-12", "requests": 80, "tokens_in": 40000, ...}]
}
```

**`GET /dashboard`** — static HTML page served by FastAPI. Vanilla JS polls `/stats` every 10s. Chart.js via CDN renders:
- Total requests / tokens / cost cards
- Daily request volume (line chart)
- Cost by day (bar chart)
- Recent requests table (last 50, with status, model, latency, cost)
- `"unknown"` shown in cost cells where `est_cost_usd` is null

---

## 7. CLI (`cli/`)

### `cli/main.py` — typer app, commands: `install`, `start`, `stop`, `status`

### `cli/wizard.py` — `rait install`
1. Print ASCII banner + version
2. Ask provider (Anthropic / OpenAI / both)
3. Read API key (hidden input) → write `~/.rait/.env` with owner-only permissions (Unix: `chmod 600`; Windows: `icacls` / `stat` ACL)
4. Print: _"API key stored but not validated. Run `rait test --live` (coming in a later release) to verify."_
5. Ask port (default 8787)
6. Write `~/.rait/tokengate.yaml` from defaults + user choices
7. Init SQLite DB
8. Run self-test against mock provider, print per-step checkmarks
9. Print integration snippet (`base_url="http://localhost:<port>"`)
10. Offer `rait start --detach`

All inputs overridable via flags: `--provider`, `--port`, `--yes` (accept all defaults non-interactively).

### `cli/daemon.py` — cross-platform process management

**PID file** at `~/.rait/tokengate.pid` — JSON:
```json
{"pid": 12345, "port": 8787, "started_at": 1749734400.0}
```

**`rait start`:**
1. Check port is free (`socket.bind()` probe) — if busy, exit with: _"Port 8787 in use — is TokenGate already running? Try `rait status`."_
2. Foreground mode (default): write PID file (`{pid: os.getpid(), port, started_at}`), register `atexit` handler and `SIGINT` handler to delete it on clean exit, then call `uvicorn.run()` directly. This allows `rait status` and `rait stop` to work from a second terminal.
3. `--detach` mode:
   - Unix: `subprocess.Popen([sys.executable, "-m", "tokengate.proxy.server", ...], start_new_session=True, stdout/stderr → ~/.rait/logs/tokengate.log)`
   - Windows: `subprocess.Popen(..., creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)`
   - Write PID file after successful spawn (parent process writes it; child owns cleanup via same atexit/signal handler)

**`rait stop`:**
1. Read PID file → if missing, print "Not running" and exit 0
2. Check process alive (psutil if available, else `os.kill(pid, 0)`)
3. If dead (stale PID): print "Not running (stale PID file removed)", delete file, exit 0
4. Optional cmdline verification via psutil: confirm `"tokengate"` in cmdline before killing
5. Send `SIGTERM` (Unix) / `os.kill(pid, signal.CTRL_BREAK_EVENT)` (Windows)
6. Delete PID file

**`rait status`:**
1. If no PID file → print "Not running"
2. Check process alive; if stale → remove file, print "Not running (stale PID file removed)"
3. If alive → print: running, port (from PID JSON), uptime (`now - started_at`), DB path, DB size

---

## 8. Auth & Security

**Startup check** (FastAPI lifespan):
```python
if settings.bind != "127.0.0.1" and not settings.tokengate_key:
    logger.error("TOKENGATE_KEY must be set when binding to non-loopback. Refusing to start.")
    sys.exit(1)
```

**Middleware** (runs before every request):
```python
if not is_loopback(request.client.host):
    if request.headers.get("X-TokenGate-Key") != settings.tokengate_key:
        log_auth_error(request)
        return JSONResponse({"error": "unauthorized"}, status_code=401)
```

`is_loopback()` checks IPv4 `127.x.x.x` and IPv6 `::1`.

---

## 9. Mock Provider (`core/mock_provider.py`)

An `httpx` transport (or small ASGI app mounted at a test base URL) that handles:

- **Normal mode**: returns a fixed response with real `usage` fields
- **Streaming mode**: yields proper SSE chunks including a final `usage` event
- **Error mode**: configurable per-request — returns 500 or 429 with upstream error body

Activated in tests by passing `transport=MockTransport(mode="normal"|"error", status=500|429)` to the `httpx.AsyncClient`.

---

## 10. Test Suite (`tests/`)

| File | What it covers |
|---|---|
| `conftest.py` | Mock transport fixture, temp data dir, `TestClient` |
| `test_proxy.py` | Both API shapes normalize and pass through; response headers present; streaming assembles correctly; extra fields (`top_p`, `stop_sequences`) relayed to upstream unchanged |
| `test_analytics.py` | One request → one DB row; columns populated; unknown model → `est_cost_usd = NULL`; error request → `status="upstream_error"` + `error_detail` set |
| `test_streaming.py` | Token counts come from final usage event; analytics row written after stream ends; mid-stream disconnect → row written with partial data + error status |
| `test_security.py` | Start with `bind=0.0.0.0` + no key → refuses; with key set, non-loopback request without header → 401; non-loopback with correct key → 200 |
| `test_cli.py` | `rait install --yes`; `rait status` with no PID; stale PID cleanup; port-busy error message |
| `test_tokens.py` | Price table defaults; YAML override merges correctly; unknown model → NULL |

All tests pass with **zero real API calls** (mock provider only).

---

## 11. Project File Tree (Phase 1 + stubs)

```
tokengate/
├── pyproject.toml
├── tokengate.yaml                    # default config, copied to ~/.rait/ on install
├── proxy/
│   ├── __init__.py
│   └── server.py                     # FastAPI app, lifespan, endpoints, middleware
├── core/
│   ├── __init__.py
│   ├── context.py                    # LayerContext, LayerDecision
│   ├── normalize.py                  # GatewayRequest, GatewayResponse, both normalizers
│   ├── provider.py                   # httpx upstream client (streaming + non-streaming)
│   ├── mock_provider.py              # test transport
│   └── tokens.py                     # price table, token extraction helpers
├── layers/
│   ├── __init__.py
│   ├── exact_cache.py                # stub: apply(ctx) -> ctx
│   ├── semantic_cache.py             # stub
│   ├── distiller.py                  # stub
│   ├── compressor.py                 # stub
│   ├── router.py                     # stub
│   └── budgeter.py                   # stub
├── analytics/
│   ├── __init__.py
│   ├── db.py                         # init, WAL, schema, write_row()
│   ├── stats.py                      # /stats aggregation queries
│   └── dashboard.html                # static page, polls /stats, Chart.js
├── cli/
│   ├── __init__.py
│   ├── main.py                       # typer app, command registration
│   ├── wizard.py                     # rait install flow
│   └── daemon.py                     # start/stop/status, PID file, port check
├── scripts/
│   ├── install.sh                    # one-liner bootstrap (Phase 5, stub for now)
│   └── retrain_router.py             # Phase 4, stub for now
├── tests/
│   ├── conftest.py
│   ├── test_proxy.py
│   ├── test_analytics.py
│   ├── test_streaming.py
│   ├── test_security.py
│   ├── test_cli.py
│   └── test_tokens.py
└── Dockerfile                        # Phase 5, stub for now
```

---

## 12. Out of Scope for Phase 1

- L1/L2 caches (Phase 2)
- Context distillation, prompt compression (Phase 3)
- Cascade router + escalation learning (Phase 4)
- Rate limiting, request size caps, config hot-reload, autostart units, `rait update/uninstall` (Phase 5)
- `rait test --live` (real API key validation)
- `scripts/install.sh` one-liner (Phase 5)
