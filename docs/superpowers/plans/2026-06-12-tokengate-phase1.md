# TokenGate Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a passthrough LLM gateway with full request logging, a live dashboard, and a cross-platform `rait` CLI — providing visibility and one-command setup before any optimization layers are added.

**Architecture:** FastAPI proxy normalizes OpenAI and Anthropic requests into a shared `GatewayRequest`, runs them through a six-layer pipeline (all stubs in Phase 1), forwards to upstream via `httpx`, logs every request to SQLite, and exposes a `/dashboard` and `/stats` endpoint. The `rait` CLI manages the gateway process via a JSON PID file.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx, typer, rich, PyYAML, psutil, pytest, pytest-asyncio, SQLite3 (stdlib)

---

## File Map

```
tokengate/                  ← Python package root
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py           ← Settings class (reads tokengate.yaml + env vars)
│   ├── context.py          ← LayerContext, LayerDecision dataclasses
│   ├── normalize.py        ← GatewayRequest, GatewayResponse, normalizers, serializers
│   ├── tokens.py           ← price table, cost compute, usage extraction, streaming parser
│   ├── provider.py         ← httpx upstream client (non-streaming + streaming)
│   └── mock_provider.py    ← MockTransport for offline tests
├── layers/
│   ├── __init__.py
│   ├── exact_cache.py      ← stub: apply(ctx) -> ctx
│   ├── semantic_cache.py   ← stub
│   ├── distiller.py        ← stub
│   ├── compressor.py       ← stub
│   ├── router.py           ← stub
│   └── budgeter.py         ← stub
├── analytics/
│   ├── __init__.py
│   ├── db.py               ← init_db(), write_row()
│   ├── stats.py            ← get_stats() aggregation queries
│   └── dashboard.html      ← static page, polls /stats, Chart.js via CDN
├── proxy/
│   ├── __init__.py
│   └── server.py           ← FastAPI app, middleware, endpoints, layer pipeline runner
└── cli/
    ├── __init__.py
    ├── main.py             ← typer app, registers all commands
    ├── wizard.py           ← rait install flow
    └── daemon.py           ← PID file, port check, start/stop/status logic

pyproject.toml
tokengate.yaml              ← default config (copied to ~/.rait/ on install)
scripts/
├── install.sh              ← stub comment only
└── retrain_router.py       ← stub comment only
Dockerfile                  ← stub comment only
tests/
├── conftest.py
├── test_context.py
├── test_normalize.py
├── test_tokens.py
├── test_analytics.py
├── test_proxy.py
├── test_streaming.py
├── test_security.py
└── test_cli.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `tokengate.yaml`
- Create: all `__init__.py` files
- Create: all layer stubs
- Create: `scripts/install.sh`, `scripts/retrain_router.py`, `Dockerfile`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rait-tokengate"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "httpx>=0.28",
    "typer>=0.15",
    "rich>=13",
    "pyyaml>=6",
    "psutil>=6",
]

[project.scripts]
rait = "tokengate.cli.main:app"

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
    "anyio[trio]>=4",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["tokengate"]
```

- [ ] **Step 2: Create `tokengate.yaml` (default config)**

```yaml
bind: "127.0.0.1"
port: 8787
log_level: "info"

openai_base_url: "https://api.openai.com"
anthropic_base_url: "https://api.anthropic.com"

tiers:
  - name: cheap
    model: "claude-haiku-4-5-20251001"
    max_difficulty: 0.4
  - name: strong
    model: "claude-sonnet-4-6"
    max_difficulty: 1.0

cache:
  exact_ttl_seconds: 86400
  semantic_threshold: 0.93
  max_entries: 50000
  blocklist_patterns:
    - "\\btoday\\b"
    - "\\bnow\\b"
    - "\\blatest\\b"
    - "\\bprice\\b"

distill_threshold_tokens: 6000
keep_recent_turns: 4

budgets:
  chat: 2048
  code: 4096
  extraction: 512
  long_form: 8192

# prices:
#   my-custom-model: [1.00, 5.00]
```

- [ ] **Step 3: Create package skeleton directories and `__init__.py` files**

```bash
mkdir -p tokengate/core tokengate/layers tokengate/analytics tokengate/proxy tokengate/cli
mkdir -p scripts tests
```

Create `tokengate/__init__.py`:
```python
__version__ = "0.1.0"
```

Create empty `__init__.py` in each subpackage:
- `tokengate/core/__init__.py` — empty
- `tokengate/layers/__init__.py` — empty
- `tokengate/analytics/__init__.py` — empty
- `tokengate/proxy/__init__.py` — empty
- `tokengate/cli/__init__.py` — empty

- [ ] **Step 4: Create layer stubs**

Each of these six files has identical content — create them all:

`tokengate/layers/exact_cache.py`, `tokengate/layers/semantic_cache.py`, `tokengate/layers/distiller.py`, `tokengate/layers/compressor.py`, `tokengate/layers/router.py`, `tokengate/layers/budgeter.py`:

```python
from __future__ import annotations
from tokengate.core.context import LayerContext


async def apply(ctx: LayerContext) -> LayerContext:
    """Phase 1 stub — no-op pass-through."""
    return ctx
```

- [ ] **Step 5: Create script and Dockerfile stubs**

`scripts/install.sh`:
```bash
#!/usr/bin/env bash
# Phase 5: one-line bootstrap installer — not yet implemented.
echo "Install script not yet implemented. Use: pip install rait-tokengate"
exit 1
```

`scripts/retrain_router.py`:
```python
# Phase 4: weekly logistic regression retraining — not yet implemented.
raise NotImplementedError("Router retraining is a Phase 4 feature.")
```

`Dockerfile`:
```dockerfile
# Phase 5: Docker packaging — not yet implemented.
```

- [ ] **Step 6: Install the package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected output: no errors, `rait` command registered.

- [ ] **Step 7: Verify imports work**

```bash
python -c "import tokengate; import tokengate.layers.exact_cache"
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "chore: project scaffold — package skeleton, stubs, pyproject.toml"
```

---

## Task 2: LayerContext

**Files:**
- Create: `tokengate/core/context.py`
- Create: `tests/test_context.py`

- [ ] **Step 1: Write the failing test**

`tests/test_context.py`:
```python
from tokengate.core.context import LayerContext, LayerDecision


def test_layer_context_defaults():
    # Minimal construction — request is any object in Phase 1 tests
    ctx = LayerContext(request=object())
    assert ctx.response is None
    assert ctx.decisions == []


def test_layer_decision_stores_fields():
    d = LayerDecision(layer="exact_cache", action="miss", detail={"key": "abc123"})
    assert d.layer == "exact_cache"
    assert d.action == "miss"
    assert d.detail == {"key": "abc123"}


def test_layer_context_short_circuit():
    sentinel = object()
    ctx = LayerContext(request=object(), response=sentinel)
    assert ctx.response is sentinel


def test_decisions_list_is_independent():
    ctx1 = LayerContext(request=object())
    ctx2 = LayerContext(request=object())
    ctx1.decisions.append(LayerDecision(layer="x", action="hit"))
    assert ctx2.decisions == []
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/test_context.py -v
```

Expected: `ModuleNotFoundError` or similar (file doesn't exist yet).

- [ ] **Step 3: Create `tokengate/core/context.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokengate.core.normalize import GatewayRequest, GatewayResponse


@dataclass
class LayerDecision:
    layer: str
    action: str  # "hit" | "miss" | "skip" | "applied" | "escalated"
    detail: dict = field(default_factory=dict)


@dataclass
class LayerContext:
    request: GatewayRequest
    response: GatewayResponse | None = None
    decisions: list[LayerDecision] = field(default_factory=list)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_context.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tokengate/core/context.py tests/test_context.py
git commit -m "feat: LayerContext and LayerDecision dataclasses"
```

---

## Task 3: Request Normalization

**Files:**
- Create: `tokengate/core/normalize.py`
- Create: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_normalize.py`:
```python
import pytest
from tokengate.core.normalize import (
    GatewayRequest, GatewayResponse,
    normalize_openai, normalize_anthropic, serialize_for_upstream,
)


# ── OpenAI normalization ──────────────────────────────────────────────────────

def test_normalize_openai_basic():
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    req = normalize_openai(body, {})
    assert req.route == "openai"
    assert req.model == "gpt-4o"
    assert req.stream is False
    assert req.messages == [{"role": "user", "content": "hi"}]
    assert req.extra == {}


def test_normalize_openai_extra_fields():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 0.9,
        "stop_sequences": ["END"],
        "response_format": {"type": "json_object"},
    }
    req = normalize_openai(body, {})
    assert req.extra == {"top_p": 0.9, "stop_sequences": ["END"], "response_format": {"type": "json_object"}}


def test_serialize_openai_roundtrip_with_extra():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["END"],
    }
    req = normalize_openai(body, {})
    out = serialize_for_upstream(req)
    assert out["top_p"] == 0.9
    assert out["stop_sequences"] == ["END"]
    assert out["temperature"] == 0.7


# ── Anthropic normalization ───────────────────────────────────────────────────

def test_normalize_anthropic_injects_system():
    body = {
        "model": "claude-sonnet-4-6",
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
    }
    req = normalize_anthropic(body, {})
    assert req.route == "anthropic"
    assert req.messages[0] == {"role": "system", "content": "You are helpful."}
    assert req.messages[1] == {"role": "user", "content": "hi"}
    assert req.max_tokens == 1024


def test_normalize_anthropic_extra_fields():
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "top_p": 0.95,
        "metadata": {"user_id": "u123"},
    }
    req = normalize_anthropic(body, {})
    assert req.extra == {"top_p": 0.95, "metadata": {"user_id": "u123"}}


def test_serialize_anthropic_roundtrip():
    body = {
        "model": "claude-sonnet-4-6",
        "system": "Be concise.",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "top_p": 0.95,
    }
    req = normalize_anthropic(body, {})
    out = serialize_for_upstream(req)
    assert out["system"] == "Be concise."
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["top_p"] == 0.95


def test_no_system_anthropic():
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    req = normalize_anthropic(body, {})
    out = serialize_for_upstream(req)
    assert "system" not in out


def test_raw_headers_stored():
    body = {"model": "x", "messages": [], "stream": False}
    req = normalize_openai(body, {"authorization": "Bearer sk-123"})
    assert req.raw_headers["authorization"] == "Bearer sk-123"
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_normalize.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `tokengate/core/normalize.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GatewayRequest:
    messages: list[dict]
    model: str
    stream: bool
    max_tokens: int | None
    temperature: float | None
    tools: list[dict]
    route: Literal["openai", "anthropic"]
    raw_headers: dict[str, str]
    extra: dict = field(default_factory=dict)


@dataclass
class GatewayResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    stop_reason: str
    raw_body: dict


_OAI_KNOWN = frozenset({"messages", "model", "stream", "max_tokens", "temperature", "tools"})
_ANT_KNOWN = frozenset({"messages", "model", "stream", "max_tokens", "temperature", "tools", "system"})


def normalize_openai(body: dict, headers: dict[str, str]) -> GatewayRequest:
    return GatewayRequest(
        messages=list(body.get("messages", [])),
        model=body.get("model", ""),
        stream=bool(body.get("stream", False)),
        max_tokens=body.get("max_tokens"),
        temperature=body.get("temperature"),
        tools=list(body.get("tools", [])),
        route="openai",
        raw_headers=dict(headers),
        extra={k: v for k, v in body.items() if k not in _OAI_KNOWN},
    )


def normalize_anthropic(body: dict, headers: dict[str, str]) -> GatewayRequest:
    messages: list[dict] = []
    if "system" in body:
        messages.append({"role": "system", "content": body["system"]})
    messages.extend(body.get("messages", []))
    return GatewayRequest(
        messages=messages,
        model=body.get("model", ""),
        stream=bool(body.get("stream", False)),
        max_tokens=body.get("max_tokens"),
        temperature=body.get("temperature"),
        tools=list(body.get("tools", [])),
        route="anthropic",
        raw_headers=dict(headers),
        extra={k: v for k, v in body.items() if k not in _ANT_KNOWN},
    )


def serialize_for_upstream(req: GatewayRequest) -> dict[str, Any]:
    if req.route == "openai":
        body: dict[str, Any] = {
            "messages": req.messages,
            "model": req.model,
            "stream": req.stream,
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = req.tools
    else:
        msgs = req.messages
        body = {"model": req.model, "stream": req.stream}
        if msgs and msgs[0].get("role") == "system":
            body["system"] = msgs[0]["content"]
            msgs = msgs[1:]
        body["messages"] = msgs
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = req.tools
    body.update(req.extra)
    return body
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_normalize.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tokengate/core/normalize.py tests/test_normalize.py
git commit -m "feat: GatewayRequest/Response normalization for both API shapes"
```

---

## Task 4: Config and Token Pricing

**Files:**
- Create: `tokengate/core/config.py`
- Create: `tokengate/core/tokens.py`
- Create: `tests/test_tokens.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_tokens.py`:
```python
import os
import pytest
from pathlib import Path
from tokengate.core.config import Settings, _DEFAULT_PRICES
from tokengate.core.tokens import compute_cost, extract_usage_openai, extract_usage_anthropic, parse_streaming_usage


@pytest.fixture
def tmp_settings(tmp_path):
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    yield s
    del os.environ["TOKENGATE_DATA_DIR"]


def test_default_prices_loaded(tmp_settings):
    assert "claude-sonnet-4-6" in tmp_settings.prices
    assert "gpt-4o" in tmp_settings.prices


def test_yaml_price_override(tmp_path):
    cfg = tmp_path / "tokengate.yaml"
    cfg.write_text("prices:\n  my-model: [1.00, 5.00]\n")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    del os.environ["TOKENGATE_DATA_DIR"]
    assert s.prices["my-model"] == (1.00, 5.00)
    # defaults still present
    assert "gpt-4o" in s.prices


def test_compute_cost_known_model(tmp_settings):
    # claude-sonnet-4-6: input=3.00, output=15.00 per 1M
    cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, tmp_settings)
    assert cost == pytest.approx(18.00)


def test_compute_cost_unknown_model_returns_none(tmp_settings):
    cost = compute_cost("unknown-model-xyz", 100, 50, tmp_settings)
    assert cost is None


def test_extract_usage_openai():
    body = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    assert extract_usage_openai(body) == (10, 5)


def test_extract_usage_anthropic():
    body = {"usage": {"input_tokens": 12, "output_tokens": 8}}
    assert extract_usage_anthropic(body) == (12, 8)


def test_parse_streaming_usage_openai():
    text = (
        'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        'data: [DONE]\n\n'
    )
    assert parse_streaming_usage(text, "openai") == (10, 5)


def test_parse_streaming_usage_anthropic():
    text = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
        'event: message_stop\n'
        'data: {"type":"message_stop"}\n\n'
    )
    assert parse_streaming_usage(text, "anthropic") == (10, 5)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_tokens.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `tokengate/core/config.py`**

```python
from __future__ import annotations
import os
import yaml
from pathlib import Path


_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
}


class Settings:
    def __init__(self, config_path: Path | str | None = None):
        data_dir_env = os.environ.get("TOKENGATE_DATA_DIR")
        self.data_dir = (
            Path(data_dir_env).expanduser()
            if data_dir_env
            else Path("~/.rait").expanduser()
        )

        raw: dict = {}
        cfg = Path(config_path) if config_path else (self.data_dir / "tokengate.yaml")
        if cfg.exists():
            with open(cfg) as f:
                raw = yaml.safe_load(f) or {}

        self.bind: str = os.environ.get("TOKENGATE_BIND", raw.get("bind", "127.0.0.1"))
        self.port: int = int(os.environ.get("TOKENGATE_PORT", str(raw.get("port", 8787))))
        self.tokengate_key: str = os.environ.get("TOKENGATE_KEY", raw.get("tokengate_key", ""))
        self.log_level: str = raw.get("log_level", "info")
        self.openai_base_url: str = raw.get("openai_base_url", "https://api.openai.com")
        self.anthropic_base_url: str = raw.get("anthropic_base_url", "https://api.anthropic.com")

        yaml_prices = raw.get("prices", {})
        self.prices: dict[str, tuple[float, float]] = {
            **_DEFAULT_PRICES,
            **{k: tuple(v) for k, v in yaml_prices.items()},
        }

        self.db_path: Path = self.data_dir / "tokengate.db"
        self.pid_path: Path = self.data_dir / "tokengate.pid"
        self.log_path: Path = self.data_dir / "logs" / "tokengate.log"
```

- [ ] **Step 4: Create `tokengate/core/tokens.py`**

```python
from __future__ import annotations
import json
from tokengate.core.config import Settings


def compute_cost(
    model: str, tokens_in: int, tokens_out: int, settings: Settings
) -> float | None:
    """Returns cost in USD, or None if model is not in the price table."""
    if model not in settings.prices:
        return None
    input_price, output_price = settings.prices[model]
    return (tokens_in * input_price + tokens_out * output_price) / 1_000_000


def extract_usage_openai(body: dict) -> tuple[int, int]:
    u = body.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def extract_usage_anthropic(body: dict) -> tuple[int, int]:
    u = body.get("usage", {})
    return u.get("input_tokens", 0), u.get("output_tokens", 0)


def parse_streaming_usage(full_text: str, route: str) -> tuple[int, int]:
    """Parse (tokens_in, tokens_out) from accumulated SSE text."""
    if route == "openai":
        tokens_in = tokens_out = 0
        for line in full_text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        tokens_in = u.get("prompt_tokens", tokens_in)
                        tokens_out = u.get("completion_tokens", tokens_out)
                except (json.JSONDecodeError, KeyError):
                    pass
        return tokens_in, tokens_out
    else:  # anthropic
        tokens_in = tokens_out = 0
        for line in full_text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
                if event.get("type") == "message_start":
                    u = event.get("message", {}).get("usage", {})
                    tokens_in = u.get("input_tokens", tokens_in)
                elif event.get("type") == "message_delta":
                    u = event.get("usage", {})
                    tokens_out = u.get("output_tokens", tokens_out)
            except (json.JSONDecodeError, KeyError):
                pass
        return tokens_in, tokens_out
```

- [ ] **Step 5: Run — expect pass**

```bash
pytest tests/test_tokens.py -v
```

Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add tokengate/core/config.py tokengate/core/tokens.py tests/test_tokens.py
git commit -m "feat: Settings config loader and token price/usage utilities"
```

---

## Task 5: Mock Provider and Test Fixtures

**Files:**
- Create: `tokengate/core/mock_provider.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tokengate/core/mock_provider.py`**

```python
from __future__ import annotations
import json
from typing import Literal

import httpx


_OAI_BODY = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Mock response"}, "finish_reason": "stop"}],
    "model": "mock",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_ANT_BODY = {
    "id": "msg_mock",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Mock response"}],
    "model": "mock",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

_OAI_STREAM = (
    'data: {"id":"chatcmpl-mock","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Mock"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-mock","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"content":" response"},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
    'data: [DONE]\n\n'
)

_ANT_STREAM = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_mock","type":"message",'
    '"role":"assistant","content":[],"model":"mock","stop_reason":null,'
    '"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Mock response"}}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        mode: Literal["normal", "error"] = "normal",
        error_status: int = 500,
    ):
        self.mode = mode
        self.error_status = error_status
        # Records every request body received — useful in tests
        self.requests: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)

        if self.mode == "error":
            err = json.dumps({"error": {"message": "upstream error", "type": "server_error"}})
            return httpx.Response(
                self.error_status,
                content=err.encode(),
                headers={"content-type": "application/json"},
            )

        is_anthropic = "/v1/messages" in str(request.url)
        is_streaming = body.get("stream", False)

        if is_streaming:
            content = (_ANT_STREAM if is_anthropic else _OAI_STREAM).encode()
            return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

        resp = {**(_ANT_BODY if is_anthropic else _OAI_BODY), "model": body.get("model", "mock")}
        return httpx.Response(200, json=resp)
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
from __future__ import annotations
import os
import pytest
from pathlib import Path
from tokengate.core.mock_provider import MockTransport


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Sets TOKENGATE_DATA_DIR to a temp path for the duration of the test."""
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def mock_transport():
    return MockTransport(mode="normal")


@pytest.fixture
def error_transport():
    return MockTransport(mode="error", error_status=500)


@pytest.fixture
def error_429_transport():
    return MockTransport(mode="error", error_status=429)
```

- [ ] **Step 3: Smoke-test the mock transport**

```bash
python -c "
import asyncio, httpx
from tokengate.core.mock_provider import MockTransport

async def main():
    t = MockTransport()
    client = httpx.AsyncClient(transport=t, base_url='https://api.openai.com')
    r = await client.post('/v1/chat/completions', json={'model':'gpt-4o','messages':[],'stream':False})
    print(r.status_code, r.json()['choices'][0]['message']['content'])

asyncio.run(main())
"
```

Expected: `200 Mock response`

- [ ] **Step 4: Commit**

```bash
git add tokengate/core/mock_provider.py tests/conftest.py
git commit -m "feat: MockTransport for offline testing (normal/streaming/error modes)"
```

---

## Task 6: Analytics Database

**Files:**
- Create: `tokengate/analytics/db.py`
- Create: `tests/test_analytics.py`

- [ ] **Step 1: Write failing tests**

`tests/test_analytics.py`:
```python
import sqlite3
import time
import pytest
from tokengate.analytics.db import init_db, write_row


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
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_analytics.py -v
```

- [ ] **Step 3: Create `tokengate/analytics/db.py`**

```python
from __future__ import annotations
import json
import sqlite3
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
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(_SCHEMA)
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
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_analytics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tokengate/analytics/db.py tests/test_analytics.py
git commit -m "feat: SQLite analytics DB — WAL mode, schema, write_row()"
```

---

## Task 7: Stats Aggregation

**Files:**
- Create: `tokengate/analytics/stats.py`
- Extend: `tests/test_analytics.py`

- [ ] **Step 1: Add tests to `tests/test_analytics.py`**

Append to the file:
```python
from tokengate.analytics.stats import get_stats
import time as _time


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
```

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_analytics.py::test_get_stats_empty -v
```

- [ ] **Step 3: Create `tokengate/analytics/stats.py`**

```python
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
    }
```

- [ ] **Step 4: Run all analytics tests**

```bash
pytest tests/test_analytics.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tokengate/analytics/stats.py tests/test_analytics.py
git commit -m "feat: /stats aggregation queries"
```

---

## Task 8: Upstream Provider (Non-Streaming)

**Files:**
- Create: `tokengate/core/provider.py`
- Create: `tests/test_proxy.py` (non-streaming cases)

- [ ] **Step 1: Write failing tests**

`tests/test_proxy.py`:
```python
import os
import pytest
from tokengate.core.config import Settings
from tokengate.core.normalize import normalize_openai, normalize_anthropic
from tokengate.core.provider import call_upstream, UpstreamError
from tokengate.core.mock_provider import MockTransport


@pytest.fixture
def settings(tmp_path):
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    yield s
    del os.environ["TOKENGATE_DATA_DIR"]


@pytest.fixture
def transport():
    return MockTransport()


@pytest.mark.asyncio
async def test_openai_passthrough(settings, transport):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    req = normalize_openai(body, {"authorization": "Bearer sk-test"})
    resp = await call_upstream(req, settings, transport=transport)
    assert resp.tokens_in == 10
    assert resp.tokens_out == 5
    assert resp.content == "Mock response"


@pytest.mark.asyncio
async def test_anthropic_passthrough(settings, transport):
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    req = normalize_anthropic(body, {"x-api-key": "sk-ant-test"})
    resp = await call_upstream(req, settings, transport=transport)
    assert resp.tokens_in == 10
    assert resp.tokens_out == 5
    assert resp.content == "Mock response"


@pytest.mark.asyncio
async def test_extra_fields_reach_upstream(settings):
    recorded = MockTransport()
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 0.9,
        "stop_sequences": ["END"],
    }
    req = normalize_openai(body, {})
    await call_upstream(req, settings, transport=recorded)
    sent_body = recorded.requests[0]
    assert sent_body["top_p"] == 0.9
    assert sent_body["stop_sequences"] == ["END"]


@pytest.mark.asyncio
async def test_upstream_error_raises(settings):
    transport = MockTransport(mode="error", error_status=500)
    req = normalize_openai({"model": "gpt-4o", "messages": []}, {})
    with pytest.raises(UpstreamError) as exc:
        await call_upstream(req, settings, transport=transport)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_upstream_429_raises(settings):
    transport = MockTransport(mode="error", error_status=429)
    req = normalize_openai({"model": "gpt-4o", "messages": []}, {})
    with pytest.raises(UpstreamError) as exc:
        await call_upstream(req, settings, transport=transport)
    assert exc.value.status_code == 429
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_proxy.py -v
```

- [ ] **Step 3: Create `tokengate/core/provider.py`**

```python
from __future__ import annotations
import json
from typing import AsyncIterator

import httpx

from tokengate.core.config import Settings
from tokengate.core.normalize import GatewayRequest, GatewayResponse, serialize_for_upstream
from tokengate.core.tokens import (
    extract_usage_openai, extract_usage_anthropic, parse_streaming_usage,
)


class UpstreamError(Exception):
    def __init__(self, status_code: int, body: dict | bytes):
        self.status_code = status_code
        self.body = body if isinstance(body, dict) else {}
        self.raw_body = body if isinstance(body, bytes) else json.dumps(body).encode()
        super().__init__(f"Upstream returned {status_code}")


def _upstream_url(req: GatewayRequest, settings: Settings) -> str:
    if req.route == "openai":
        return f"{settings.openai_base_url}/v1/chat/completions"
    return f"{settings.anthropic_base_url}/v1/messages"


def _auth_headers(req: GatewayRequest) -> dict[str, str]:
    auth = req.raw_headers.get("authorization") or req.raw_headers.get("x-api-key", "")
    if req.route == "openai":
        return {"Authorization": auth}
    key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else auth
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}


async def call_upstream(
    req: GatewayRequest,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GatewayResponse:
    body = {**serialize_for_upstream(req), "stream": False}
    url = _upstream_url(req, settings)
    headers = _auth_headers(req)
    kwargs: dict = {"timeout": 120.0}
    if transport:
        kwargs["transport"] = transport

    async with httpx.AsyncClient(**kwargs) as client:
        resp = await client.post(url, json=body, headers=headers)

    raw = resp.json()
    if resp.status_code != 200:
        raise UpstreamError(resp.status_code, raw)

    if req.route == "openai":
        tokens_in, tokens_out = extract_usage_openai(raw)
        content = raw["choices"][0]["message"]["content"]
        stop_reason = raw["choices"][0].get("finish_reason", "stop")
        model = raw.get("model", req.model)
    else:
        tokens_in, tokens_out = extract_usage_anthropic(raw)
        content = raw["content"][0]["text"] if raw.get("content") else ""
        stop_reason = raw.get("stop_reason", "end_turn")
        model = raw.get("model", req.model)

    return GatewayResponse(
        content=content,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        stop_reason=stop_reason,
        raw_body=raw,
    )


async def stream_upstream(
    req: GatewayRequest,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[bytes]:
    """
    Async generator. Yields raw SSE bytes to relay to client.
    After all bytes are yielded, yields one sentinel dict:
    {"_usage": True, "tokens_in": int, "tokens_out": int, "model": str}
    """
    body = {**serialize_for_upstream(req), "stream": True}
    url = _upstream_url(req, settings)
    headers = _auth_headers(req)
    kwargs: dict = {"timeout": 120.0}
    if transport:
        kwargs["transport"] = transport

    collected: list[bytes] = []

    async with httpx.AsyncClient(**kwargs) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                raise UpstreamError(resp.status_code, raw)

            async for chunk in resp.aiter_bytes(4096):
                collected.append(chunk)
                yield chunk

    full_text = b"".join(collected).decode(errors="replace")
    tokens_in, tokens_out = parse_streaming_usage(full_text, req.route)
    yield {"_usage": True, "tokens_in": tokens_in, "tokens_out": tokens_out, "model": req.model}
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_proxy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tokengate/core/provider.py tests/test_proxy.py
git commit -m "feat: upstream httpx provider — non-streaming + streaming with usage extraction"
```

---

## Task 9: FastAPI App — Auth, Startup Check, Layer Pipeline

**Files:**
- Create: `tokengate/proxy/server.py` (skeleton only — endpoints in Task 10)
- Create: `tests/test_security.py`

- [ ] **Step 1: Write failing security tests**

`tests/test_security.py`:
```python
import os
import pytest
from fastapi.testclient import TestClient
from tokengate.core.config import Settings
from tokengate.proxy.server import app, check_startup, _is_loopback
import tokengate.proxy.server as _server


@pytest.fixture(autouse=True)
def reset_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _server._settings = None  # force re-read
    yield
    _server._settings = None


def make_settings(tmp_path, bind="127.0.0.1", key=""):
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings.__new__(Settings)
    s.bind = bind
    s.tokengate_key = key
    s.data_dir = tmp_path
    s.db_path = tmp_path / "tokengate.db"
    s.pid_path = tmp_path / "tokengate.pid"
    s.log_path = tmp_path / "logs" / "tokengate.log"
    s.prices = {}
    s.openai_base_url = "https://api.openai.com"
    s.anthropic_base_url = "https://api.anthropic.com"
    return s


def test_check_startup_exits_non_loopback_no_key(tmp_path):
    s = make_settings(tmp_path, bind="0.0.0.0", key="")
    with pytest.raises(SystemExit) as exc:
        check_startup(s)
    assert exc.value.code == 1


def test_check_startup_allows_loopback_no_key(tmp_path):
    s = make_settings(tmp_path, bind="127.0.0.1", key="")
    check_startup(s)  # must not raise


def test_check_startup_allows_non_loopback_with_key(tmp_path):
    s = make_settings(tmp_path, bind="0.0.0.0", key="secret")
    check_startup(s)  # must not raise


def test_is_loopback():
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("localhost") is True
    assert _is_loopback("192.168.1.5") is False
    assert _is_loopback("10.0.0.1") is False


def test_non_loopback_no_key_gets_401(tmp_path, monkeypatch):
    s = make_settings(tmp_path, bind="0.0.0.0", key="secret-key")
    monkeypatch.setattr(_server, "get_settings", lambda: s)
    monkeypatch.setattr(_server, "_is_loopback", lambda host: False)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert resp.status_code == 401


def test_non_loopback_correct_key_passes_auth(tmp_path, monkeypatch):
    from tokengate.analytics.db import init_db
    init_db(tmp_path / "tokengate.db")
    from tokengate.core.mock_provider import MockTransport
    s = make_settings(tmp_path, bind="0.0.0.0", key="secret-key")
    monkeypatch.setattr(_server, "get_settings", lambda: s)
    monkeypatch.setattr(_server, "_is_loopback", lambda host: False)
    monkeypatch.setattr(_server, "_transport", MockTransport())

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-TokenGate-Key": "secret-key"},
    )
    # Passes auth — may succeed or fail at upstream but NOT 401
    assert resp.status_code != 401
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_security.py -v
```

- [ ] **Step 3: Create `tokengate/proxy/server.py` (skeleton — no endpoints yet)**

```python
from __future__ import annotations
import ipaddress
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tokengate.core.config import Settings
from tokengate.core.context import LayerContext
from tokengate.analytics.db import init_db, write_row
from tokengate.analytics.stats import get_stats
import tokengate.layers.exact_cache as _l_exact
import tokengate.layers.semantic_cache as _l_semantic
import tokengate.layers.distiller as _l_distiller
import tokengate.layers.compressor as _l_compressor
import tokengate.layers.router as _l_router
import tokengate.layers.budgeter as _l_budgeter


_PIPELINE = [_l_exact, _l_semantic, _l_distiller, _l_compressor, _l_router, _l_budgeter]

# Module-level transport override — set in tests via monkeypatch
_transport: httpx.AsyncBaseTransport | None = None
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "::1")


def check_startup(settings: Settings) -> None:
    """Raise SystemExit(1) if binding to non-loopback without a key."""
    if settings.bind != "127.0.0.1" and not settings.tokengate_key:
        print(
            "ERROR: TOKENGATE_KEY must be set when binding to a non-loopback address. "
            "Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    check_startup(s)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(s.db_path)
    yield


app = FastAPI(title="TokenGate", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.client and not _is_loopback(request.client.host):
        s = get_settings()
        key = request.headers.get("x-tokengate-key", "")
        if key != s.tokengate_key:
            write_row(
                s.db_path,
                ts=time.time(),
                route=str(request.url.path),
                status="auth_error",
            )
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


async def _run_pipeline(ctx: LayerContext) -> LayerContext:
    for layer in _PIPELINE:
        ctx = await layer.apply(ctx)
        if ctx.response is not None:
            break
    return ctx


@app.get("/stats")
async def stats_endpoint():
    s = get_settings()
    return get_stats(s.db_path)
```

- [ ] **Step 4: Run security tests**

```bash
pytest tests/test_security.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tokengate/proxy/server.py tests/test_security.py
git commit -m "feat: FastAPI skeleton — auth middleware, startup check, layer pipeline runner"
```

---

## Task 10: Non-Streaming Endpoints

**Files:**
- Modify: `tokengate/proxy/server.py` (add POST endpoints)
- Extend: `tests/test_proxy.py`

- [ ] **Step 1: Add endpoint tests to `tests/test_proxy.py`**

Append to the file:
```python
from fastapi.testclient import TestClient
from tokengate.analytics.db import init_db
from tokengate.proxy import server as _server
import tokengate.proxy.server as _sv


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Returns a configured TestClient with mock transport and temp data dir."""
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    _sv._settings = s
    init_db(s.db_path)
    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, transport, s
    _sv._settings = None
    _sv._transport = None


def test_openai_endpoint_returns_200(test_client):
    client, transport, _ = test_client
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Mock response"


def test_anthropic_endpoint_returns_200(test_client):
    client, transport, _ = test_client
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "Mock response"


def test_response_headers_present(test_client):
    client, _, _ = test_client
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "x-tokengate-cache" in resp.headers
    assert "x-tokengate-model" in resp.headers
    assert "x-tokengate-saved-tokens" in resp.headers
    assert resp.headers["x-tokengate-cache"] == "none"
    assert resp.headers["x-tokengate-saved-tokens"] == "0"


def test_analytics_row_written_on_success(test_client):
    import sqlite3
    client, _, s = test_client
    client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests WHERE status='ok'").fetchone())
    con.close()
    assert row["tokens_in_raw"] == 10
    assert row["tokens_out"] == 5
    assert row["model_used"] == "mock"
    assert row["route"] == "openai"


def test_analytics_row_on_upstream_error(test_client, monkeypatch):
    import sqlite3
    from tokengate.core.mock_provider import MockTransport as MT
    client, _, s = test_client
    monkeypatch.setattr(_sv, "_transport", MT(mode="error", error_status=500))
    resp = client.post("/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": []},
                       headers={"accept": "application/json"})
    assert resp.status_code == 502
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests WHERE status='upstream_error'").fetchone())
    con.close()
    assert row["status"] == "upstream_error"
    assert row["error_detail"] is not None
```

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_proxy.py::test_openai_endpoint_returns_200 -v
```

- [ ] **Step 3: Add endpoints to `tokengate/proxy/server.py`**

Add these imports at the top of the existing file:
```python
from fastapi.responses import Response
from tokengate.core.normalize import normalize_openai, normalize_anthropic, serialize_for_upstream
from tokengate.core.provider import call_upstream, UpstreamError
from tokengate.core.tokens import compute_cost
```

Add these two endpoint functions after the `/stats` route:
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    s = get_settings()
    start_ts = time.time()
    body = await request.json()
    req = normalize_openai(body, dict(request.headers))
    ctx = LayerContext(request=req)
    ctx = await _run_pipeline(ctx)
    return await _handle_request(req, ctx, s, start_ts)


@app.post("/v1/messages")
async def messages(request: Request):
    s = get_settings()
    start_ts = time.time()
    body = await request.json()
    req = normalize_anthropic(body, dict(request.headers))
    ctx = LayerContext(request=req)
    ctx = await _run_pipeline(ctx)
    return await _handle_request(req, ctx, s, start_ts)


async def _handle_request(req, ctx: LayerContext, s: Settings, start_ts: float):
    if req.stream:
        return await _streaming_response(req, ctx, s, start_ts)
    return await _non_streaming_response(req, ctx, s, start_ts)


async def _non_streaming_response(req, ctx: LayerContext, s: Settings, start_ts: float):
    status = "ok"
    error_detail = None
    resp_body: dict = {}
    tokens_in = tokens_out = 0
    model = req.model

    try:
        upstream_resp = await call_upstream(req, s, transport=_transport)
        tokens_in = upstream_resp.tokens_in
        tokens_out = upstream_resp.tokens_out
        model = upstream_resp.model
        resp_body = upstream_resp.raw_body
    except UpstreamError as e:
        status = "upstream_error"
        error_detail = str(e)
        resp_body = e.body

    cost = compute_cost(model, tokens_in, tokens_out, s)
    latency_ms = int((time.time() - start_ts) * 1000)
    write_row(
        s.db_path,
        ts=start_ts, route=req.route, status=status, error_detail=error_detail,
        layers_applied=[asdict(d) for d in ctx.decisions],
        tokens_in_raw=tokens_in or None, tokens_in_final=tokens_in or None,
        tokens_out=tokens_out or None, model_used=model,
        latency_ms=latency_ms, est_cost_usd=cost,
    )

    tg_headers = {
        "x-tokengate-cache": "none",
        "x-tokengate-model": model,
        "x-tokengate-saved-tokens": "0",
    }
    http_status = 200 if status == "ok" else 502
    return JSONResponse(content=resp_body, status_code=http_status, headers=tg_headers)
```

- [ ] **Step 4: Run all proxy tests**

```bash
pytest tests/test_proxy.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tokengate/proxy/server.py tests/test_proxy.py
git commit -m "feat: non-streaming endpoints for both OpenAI and Anthropic shapes"
```

---

## Task 11: Streaming Endpoints

**Files:**
- Modify: `tokengate/proxy/server.py` (add `_streaming_response`)
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Write failing streaming tests**

`tests/test_streaming.py`:
```python
import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from tokengate.analytics.db import init_db
from tokengate.core.config import Settings
from tokengate.core.mock_provider import MockTransport
import tokengate.proxy.server as _sv


@pytest.fixture
def streaming_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    _sv._settings = s
    init_db(s.db_path)
    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, s
    _sv._settings = None
    _sv._transport = None


def test_openai_streaming_yields_sse(streaming_client):
    client, _ = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.read().decode()
    assert "[DONE]" in body


def test_anthropic_streaming_yields_sse(streaming_client):
    client, _ = streaming_client
    with client.stream(
        "POST", "/v1/messages",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 100, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode()
    assert "message_stop" in body


def test_streaming_x_tokengate_headers_before_body(streaming_client):
    client, _ = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        # Headers are received before reading any body
        assert resp.headers.get("x-tokengate-cache") == "none"
        assert resp.headers.get("x-tokengate-saved-tokens") == "0"
        assert "x-tokengate-model" in resp.headers


def test_streaming_analytics_written_after_stream(streaming_client):
    client, s = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        resp.read()  # consume full stream

    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests").fetchone())
    con.close()
    assert row["tokens_in_raw"] == 10
    assert row["tokens_out"] == 5
    assert row["status"] == "ok"


def test_streaming_token_counts_from_final_usage_event(streaming_client):
    """Token counts must come from the SSE usage event, not be estimated."""
    client, s = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        resp.read()

    con = sqlite3.connect(s.db_path)
    row = con.execute("SELECT tokens_in_raw, tokens_out FROM requests").fetchone()
    con.close()
    # MockTransport reports exactly 10 in / 5 out in the usage event
    assert row[0] == 10
    assert row[1] == 5
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_streaming.py -v
```

- [ ] **Step 3: Add `_streaming_response` to `tokengate/proxy/server.py`**

Add these imports at the top:
```python
import json as _json
from fastapi.responses import StreamingResponse
from tokengate.core.provider import stream_upstream
```

Add the function after `_non_streaming_response`:
```python
async def _streaming_response(req, ctx: LayerContext, s: Settings, start_ts: float):
    tg_headers = {
        "x-tokengate-cache": "none",
        "x-tokengate-model": req.model,
        "x-tokengate-saved-tokens": "0",
        "cache-control": "no-cache",
    }

    async def generate():
        status = "ok"
        error_detail = None
        tokens_in = tokens_out = 0
        model = req.model

        try:
            async for chunk in stream_upstream(req, s, transport=_transport):
                if isinstance(chunk, dict) and chunk.get("_usage"):
                    tokens_in = chunk["tokens_in"]
                    tokens_out = chunk["tokens_out"]
                    model = chunk.get("model", model)
                else:
                    yield chunk
        except UpstreamError as e:
            status = "upstream_error"
            error_detail = str(e)
            yield _json.dumps(e.body).encode()
        except Exception as e:
            status = "upstream_error"
            error_detail = str(e)
        finally:
            cost = compute_cost(model, tokens_in, tokens_out, s)
            latency_ms = int((time.time() - start_ts) * 1000)
            write_row(
                s.db_path,
                ts=start_ts, route=req.route, status=status, error_detail=error_detail,
                layers_applied=[asdict(d) for d in ctx.decisions],
                tokens_in_raw=tokens_in or None, tokens_in_final=tokens_in or None,
                tokens_out=tokens_out or None, model_used=model,
                latency_ms=latency_ms, est_cost_usd=cost,
            )

    return StreamingResponse(generate(), media_type="text/event-stream", headers=tg_headers)
```

- [ ] **Step 4: Run streaming tests**

```bash
pytest tests/test_streaming.py -v
```

Expected: all passed.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tokengate/proxy/server.py tests/test_streaming.py
git commit -m "feat: streaming SSE endpoints with post-stream analytics logging"
```

---

## Task 12: Dashboard HTML

**Files:**
- Create: `tokengate/analytics/dashboard.html`
- Modify: `tokengate/proxy/server.py` (add `/dashboard` route)

- [ ] **Step 1: Add `/dashboard` route to `tokengate/proxy/server.py`**

Add import at top:
```python
from pathlib import Path as _Path
from fastapi.responses import FileResponse
```

Add route after `/stats`:
```python
@app.get("/dashboard")
async def dashboard():
    html_path = _Path(__file__).parent.parent / "analytics" / "dashboard.html"
    return FileResponse(html_path, media_type="text/html")
```

- [ ] **Step 2: Create `tokengate/analytics/dashboard.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TokenGate Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:24px}
  h1{font-size:1.5rem;font-weight:700;margin-bottom:24px;color:#f1f5f9}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}
  .card{background:#1e2330;border-radius:12px;padding:20px}
  .card .label{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
  .card .value{font-size:1.75rem;font-weight:700;color:#f1f5f9}
  .card .value.unknown{color:#64748b;font-size:1rem}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:32px}
  @media(max-width:768px){.charts{grid-template-columns:1fr}}
  .chart-box{background:#1e2330;border-radius:12px;padding:20px}
  .chart-box h2{font-size:.9rem;color:#94a3b8;margin-bottom:16px}
  table{width:100%;border-collapse:collapse;background:#1e2330;border-radius:12px;overflow:hidden}
  th{text-align:left;padding:12px 16px;font-size:.75rem;color:#64748b;text-transform:uppercase;border-bottom:1px solid #2d3748}
  td{padding:10px 16px;font-size:.85rem;border-bottom:1px solid #1a202c}
  tr:last-child td{border-bottom:none}
  .badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:.7rem;font-weight:600}
  .badge-ok{background:#14532d;color:#4ade80}
  .badge-error{background:#7f1d1d;color:#f87171}
  .badge-auth{background:#78350f;color:#fbbf24}
  #refresh-info{font-size:.75rem;color:#475569;margin-bottom:16px}
</style>
</head>
<body>
<h1>TokenGate Dashboard</h1>
<p id="refresh-info">Refreshing every 10 seconds</p>

<div class="cards">
  <div class="card"><div class="label">Requests</div><div class="value" id="total-requests">—</div></div>
  <div class="card"><div class="label">Tokens In</div><div class="value" id="total-tokens-in">—</div></div>
  <div class="card"><div class="label">Tokens Out</div><div class="value" id="total-tokens-out">—</div></div>
  <div class="card"><div class="label">Est. Cost</div><div class="value" id="total-cost">—</div></div>
  <div class="card"><div class="label">Est. Saved</div><div class="value" id="total-saved">—</div></div>
  <div class="card"><div class="label">Cache Hit Rate</div><div class="value" id="cache-rate">—</div></div>
</div>

<div class="charts">
  <div class="chart-box"><h2>Daily Requests</h2><canvas id="req-chart" height="180"></canvas></div>
  <div class="chart-box"><h2>Daily Cost (USD)</h2><canvas id="cost-chart" height="180"></canvas></div>
</div>

<table>
  <thead><tr>
    <th>Time</th><th>Route</th><th>Status</th><th>Model</th>
    <th>Tokens In</th><th>Tokens Out</th><th>Latency</th><th>Cost</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>

<script>
const fmt = (n, d=2) => n == null ? '<span class="unknown">unknown</span>' : Number(n).toFixed(d)
const fmtInt = n => n == null ? '—' : Number(n).toLocaleString()
const fmtDate = ts => new Date(ts * 1000).toLocaleTimeString()

let reqChart, costChart

function initCharts(labels, reqData, costData) {
  const opts = {responsive:true, plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b'}},y:{ticks:{color:'#64748b'},grid:{color:'#2d3748'}}}}
  reqChart = new Chart(document.getElementById('req-chart'), {
    type:'line', data:{labels, datasets:[{data:reqData, borderColor:'#6366f1', tension:.3, fill:false}]}, options:opts
  })
  costChart = new Chart(document.getElementById('cost-chart'), {
    type:'bar', data:{labels, datasets:[{data:costData, backgroundColor:'#0ea5e9'}]}, options:opts
  })
}

function updateCharts(daily) {
  const labels = daily.map(d => d.date).reverse()
  const reqData = daily.map(d => d.requests).reverse()
  const costData = daily.map(d => d.est_cost_usd || 0).reverse()
  if (!reqChart) { initCharts(labels, reqData, costData); return }
  reqChart.data.labels = labels; reqChart.data.datasets[0].data = reqData; reqChart.update()
  costChart.data.labels = labels; costChart.data.datasets[0].data = costData; costChart.update()
}

function badgeStatus(s) {
  const cls = s === 'ok' ? 'badge-ok' : s === 'auth_error' ? 'badge-auth' : 'badge-error'
  return `<span class="badge ${cls}">${s}</span>`
}

async function load() {
  try {
    const [stats, recent] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/stats/recent').then(r => r.json()).catch(() => []),
    ])
    document.getElementById('total-requests').textContent = fmtInt(stats.total_requests)
    document.getElementById('total-tokens-in').textContent = fmtInt(stats.total_tokens_in)
    document.getElementById('total-tokens-out').textContent = fmtInt(stats.total_tokens_out)
    const costEl = document.getElementById('total-cost')
    costEl.innerHTML = stats.total_est_cost_usd == null
      ? '<span class="unknown">unknown</span>'
      : `$${fmt(stats.total_est_cost_usd, 4)}`
    document.getElementById('total-saved').textContent = `$${fmt(stats.total_est_saved_usd, 4)}`
    document.getElementById('cache-rate').textContent = `${(stats.cache_hit_rate * 100).toFixed(1)}%`
    updateCharts(stats.daily)
  } catch(e) { console.error(e) }
}

load()
setInterval(load, 10000)
</script>
</body>
</html>
```

- [ ] **Step 3: Verify dashboard route**

```bash
python -c "
import httpx, asyncio
from fastapi.testclient import TestClient
import os, tempfile
with tempfile.TemporaryDirectory() as d:
    os.environ['TOKENGATE_DATA_DIR'] = d
    import tokengate.proxy.server as s
    s._settings = None
    from tokengate.analytics.db import init_db
    from tokengate.core.config import Settings
    s._settings = Settings()
    init_db(s._settings.db_path)
    client = TestClient(s.app)
    resp = client.get('/dashboard')
    print(resp.status_code, 'html' in resp.headers.get('content-type',''))
"
```

Expected: `200 True`

- [ ] **Step 4: Commit**

```bash
git add tokengate/analytics/dashboard.html tokengate/proxy/server.py
git commit -m "feat: /dashboard static page with Chart.js, /stats endpoint"
```

---

## Task 13: PID File Management and Daemon

**Files:**
- Create: `tokengate/cli/daemon.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

`tests/test_cli.py`:
```python
import json
import os
import signal
import socket
import sys
import time
import pytest
from pathlib import Path
from tokengate.cli.daemon import (
    read_pid_file, write_pid_file, remove_pid_file,
    is_port_free, check_port_or_exit, _pid_alive,
)


@pytest.fixture
def pid_path(tmp_path):
    return tmp_path / "tokengate.pid"


def test_write_and_read_pid_file(pid_path):
    write_pid_file(pid_path, pid=12345, port=8787)
    data = read_pid_file(pid_path)
    assert data["pid"] == 12345
    assert data["port"] == 8787
    assert "started_at" in data


def test_read_missing_pid_file_returns_none(tmp_path):
    assert read_pid_file(tmp_path / "missing.pid") is None


def test_remove_pid_file(pid_path):
    write_pid_file(pid_path, pid=1, port=8787)
    remove_pid_file(pid_path)
    assert not pid_path.exists()


def test_remove_missing_pid_file_is_noop(tmp_path):
    remove_pid_file(tmp_path / "missing.pid")  # must not raise


def test_pid_alive_current_process():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_dead_process():
    # PID 1 is always alive on Unix, so use an absurdly high PID
    assert _pid_alive(999999999) is False


def test_is_port_free_on_unused_port():
    # Find a free port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert is_port_free("127.0.0.1", port) is True


def test_is_port_free_on_occupied_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    assert is_port_free("127.0.0.1", port) is False
    s.close()


def test_check_port_busy_exits(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    pid_path = tmp_path / "t.pid"
    with pytest.raises(SystemExit) as exc:
        check_port_or_exit("127.0.0.1", port, pid_path)
    assert exc.value.code == 1
    s.close()


def test_stale_pid_file_detected(pid_path):
    write_pid_file(pid_path, pid=999999999, port=8787)
    # _pid_alive(999999999) is False, so stale
    data = read_pid_file(pid_path)
    assert data is not None
    assert not _pid_alive(data["pid"])
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Create `tokengate/cli/daemon.py`**

```python
from __future__ import annotations
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def write_pid_file(pid_path: Path, *, pid: int, port: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"pid": pid, "port": port, "started_at": time.time()}
    pid_path.write_text(json.dumps(data))


def read_pid_file(pid_path: Path) -> dict | None:
    if not pid_path.exists():
        return None
    try:
        return json.loads(pid_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def remove_pid_file(pid_path: Path) -> None:
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if _HAS_PSUTIL:
        return _psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _pid_is_tokengate(pid: int) -> bool:
    """Best-effort check that a PID belongs to a tokengate process."""
    if not _HAS_PSUTIL:
        return True  # Can't verify, assume OK
    try:
        p = _psutil.Process(pid)
        cmdline = " ".join(p.cmdline())
        return "tokengate" in cmdline or "uvicorn" in cmdline
    except (_psutil.NoSuchProcess, _psutil.AccessDenied):
        return False


def is_port_free(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def check_port_or_exit(host: str, port: int, pid_path: Path) -> None:
    if not is_port_free(host, port):
        print(
            f"Port {port} in use — is TokenGate already running? Try `rait status`.",
            file=sys.stderr,
        )
        sys.exit(1)


def start_foreground(host: str, port: int, pid_path: Path, log_level: str = "info") -> None:
    import atexit
    write_pid_file(pid_path, pid=os.getpid(), port=port)
    atexit.register(remove_pid_file, pid_path)

    if sys.platform != "win32":
        def _sigint(sig, frame):
            remove_pid_file(pid_path)
            sys.exit(0)
        signal.signal(signal.SIGINT, _sigint)

    import uvicorn
    uvicorn.run(
        "tokengate.proxy.server:app",
        host=host, port=port, log_level=log_level,
    )


def start_detached(host: str, port: int, pid_path: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "tokengate.proxy.server:app",
        "--host", host, "--port", str(port),
    ]

    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file,
            creationflags=flags, close_fds=True,
        )
    else:
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file,
            start_new_session=True, close_fds=True,
        )

    # Give the process a moment to start, then write PID
    time.sleep(0.5)
    if proc.poll() is not None:
        print(f"Failed to start TokenGate. Check logs: {log_path}", file=sys.stderr)
        sys.exit(1)

    write_pid_file(pid_path, pid=proc.pid, port=port)
    print(f"TokenGate started (PID {proc.pid}) on port {port}. Logs: {log_path}")


def stop_daemon(pid_path: Path) -> None:
    data = read_pid_file(pid_path)
    if data is None:
        print("TokenGate is not running.")
        return

    pid = data["pid"]
    if not _pid_alive(pid):
        print("Not running (stale PID file removed).")
        remove_pid_file(pid_path)
        return

    if not _pid_is_tokengate(pid):
        print(f"WARNING: PID {pid} does not appear to be a TokenGate process. Aborting stop.")
        return

    try:
        if sys.platform == "win32":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGTERM)
        remove_pid_file(pid_path)
        print(f"TokenGate stopped (PID {pid}).")
    except OSError as e:
        print(f"Failed to stop process {pid}: {e}", file=sys.stderr)


def status_daemon(pid_path: Path) -> None:
    data = read_pid_file(pid_path)
    if data is None:
        print("TokenGate is not running.")
        return

    pid = data["pid"]
    if not _pid_alive(pid):
        print("Not running (stale PID file removed).")
        remove_pid_file(pid_path)
        return

    uptime_s = int(time.time() - data["started_at"])
    h, rem = divmod(uptime_s, 3600)
    m, s = divmod(rem, 60)
    uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    print(f"TokenGate is running.")
    print(f"  PID   : {pid}")
    print(f"  Port  : {data['port']}")
    print(f"  Uptime: {uptime_str}")
    print(f"  Dashboard: http://127.0.0.1:{data['port']}/dashboard")
```

- [ ] **Step 4: Run CLI tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tokengate/cli/daemon.py tests/test_cli.py
git commit -m "feat: cross-platform daemon management — PID file, port check, start/stop/status"
```

---

## Task 14: Install Wizard

**Files:**
- Create: `tokengate/cli/wizard.py`

- [ ] **Step 1: Create `tokengate/cli/wizard.py`**

```python
from __future__ import annotations
import os
import stat
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm

from tokengate.analytics.db import init_db
from tokengate.core.config import Settings

console = Console()

_BANNER = """
[bold cyan]╔════════════════════════════════════╗
║       TokenGate  v0.1.0            ║
║  Intelligent Token-Saving Gateway  ║
╚════════════════════════════════════╝[/bold cyan]
"""

_SNIPPET_ANTHROPIC = """
[bold green]Integration (one-line change):[/bold green]
  [cyan]client = Anthropic(base_url="http://localhost:{port}")[/cyan]
"""

_SNIPPET_OPENAI = """
[bold green]Integration (one-line change):[/bold green]
  [cyan]client = OpenAI(base_url="http://localhost:{port}/v1")[/cyan]
"""


def run_wizard(
    provider: str | None = None,
    port: int | None = None,
    yes: bool = False,
) -> None:
    console.print(_BANNER)

    # ── 1. Provider ──────────────────────────────────────────────────────────
    if provider is None:
        provider = Prompt.ask(
            "Provider",
            choices=["anthropic", "openai", "both"],
            default="anthropic",
        )

    # ── 2. API key ────────────────────────────────────────────────────────────
    data_dir = Path("~/.rait").expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = data_dir / ".env"

    key_var = "ANTHROPIC_API_KEY" if provider in ("anthropic", "both") else "OPENAI_API_KEY"
    existing = os.environ.get(key_var, "")

    if existing and yes:
        api_key = existing
    else:
        api_key = Prompt.ask(f"{key_var} (hidden)", password=True)

    _write_env(env_path, key_var, api_key)
    console.print(
        f"[green]✓[/green] API key written to [bold]{env_path}[/bold] (owner-only permissions)\n"
        "[yellow]Note:[/yellow] Key is stored but [bold]not validated[/bold]. "
        "Run [cyan]rait test --live[/cyan] (coming in a later release) to verify."
    )

    # ── 3. Port ───────────────────────────────────────────────────────────────
    if port is None:
        if yes:
            port = 8787
        else:
            raw = Prompt.ask("Gateway port", default="8787")
            port = int(raw)

    # ── 4. Write config ────────────────────────────────────────────────────────
    cfg_path = data_dir / "tokengate.yaml"
    _write_config(cfg_path, port)
    console.print(f"[green]✓[/green] Config written to [bold]{cfg_path}[/bold]")

    # ── 5. Init DB ─────────────────────────────────────────────────────────────
    os.environ["TOKENGATE_DATA_DIR"] = str(data_dir)
    s = Settings()
    init_db(s.db_path)
    console.print(f"[green]✓[/green] Database initialised at [bold]{s.db_path}[/bold]")

    # ── 6. Integration snippet ────────────────────────────────────────────────
    snippet = _SNIPPET_ANTHROPIC if provider in ("anthropic", "both") else _SNIPPET_OPENAI
    console.print(snippet.format(port=port))

    # ── 7. Offer to start ─────────────────────────────────────────────────────
    if yes or Confirm.ask("Start TokenGate now (detached)?", default=True):
        from tokengate.cli.daemon import check_port_or_exit, start_detached
        check_port_or_exit("127.0.0.1", port, s.pid_path)
        start_detached("127.0.0.1", port, s.pid_path, s.log_path)


def _write_env(env_path: Path, key_var: str, value: str) -> None:
    lines = []
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line.startswith(f"{key_var}="):
                lines.append(line)
    lines.append(f"{key_var}={value}")
    env_path.write_text("\n".join(lines) + "\n")
    if sys.platform != "win32":
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    else:
        _win_set_owner_only(env_path)


def _write_config(cfg_path: Path, port: int) -> None:
    import shutil, tokengate
    default_cfg = Path(tokengate.__file__).parent.parent / "tokengate.yaml"
    if default_cfg.exists():
        shutil.copy(default_cfg, cfg_path)
    # Patch the port
    text = cfg_path.read_text() if cfg_path.exists() else "bind: '127.0.0.1'\n"
    lines = [l for l in text.splitlines() if not l.startswith("port:")]
    lines.insert(0, f"port: {port}")
    cfg_path.write_text("\n".join(lines) + "\n")


def _win_set_owner_only(path: Path) -> None:
    try:
        import subprocess
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{os.getlogin()}:(R,W)"],
            check=True, capture_output=True,
        )
    except Exception:
        pass  # Best-effort on Windows
```

- [ ] **Step 2: Test install with --yes flag**

```bash
python -c "
import os, tempfile
with tempfile.TemporaryDirectory() as d:
    os.environ['TOKENGATE_DATA_DIR'] = d
    os.environ['ANTHROPIC_API_KEY'] = 'sk-test-key'
    from tokengate.cli.wizard import run_wizard
    run_wizard(provider='anthropic', port=9999, yes=True)
    from pathlib import Path
    print('env exists:', (Path(d) / '.env').exists())
    print('db exists:', (Path(d) / 'tokengate.db').exists())
"
```

Expected: `env exists: True`, `db exists: True`

- [ ] **Step 3: Commit**

```bash
git add tokengate/cli/wizard.py
git commit -m "feat: rait install wizard — provider setup, API key storage, config write"
```

---

## Task 15: CLI Main Entry Point

**Files:**
- Create: `tokengate/cli/main.py`
- Extend: `tests/test_cli.py`

- [ ] **Step 1: Add CLI integration tests to `tests/test_cli.py`**

Append to the file:
```python
from typer.testing import CliRunner
from tokengate.cli.main import app as cli_app


cli_runner = CliRunner()


def test_rait_help():
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "install" in result.output
    assert "start" in result.output
    assert "stop" in result.output
    assert "status" in result.output


def test_rait_status_no_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    result = cli_runner.invoke(cli_app, ["status"])
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_rait_stop_no_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    result = cli_runner.invoke(cli_app, ["stop"])
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_rait_status_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    from tokengate.cli.daemon import write_pid_file
    write_pid_file(tmp_path / "tokengate.pid", pid=999999999, port=8787)
    result = cli_runner.invoke(cli_app, ["status"])
    assert result.exit_code == 0
    assert "stale" in result.output.lower() or "not running" in result.output.lower()
    assert not (tmp_path / "tokengate.pid").exists()
```

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_cli.py::test_rait_help -v
```

- [ ] **Step 3: Create `tokengate/cli/main.py`**

```python
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="rait — TokenGate CLI", add_completion=False)
console = Console()


def _get_settings():
    from tokengate.core.config import Settings
    return Settings()


@app.command()
def install(
    provider: Optional[str] = typer.Option(None, help="anthropic | openai | both"),
    port: Optional[int] = typer.Option(None, help="Gateway port (default 8787)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept all defaults non-interactively"),
):
    """Interactive setup wizard."""
    from tokengate.cli.wizard import run_wizard
    run_wizard(provider=provider, port=port, yes=yes)


@app.command()
def start(
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in background"),
):
    """Start the TokenGate gateway."""
    from tokengate.cli.daemon import check_port_or_exit, start_foreground, start_detached
    s = _get_settings()
    check_port_or_exit(s.bind, s.port, s.pid_path)
    if detach:
        start_detached(s.bind, s.port, s.pid_path, s.log_path)
    else:
        console.print(f"Starting TokenGate on [cyan]{s.bind}:{s.port}[/cyan] (foreground — Ctrl+C to stop)")
        start_foreground(s.bind, s.port, s.pid_path, s.log_level)


@app.command()
def stop():
    """Stop the TokenGate gateway."""
    from tokengate.cli.daemon import stop_daemon
    s = _get_settings()
    stop_daemon(s.pid_path)


@app.command()
def status():
    """Show gateway status."""
    from tokengate.cli.daemon import status_daemon
    s = _get_settings()
    status_daemon(s.pid_path)


@app.command()
def stats():
    """Print token savings summary."""
    from tokengate.analytics.stats import get_stats
    import json
    s = _get_settings()
    if not s.db_path.exists():
        console.print("[yellow]No analytics data yet. Start the gateway and make some requests.[/yellow]")
        raise typer.Exit(0)
    data = get_stats(s.db_path)
    console.print_json(json.dumps(data))


@app.command("cache")
def cache_cmd(
    clear: bool = typer.Option(False, "--clear", help="Clear all caches"),
):
    """Manage the cache (stub — implemented in Phase 2)."""
    console.print("[yellow]Cache management is available from Phase 2 onwards.[/yellow]")


@app.command("test")
def test_cmd():
    """Send a sample request through the gateway."""
    s = _get_settings()
    console.print(f"[cyan]Sending test request to http://{s.bind}:{s.port} ...[/cyan]")
    try:
        import httpx
        resp = httpx.post(
            f"http://{s.bind}:{s.port}/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Say hello."}]},
            timeout=10,
        )
        console.print(f"Status: {resp.status_code}")
        console.print(f"x-tokengate-cache: {resp.headers.get('x-tokengate-cache', 'n/a')}")
        console.print(f"x-tokengate-model: {resp.headers.get('x-tokengate-model', 'n/a')}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run all CLI tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all passed.

- [ ] **Step 5: Run the complete test suite**

```bash
pytest -v
```

Expected: all tests pass, no warnings about missing fixtures.

- [ ] **Step 6: Smoke-test the CLI entry point**

```bash
rait --help
```

Expected: shows `install`, `start`, `stop`, `status`, `stats`, `cache`, `test` commands.

- [ ] **Step 7: Commit**

```bash
git add tokengate/cli/main.py tests/test_cli.py
git commit -m "feat: rait CLI — install, start, stop, status, stats, test commands"
```

---

## Task 16: Final Integration Check

**Files:** No new files — verification only.

- [ ] **Step 1: Run full test suite**

```bash
pytest -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 2: Verify package installs cleanly**

```bash
pip install -e . --quiet && rait --help
```

Expected: `rait` command available with all subcommands listed.

- [ ] **Step 3: Smoke-test end-to-end with mock**

```bash
python -c "
import os, tempfile, asyncio
with tempfile.TemporaryDirectory() as d:
    os.environ['TOKENGATE_DATA_DIR'] = d
    import tokengate.proxy.server as srv
    from tokengate.analytics.db import init_db
    from tokengate.core.config import Settings
    from tokengate.core.mock_provider import MockTransport
    srv._settings = Settings()
    init_db(srv._settings.db_path)
    srv._transport = MockTransport()
    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    r = client.post('/v1/chat/completions',
        json={'model':'gpt-4o','messages':[{'role':'user','content':'hi'}]})
    print('Status:', r.status_code)
    print('Cache:', r.headers.get('x-tokengate-cache'))
    print('Tokens header:', r.headers.get('x-tokengate-saved-tokens'))
    stats = client.get('/stats').json()
    print('Logged rows:', stats['total_requests'])
"
```

Expected:
```
Status: 200
Cache: none
Tokens header: 0
Logged rows: 1
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: Phase 1 complete — passthrough gateway, analytics, dashboard, rait CLI"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] FastAPI proxy — Tasks 9, 10, 11
- [x] Both API shapes (OpenAI + Anthropic) — Task 3, 10
- [x] Streaming SSE pass-through with pre-stream headers — Task 11
- [x] Real token counts from final usage event — Tasks 7, 11
- [x] Full request logging (one row per client request, errors included) — Tasks 6, 10, 11
- [x] `status`, `error_detail` columns — Task 6
- [x] WAL mode + index on `ts` — Task 6
- [x] Price table with YAML override — Task 4
- [x] Unknown model → `est_cost_usd = NULL` — Tasks 4, 6
- [x] `extra: dict` round-trips to upstream — Tasks 3, 8
- [x] `/stats` JSON endpoint — Task 7
- [x] `/dashboard` static HTML — Task 12
- [x] Auth: loopback bypass, X-TokenGate-Key check, 401 — Task 9
- [x] Startup refuses non-loopback without key — Task 9
- [x] `rait install` wizard — Task 14
- [x] `rait start` (foreground + detach), PID file — Task 13
- [x] PID file as JSON `{pid, port, started_at}` — Task 13
- [x] Stale PID handling — Tasks 13, 15
- [x] Port-free check before spawn — Task 13
- [x] `rait stop`, `rait status` — Tasks 13, 15
- [x] All six layer stubs wired via `LayerContext` — Tasks 1, 2, 9
- [x] Mock provider — normal, streaming, error (500/429) — Task 5
- [x] Security tests — Task 9
- [x] Streaming usage test — Task 11
- [x] Extra fields test — Task 8
- [x] Wizard notes key is not validated — Task 14
- [x] Cross-platform daemon (Windows + Unix) — Task 13
