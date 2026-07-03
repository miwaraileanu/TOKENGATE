# TokenGate Phase 4 — Cascade Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the cascade router layer that scores request difficulty, routes cheap requests to a fast model, runs a self-confidence check, and escalates to the strong model only when needed.

**Architecture:** `router.py` becomes the terminal pipeline layer for non-streaming requests — it owns the upstream call and sets `ctx.response`. Pipeline order changes to `distiller → compressor → budgeter → router` so budgeter's `max_tokens` injection is in place before the router makes any upstream call. Streaming requests skip the router (logged decision, server fallback handles the call).

**Tech Stack:** Python asyncio, httpx, SQLite (existing), `dataclasses`, `re`, `copy`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tokengate/core/config.py` | Modify | Add router settings block |
| `tokengate/proxy/server.py` | Modify | Reorder pipeline; read router decision for cost/escalated |
| `tokengate/layers/router.py` | Replace stub | Full cascade router |
| `tests/test_router.py` | Create | Unit tests for router |
| `tests/test_proxy.py` | Modify | 4 integration tests |
| `scripts/retrain_router.py` | Create | Offline retraining script |

---

### Task 1: Router Settings in Config

**Files:**
- Modify: `tokengate/core/config.py:71-88`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_router.py`:

```python
"""Unit tests for Phase 4 cascade router."""
from __future__ import annotations
import os
import pytest
from tokengate.core.config import Settings
from tokengate.analytics.db import init_db


def make_settings(tmp_path) -> Settings:
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    init_db(s.db_path)
    return s


def test_router_settings_defaults(tmp_path):
    s = make_settings(tmp_path)
    assert s.router_enabled is True
    assert s.router_difficulty_threshold == 0.4
    assert s.router_escalation_enabled is True
    assert s.router_escalation_threshold == 3
    assert s.router_tools_tier == "strong"
    assert s.router_cheap_model == {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini"}
    assert s.router_strong_model == {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o"}
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_router.py::test_router_settings_defaults -v
```

Expected: `AttributeError: 'Settings' object has no attribute 'router_enabled'`

- [ ] **Step 3: Add router settings to `Settings.__init__`**

Append to the end of `tokengate/core/config.py` (after the `budget_table` block, before the closing of `__init__`):

```python
        _r = raw.get("router", {})
        self.router_enabled: bool = bool(_r.get("enabled", True))
        self.router_difficulty_threshold: float = float(_r.get("difficulty_threshold", 0.4))
        self.router_escalation_enabled: bool = bool(_r.get("escalation_enabled", True))
        self.router_escalation_threshold: int = int(_r.get("escalation_threshold", 3))
        self.router_tools_tier: str = _r.get("tools_tier", "strong")
        _cm = _r.get("cheap_model", {})
        self.router_cheap_model: dict[str, str] = {
            "anthropic": _cm.get("anthropic", "claude-haiku-4-5"),
            "openai": _cm.get("openai", "gpt-4o-mini"),
        }
        _sm = _r.get("strong_model", {})
        self.router_strong_model: dict[str, str] = {
            "anthropic": _sm.get("anthropic", "claude-sonnet-4-6"),
            "openai": _sm.get("openai", "gpt-4o"),
        }
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_router.py::test_router_settings_defaults -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tokengate/core/config.py tests/test_router.py
git commit -m "feat: add router settings to Settings (Phase 4)"
```

---

### Task 2: Pipeline Reorder + Server Cost Integration

**Files:**
- Modify: `tokengate/proxy/server.py:29` (pipeline list)
- Modify: `tokengate/proxy/server.py:160-219` (`_non_streaming_response`)

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_proxy.py` (at the end, under a new `# ── Phase 4 integration tests` comment):

```python
# ── Phase 4 integration tests ─────────────────────────────────────────────────

import tokengate.layers.router as _l_router


def test_pipeline_order_budgeter_before_router(tmp_path, monkeypatch):
    """Budgeter's max_tokens injection reaches the upstream call made by the router."""
    import tokengate.proxy.server as _sv2
    from tokengate.core.config import Settings as _S
    from tokengate.analytics.db import init_db as _init
    from tokengate.core.mock_provider import MockTransport as _MT

    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv2._settings = None
    s = _S()
    # Disable router so the server's fallback call_upstream fires
    # (this test is about the PIPELINE ORDER, not the router itself)
    s.router_enabled = False
    _sv2._settings = s
    _init(s.db_path)

    transport = _MT()
    monkeypatch.setattr(_sv2, "_transport", transport)

    with TestClient(_sv2.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "explain the difference between TCP and UDP in networking"}],
                # No max_tokens — budgeter should inject 1024
            },
        )

    assert resp.status_code == 200
    # The transport records the body sent to upstream
    assert transport.requests[-1].get("max_tokens") == 1024

    _sv2._settings = None
    _sv2._transport = None
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_proxy.py::test_pipeline_order_budgeter_before_router -v
```

Expected: FAIL — because `_PIPELINE` currently has router before budgeter, but since router is a no-op stub the test may actually pass. This test becomes meaningful after Task 6 when router makes the call.

> Note: This test may pass now because the router is still a stub. The real guard is that the test verifies `max_tokens` reaches the transport, which proves budgeter ran before any upstream call.

- [ ] **Step 3: Reorder `_PIPELINE` in `server.py`**

In `tokengate/proxy/server.py`, change line 29:

```python
# Before:
_PIPELINE = [_l_exact, _l_semantic, _l_distiller, _l_compressor, _l_router, _l_budgeter]

# After:
_PIPELINE = [_l_exact, _l_semantic, _l_distiller, _l_compressor, _l_budgeter, _l_router]
```

- [ ] **Step 4: Update `_non_streaming_response` to read router decision**

Replace the `if ctx.response is not None:` block in `_non_streaming_response` (lines 169-190):

```python
    if ctx.response is not None:
        tokens_in = ctx.response.tokens_in
        tokens_out = ctx.response.tokens_out
        model = ctx.response.model
        resp_body = ctx.response.raw_body

        router_decision = next(
            (d for d in ctx.decisions if d.layer == "router" and d.action == "applied"),
            None,
        )
        if router_decision:
            est_cost = router_decision.detail["est_cost_usd"]
            est_saved = router_decision.detail["est_saved_usd"]
            escalated = int(router_decision.detail["escalated"])
        else:
            # cache hit — no router decision
            est_cost = 0.0
            est_saved = compute_cost(model, tokens_in, tokens_out, s) or 0.0
            escalated = 0
    else:
        try:
            upstream_resp = await call_upstream(req, s, transport=_transport)
            tokens_in = upstream_resp.tokens_in
            tokens_out = upstream_resp.tokens_out
            model = upstream_resp.model
            resp_body = upstream_resp.raw_body
            for writer in ctx.cache_writers:
                await writer(upstream_resp)
        except UpstreamError as e:
            status = "upstream_error"
            error_detail = str(e)
            resp_body = e.body
        est_cost = compute_cost(model, tokens_in, tokens_out, s)
        est_saved = 0.0
        escalated = 0
```

Also update the `write_row` call to pass `escalated`:

```python
    write_row(
        s.db_path,
        ts=start_ts, route=req.route, status=status, error_detail=error_detail,
        layers_applied=[asdict(d) for d in ctx.decisions],
        tokens_in_raw=tokens_in_raw, tokens_in_final=tokens_in or None,
        tokens_out=tokens_out or None, model_used=model,
        cache_kind=cache_kind,
        escalated=escalated,
        latency_ms=latency_ms,
        est_cost_usd=est_cost,
        est_saved_usd=est_saved,
    )
```

- [ ] **Step 5: Run all existing tests to verify nothing broke**

```
pytest tests/test_proxy.py tests/test_budgeter.py tests/test_distiller.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add tokengate/proxy/server.py tests/test_proxy.py
git commit -m "feat: reorder pipeline (budgeter before router) and wire router cost/escalated to DB"
```

---

### Task 3: Router Foundation — Transport + Skip Paths

**Files:**
- Replace: `tokengate/layers/router.py`

- [ ] **Step 1: Write failing tests for skip paths**

Add to `tests/test_router.py`:

```python
import copy
import pytest
import httpx
from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayRequest
import tokengate.layers.router as _router


def make_req(
    route="openai",
    stream=False,
    tools=None,
    messages=None,
    max_tokens=None,
    raw_headers=None,
) -> GatewayRequest:
    return GatewayRequest(
        messages=messages or [{"role": "user", "content": "hello world how are you doing today"}],
        model="gpt-4o",
        stream=stream,
        max_tokens=max_tokens,
        temperature=None,
        tools=tools or [],
        route=route,
        raw_headers=raw_headers or {},
        extra={},
    )


def make_ctx(req: GatewayRequest, settings) -> LayerContext:
    return LayerContext(request=req, settings=settings)


@pytest.mark.asyncio
async def test_streaming_skip(tmp_path):
    s = make_settings(tmp_path)
    req = make_req(stream=True)
    ctx = make_ctx(req, s)
    result = await _router.apply(ctx)
    assert result.response is None
    skip = next(d for d in result.decisions if d.layer == "router")
    assert skip.action == "skip"
    assert skip.detail["reason"] == "streaming"


@pytest.mark.asyncio
async def test_disabled_skip(tmp_path):
    s = make_settings(tmp_path)
    s.router_enabled = False
    req = make_req()
    ctx = make_ctx(req, s)
    result = await _router.apply(ctx)
    assert result.response is None
    skip = next(d for d in result.decisions if d.layer == "router")
    assert skip.action == "skip"
    assert skip.detail["reason"] == "disabled"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_router.py::test_streaming_skip tests/test_router.py::test_disabled_skip -v
```

Expected: FAIL — router is a no-op stub with no decisions

- [ ] **Step 3: Replace `router.py` with foundation**

```python
from __future__ import annotations
import re
from dataclasses import replace
from typing import Optional

import httpx

from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayRequest, GatewayResponse
from tokengate.core.provider import call_upstream, UpstreamError
from tokengate.core.tokens import compute_cost

# Module-level transport override for tests (mirrors distiller.py pattern)
_transport: Optional[httpx.AsyncBaseTransport] = None


def set_transport(t: Optional[httpx.AsyncBaseTransport]) -> None:
    global _transport
    _transport = t


async def apply(ctx: LayerContext) -> LayerContext:
    req = ctx.request
    settings = ctx.settings

    # 1. Disabled check
    if not settings.router_enabled:
        ctx.decisions.append(LayerDecision("router", "skip", {"reason": "disabled"}))
        return ctx

    # 2. Streaming skip
    if req.stream:
        ctx.decisions.append(LayerDecision("router", "skip", {"reason": "streaming"}))
        return ctx

    return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_router.py::test_streaming_skip tests/test_router.py::test_disabled_skip -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tokengate/layers/router.py tests/test_router.py
git commit -m "feat: router foundation — transport injection, streaming skip, disabled skip"
```

---

### Task 4: Difficulty Scoring

**Files:**
- Modify: `tokengate/layers/router.py`

- [ ] **Step 1: Write failing difficulty tests**

Add to `tests/test_router.py`:

```python
from tokengate.layers.router import _score_difficulty


def make_req_with_content(content: str, route="openai", tools=None) -> GatewayRequest:
    return GatewayRequest(
        messages=[{"role": "user", "content": content}],
        model="gpt-4o",
        stream=False,
        max_tokens=None,
        temperature=None,
        tools=tools or [],
        route=route,
        raw_headers={},
        extra={},
    )


def test_difficulty_length_feature(tmp_path):
    s = make_settings(tmp_path)
    # 1000 tokens worth of chars (4000 chars) → min(1000/2000, 0.40) = 0.40 * (1000/2000)...
    # Actually: min(tokens/2000, 0.40). tokens = len//4.
    # 4000 chars → 1000 tokens → min(1000/2000, 0.40) = 0.50, clamped to 0.40
    content = "x" * 4000
    req = make_req_with_content(content)
    score, features = _score_difficulty(req, s)
    assert features["length"] == pytest.approx(0.40)


def test_difficulty_short_length(tmp_path):
    s = make_settings(tmp_path)
    # 200 chars → 50 tokens → min(50/2000, 0.40) = 0.025
    content = "x" * 200
    req = make_req_with_content(content)
    score, features = _score_difficulty(req, s)
    assert features["length"] == pytest.approx(0.025)


def test_difficulty_tools_feature(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("hello", tools=[{"name": "search"}])
    score, features = _score_difficulty(req, s)
    assert features["tools"] == pytest.approx(0.25)


def test_difficulty_no_tools(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("hello")
    score, features = _score_difficulty(req, s)
    assert features["tools"] == 0.0


def test_difficulty_code_feature(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("here is my code:\n```python\nprint('hi')\n```")
    score, features = _score_difficulty(req, s)
    assert features["code"] == pytest.approx(0.15)


def test_difficulty_math_feature(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("compute the integral ∫ of x squared")
    score, features = _score_difficulty(req, s)
    assert features["math"] == pytest.approx(0.10)


def test_difficulty_multi_step_feature(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("step 1: open the file\nstep 2: read contents\nstep 3: close")
    score, features = _score_difficulty(req, s)
    assert features["multi_step"] == pytest.approx(0.10)


def test_difficulty_depth_feature(tmp_path):
    s = make_settings(tmp_path)
    # 10 non-system turns → min(10/20, 0.10) = 0.05
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "msg"} for i in range(10)]
    req = GatewayRequest(
        messages=messages, model="gpt-4o", stream=False, max_tokens=None,
        temperature=None, tools=[], route="openai", raw_headers={}, extra={},
    )
    score, features = _score_difficulty(req, s)
    assert features["depth"] == pytest.approx(0.05)


def test_difficulty_clamped_to_one(tmp_path):
    s = make_settings(tmp_path)
    # All features fire: long + tools + code + math + multi_step + depth
    content = (
        "x" * 4000
        + "\n```python\npass\n```"
        + "\ncompute ∫ x dx"
        + "\nstep 1: do this then step 2"
    )
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "msg"} for i in range(20)]
    messages.append({"role": "user", "content": content})
    req = GatewayRequest(
        messages=messages, model="gpt-4o", stream=False, max_tokens=None,
        temperature=None, tools=[{"name": "fn"}], route="openai", raw_headers={}, extra={},
    )
    score, features = _score_difficulty(req, s)
    assert score <= 1.0
    assert score >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_router.py -k "difficulty" -v
```

Expected: `ImportError: cannot import name '_score_difficulty' from 'tokengate.layers.router'`

- [ ] **Step 3: Implement `_score_difficulty` in `router.py`**

Add after the imports, before `apply()`:

```python
_CODE_FENCE_RE = re.compile(r"```", re.MULTILINE)
_MATH_SYMBOL_RE = re.compile(r"[∑∫√±×÷=∂]|\\b(sin|cos|integral|derivative)\\b")
_MULTI_STEP_RE = re.compile(
    r"(step\s+\d|^\d+\.\s|\bthen\b\s+\b(do|run|call|use|make|set|get|go|open|close|read|write|create|delete|update|check|ensure|verify))",
    re.IGNORECASE | re.MULTILINE,
)


def _user_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
    return " ".join(parts)


def _score_difficulty(req: GatewayRequest, settings) -> tuple[float, dict]:
    """Score request difficulty as float in [0.0, 1.0] with per-feature breakdown."""
    user_text = _user_text(req.messages)
    all_text = " ".join(
        (m.get("content", "") if isinstance(m.get("content"), str) else "")
        for m in req.messages
    )

    # Token count proxy (chars ÷ 4)
    tokens = len(all_text) // 4
    length_score = min(tokens / 2000, 0.40)

    tools_score = 0.25 if req.tools else 0.0

    code_score = 0.15 if _CODE_FENCE_RE.search(user_text) else 0.0

    math_score = 0.10 if _MATH_SYMBOL_RE.search(user_text) else 0.0

    multi_step_score = 0.10 if _MULTI_STEP_RE.search(user_text) else 0.0

    non_system_turns = sum(1 for m in req.messages if m.get("role") != "system")
    depth_score = min(non_system_turns / 20, 0.10)

    features = {
        "length": length_score,
        "tools": tools_score,
        "code": code_score,
        "math": math_score,
        "multi_step": multi_step_score,
        "depth": depth_score,
    }
    total = min(sum(features.values()), 1.0)
    return total, features
```

- [ ] **Step 4: Run difficulty tests**

```
pytest tests/test_router.py -k "difficulty" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tokengate/layers/router.py tests/test_router.py
git commit -m "feat: implement difficulty scoring heuristics in router"
```

---

### Task 5: Tier Selection + Strong-Direct Path

**Files:**
- Modify: `tokengate/layers/router.py`

- [ ] **Step 1: Write failing tier selection tests**

Add to `tests/test_router.py`. These need a mock transport that records calls:

```python
class _RecordTransport(httpx.AsyncBaseTransport):
    """Records model used per call; returns valid mock response."""
    def __init__(self):
        self.calls: list[str] = []  # model names

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        self.calls.append(body.get("model", ""))
        is_ant = "/v1/messages" in str(request.url)
        if is_ant:
            resp = {
                "id": "msg_mock", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "4"}],
                "model": body.get("model", "mock"),
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        else:
            resp = {
                "id": "chatcmpl-mock", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
                "model": body.get("model", "mock"),
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        return httpx.Response(200, json=resp)


@pytest.mark.asyncio
async def test_client_override_strong(tmp_path):
    """x-tokengate-tier: strong header forces strong model, no self-check."""
    s = make_settings(tmp_path)
    transport = _RecordTransport()
    _router.set_transport(transport)
    try:
        req = make_req(raw_headers={"x-tokengate-tier": "strong"})
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["tier"] == "strong"
        assert applied.detail["reason"] == "client_override"
        assert applied.detail["escalated"] is False
        assert applied.detail["confidence_score"] is None
        # Only one upstream call (the strong model)
        assert len(transport.calls) == 1
        assert transport.calls[0] == s.router_strong_model["openai"]
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_tools_forced_strong(tmp_path):
    """tools present + tools_tier=strong → strong model, reason=tools_forced_strong."""
    s = make_settings(tmp_path)
    s.router_tools_tier = "strong"
    transport = _RecordTransport()
    _router.set_transport(transport)
    try:
        req = make_req(tools=[{"name": "search"}])
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["tier"] == "strong"
        assert applied.detail["reason"] == "tools_forced_strong"
        assert len(transport.calls) == 1
        assert transport.calls[0] == s.router_strong_model["openai"]
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_above_threshold_strong(tmp_path):
    """Difficulty >= threshold → strong directly, no check."""
    s = make_settings(tmp_path)
    s.router_difficulty_threshold = 0.0  # everything is above threshold
    transport = _RecordTransport()
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["tier"] == "strong"
        assert applied.detail["reason"] == "above_threshold"
        assert len(transport.calls) == 1
    finally:
        _router.set_transport(None)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_router.py::test_client_override_strong tests/test_router.py::test_tools_forced_strong tests/test_router.py::test_above_threshold_strong -v
```

Expected: FAIL — `apply()` doesn't call upstream yet

- [ ] **Step 3: Implement tier selection + strong-direct path in `apply()`**

Replace the stub `apply()` in `router.py` (keep all the existing code above it):

```python
def _call_model(req: GatewayRequest, model_name: str) -> GatewayRequest:
    """Return a copy of req with model replaced."""
    return GatewayRequest(
        messages=req.messages,
        model=model_name,
        stream=False,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        tools=req.tools,
        route=req.route,
        raw_headers=req.raw_headers,
        extra=req.extra,
    )


async def apply(ctx: LayerContext) -> LayerContext:
    req = ctx.request
    settings = ctx.settings

    # 1. Disabled check
    if not settings.router_enabled:
        ctx.decisions.append(LayerDecision("router", "skip", {"reason": "disabled"}))
        return ctx

    # 2. Streaming skip
    if req.stream:
        ctx.decisions.append(LayerDecision("router", "skip", {"reason": "streaming"}))
        return ctx

    # 3. Score difficulty
    difficulty, features = _score_difficulty(req, settings)

    cheap_model = settings.router_cheap_model[req.route]
    strong_model = settings.router_strong_model[req.route]

    # 4. Tier selection (first match wins)
    if req.tools and settings.router_tools_tier == "strong":
        tier = "strong"
        reason = "tools_forced_strong"
    elif req.raw_headers.get("x-tokengate-tier") == "strong":
        tier = "strong"
        reason = "client_override"
    elif difficulty >= settings.router_difficulty_threshold:
        tier = "strong"
        reason = "above_threshold"
    else:
        tier = "cheap"
        reason = "below_threshold"

    # 5. Strong-direct path (no self-check)
    if tier == "strong":
        synth = _call_model(req, strong_model)
        strong_resp = await call_upstream(synth, settings, transport=_transport)
        ctx.response = strong_resp

        strong_cost = compute_cost(strong_model, strong_resp.tokens_in, strong_resp.tokens_out, settings) or 0.0
        baseline_cost = strong_cost  # strong IS the baseline

        ctx.decisions.append(LayerDecision("router", "applied", {
            "difficulty": difficulty,
            "features": features,
            "tier": "strong",
            "model": strong_model,
            "reason": reason,
            "escalated": False,
            "confidence_score": None,
            "escalation_reason": None,
            "est_cost_usd": strong_cost,
            "baseline_cost_usd": baseline_cost,
            "baseline_is_estimate": True,
            "est_saved_usd": 0.0,
        }))
        return ctx

    # 6. Cheap path (handled in next task)
    return ctx
```

- [ ] **Step 4: Run tier selection tests**

```
pytest tests/test_router.py::test_client_override_strong tests/test_router.py::test_tools_forced_strong tests/test_router.py::test_above_threshold_strong -v
```

Expected: all PASS

- [ ] **Step 5: Also run skip tests to confirm no regression**

```
pytest tests/test_router.py -v
```

Expected: all defined tests PASS

- [ ] **Step 6: Commit**

```bash
git add tokengate/layers/router.py tests/test_router.py
git commit -m "feat: tier selection and strong-direct path in cascade router"
```

---

### Task 6: Cheap Path, Self-Check, and Escalation

**Files:**
- Modify: `tokengate/layers/router.py`

- [ ] **Step 1: Write failing cheap + self-check + escalation tests**

Add to `tests/test_router.py`:

```python
class _ConfidenceTransport(httpx.AsyncBaseTransport):
    """
    First call (cheap model) → returns 'Mock response'.
    Second call (self-check) → returns the digit in `confidence`.
    Third call if escalated (strong model) → returns 'Strong response'.
    """
    def __init__(self, confidence: str = "4", strong_fail: bool = False):
        self.confidence = confidence
        self.strong_fail = strong_fail
        self.calls: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        self.calls.append(body)
        call_index = len(self.calls)
        is_ant = "/v1/messages" in str(request.url)
        model = body.get("model", "mock")

        # Third call: strong model escalation
        if call_index == 3:
            if self.strong_fail:
                err = _json.dumps({"error": {"message": "strong fail", "type": "server_error"}})
                return httpx.Response(500, content=err.encode(), headers={"content-type": "application/json"})
            text = "Strong response"
        # Second call: self-check (max_tokens=5)
        elif call_index == 2:
            text = self.confidence
        # First call: cheap model response
        else:
            text = "Mock response"

        if is_ant:
            resp = {
                "id": "msg_mock", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": model, "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": len(text.split())},
            }
        else:
            resp = {
                "id": "chatcmpl-mock", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "model": model,
                "usage": {"prompt_tokens": 10, "completion_tokens": len(text.split()), "total_tokens": 10 + len(text.split())},
            }
        return httpx.Response(200, json=resp)


@pytest.mark.asyncio
async def test_cheap_high_confidence_served(tmp_path):
    """Confidence=4 (>threshold=3) → serve cheap, est_saved_usd > 0."""
    s = make_settings(tmp_path)
    transport = _ConfidenceTransport(confidence="4")
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["tier"] == "cheap"
        assert applied.detail["escalated"] is False
        assert applied.detail["confidence_score"] == 4
        assert applied.detail["est_saved_usd"] > 0
        assert result.response is not None
        assert result.response.content == "Mock response"
        # 2 calls: cheap + check
        assert len(transport.calls) == 2
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_cheap_low_confidence_escalates(tmp_path):
    """Confidence=2 (<=threshold=3) → escalate to strong, est_saved_usd < 0."""
    s = make_settings(tmp_path)
    transport = _ConfidenceTransport(confidence="2")
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["tier"] == "cheap"
        assert applied.detail["escalated"] is True
        assert applied.detail["confidence_score"] == 2
        assert applied.detail["escalation_reason"] == "low_confidence"
        assert applied.detail["est_saved_usd"] < 0
        assert result.response.content == "Strong response"
        # 3 calls: cheap + check + strong
        assert len(transport.calls) == 3
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_self_check_call_fails_escalates(tmp_path):
    """Self-check 500 → escalate, reason=check_call_failed."""
    s = make_settings(tmp_path)

    class _CheckFailTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.calls = []
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            import json as _j
            body = _j.loads(request.content)
            self.calls.append(body)
            call_index = len(self.calls)
            model = body.get("model", "mock")
            is_ant = "/v1/messages" in str(request.url)
            if call_index == 2:  # self-check fails
                err = _j.dumps({"error": {"message": "fail", "type": "server_error"}})
                return httpx.Response(500, content=err.encode(), headers={"content-type": "application/json"})
            text = "Mock response" if call_index == 1 else "Strong response"
            if is_ant:
                resp = {"id": "msg_mock", "type": "message", "role": "assistant",
                        "content": [{"type": "text", "text": text}], "model": model,
                        "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2}}
            else:
                resp = {"id": "chatcmpl-mock", "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "model": model, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
            return httpx.Response(200, json=resp)

    transport = _CheckFailTransport()
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["escalated"] is True
        assert applied.detail["escalation_reason"] == "check_call_failed"
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_self_check_parse_fails_escalates(tmp_path):
    """Self-check returns non-digit → escalate, reason=check_parse_failed."""
    s = make_settings(tmp_path)
    transport = _ConfidenceTransport(confidence="excellent")
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["escalated"] is True
        assert applied.detail["escalation_reason"] == "check_parse_failed"
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_self_check_call_params(tmp_path):
    """Self-check call must use max_tokens=5 and temperature=0."""
    s = make_settings(tmp_path)
    transport = _ConfidenceTransport(confidence="4")
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        await _router.apply(ctx)
        check_call = transport.calls[1]
        assert check_call.get("max_tokens") == 5
        assert check_call.get("temperature") == 0
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_strong_direct_no_check(tmp_path):
    """Strong-direct path (above threshold) never runs self-check."""
    s = make_settings(tmp_path)
    s.router_difficulty_threshold = 0.0
    transport = _RecordTransport()
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        await _router.apply(ctx)
        assert len(transport.calls) == 1  # only the strong call, no check
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_cheap_upstream_error_propagates(tmp_path):
    """Cheap upstream 500 → UpstreamError propagates (not caught by router)."""
    from tokengate.core.provider import UpstreamError as _UE
    s = make_settings(tmp_path)

    class _AlwaysErrorTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            import json as _j
            err = _j.dumps({"error": {"message": "fail", "type": "server_error"}})
            return httpx.Response(500, content=err.encode(), headers={"content-type": "application/json"})

    _router.set_transport(_AlwaysErrorTransport())
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        with pytest.raises(_UE):
            await _router.apply(ctx)
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_escalation_strong_fails_serves_cheap(tmp_path):
    """Strong fails after escalation → serve cheap response, reason=escalation_failed_served_cheap."""
    s = make_settings(tmp_path)
    transport = _ConfidenceTransport(confidence="2", strong_fail=True)
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        applied = next(d for d in result.decisions if d.layer == "router" and d.action == "applied")
        assert applied.detail["escalated"] is True
        assert applied.detail["escalation_reason"] == "escalation_failed_served_cheap"
        assert result.response.content == "Mock response"
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_both_fail_propagates_upstream_error(tmp_path):
    """Cheap succeeds but both cheap and strong can't both fail simultaneously (strong fails → serve cheap).
    Test: when cheap call itself fails → UpstreamError propagates."""
    from tokengate.core.provider import UpstreamError as _UE
    s = make_settings(tmp_path)

    class _AlwaysErrorTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            import json as _j
            err = _j.dumps({"error": {"message": "fail", "type": "server_error"}})
            return httpx.Response(500, content=err.encode(), headers={"content-type": "application/json"})

    _router.set_transport(_AlwaysErrorTransport())
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        with pytest.raises(_UE):
            await _router.apply(ctx)
    finally:
        _router.set_transport(None)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_router.py -k "cheap or self_check or escalat or confidence or propagat" -v
```

Expected: all FAIL — cheap path not implemented yet

- [ ] **Step 3: Implement cheap path, self-check, and escalation in `apply()`**

Replace the `# 6. Cheap path (handled in next task)` placeholder at the end of `apply()` with:

```python
    # 6. Cheap path
    cheap_resp = await call_upstream(_call_model(req, cheap_model), settings, transport=_transport)

    cheap_cost = compute_cost(cheap_model, cheap_resp.tokens_in, cheap_resp.tokens_out, settings) or 0.0

    # 7. Self-check (escalation check)
    escalated = False
    confidence_score: Optional[int] = None
    escalation_reason: Optional[str] = None

    if settings.router_escalation_enabled:
        last_user = ""
        for m in reversed(req.messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                last_user = content[:500] if isinstance(content, str) else ""
                break

        check_prompt = (
            "Does this response fully and correctly answer the question below?\n"
            "Rate your confidence 1 (not at all) to 5 (completely).\n"
            "Reply with exactly one digit and nothing else.\n\n"
            f"Question: {last_user}\n"
            f"Response: {cheap_resp.content[:1000]}"
        )
        check_req = GatewayRequest(
            messages=[{"role": "user", "content": check_prompt}],
            model=cheap_model,
            stream=False,
            max_tokens=5,
            temperature=0,
            tools=[],
            route=req.route,
            raw_headers=req.raw_headers,
            extra={},
        )

        try:
            check_resp = await call_upstream(check_req, settings, transport=_transport)
            raw_digit = check_resp.content.strip()
            if raw_digit in {"1", "2", "3", "4", "5"}:
                confidence_score = int(raw_digit)
                if confidence_score <= settings.router_escalation_threshold:
                    escalated = True
                    escalation_reason = "low_confidence"
            else:
                escalated = True
                escalation_reason = "check_parse_failed"
            check_cost = compute_cost(cheap_model, check_resp.tokens_in, check_resp.tokens_out, settings) or 0.0
        except UpstreamError:
            escalated = True
            escalation_reason = "check_call_failed"
            check_cost = 0.0
    else:
        check_cost = 0.0

    # 8. Escalation: call strong model
    if escalated:
        try:
            strong_resp = await call_upstream(_call_model(req, strong_model), settings, transport=_transport)
            served_resp = strong_resp
            strong_cost = compute_cost(strong_model, strong_resp.tokens_in, strong_resp.tokens_out, settings) or 0.0
        except UpstreamError:
            # Strong failed — serve cheap (better than a 500)
            served_resp = cheap_resp
            strong_cost = 0.0
            escalation_reason = "escalation_failed_served_cheap"
    else:
        served_resp = cheap_resp
        strong_cost = 0.0

    ctx.response = served_resp

    # 9. Cost accounting
    baseline_cost = compute_cost(
        strong_model,
        served_resp.tokens_in,
        served_resp.tokens_out,
        settings,
    ) or 0.0
    est_cost = cheap_cost + check_cost + strong_cost
    est_saved = baseline_cost - est_cost

    ctx.decisions.append(LayerDecision("router", "applied", {
        "difficulty": difficulty,
        "features": features,
        "tier": "cheap",
        "model": served_resp.model,
        "reason": reason,
        "escalated": escalated,
        "confidence_score": confidence_score,
        "escalation_reason": escalation_reason,
        "est_cost_usd": est_cost,
        "baseline_cost_usd": baseline_cost,
        "baseline_is_estimate": True,
        "est_saved_usd": est_saved,
    }))
    return ctx
```

- [ ] **Step 4: Run all router unit tests**

```
pytest tests/test_router.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full test suite to confirm no regressions**

```
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add tokengate/layers/router.py tests/test_router.py
git commit -m "feat: cheap path, self-check, and escalation logic in cascade router"
```

---

### Task 7: Integration Tests in `test_proxy.py`

**Files:**
- Modify: `tests/test_proxy.py`

- [ ] **Step 1: Write the 3 remaining integration tests**

Add to the Phase 4 section of `tests/test_proxy.py` (after `test_pipeline_order_budgeter_before_router`):

```python
class _RouterTransport(httpx.AsyncBaseTransport):
    """
    Simulates router's cheap + check + optional strong calls.
    call_index 1: cheap response
    call_index 2: confidence digit
    call_index 3: strong response (if escalated)
    """
    def __init__(self, confidence: str = "4"):
        self.confidence = confidence
        self.calls: list[dict] = []

    async def handle_async_request(self, request: _httpx.Request) -> _httpx.Response:
        body = _json.loads(request.content)
        self.calls.append(body)
        call_index = len(self.calls)
        is_ant = "/v1/messages" in str(request.url)
        model = body.get("model", "mock")

        if call_index == 2:
            text = self.confidence
            tokens_out = 1
        elif call_index == 3:
            text = "Strong response"
            tokens_out = 2
        else:
            text = "Cheap response"
            tokens_out = 2

        if is_ant:
            resp = {
                "id": "msg_mock", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": model, "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": tokens_out},
            }
        else:
            resp = {
                "id": "chatcmpl-mock", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "model": model,
                "usage": {"prompt_tokens": 10, "completion_tokens": tokens_out, "total_tokens": 10 + tokens_out},
            }
        return _httpx.Response(200, json=resp)


def test_escalated_request_sets_db_flag(tmp_path, monkeypatch):
    """Escalated request → requests.escalated=1 in DB."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.router_difficulty_threshold = 1.1  # force cheap path
    s.router_escalation_threshold = 5    # always escalate (confidence ≤ 5)
    _sv._settings = s
    init_db(s.db_path)

    transport = _RouterTransport(confidence="2")  # low confidence → escalate
    monkeypatch.setattr(_sv, "_transport", transport)
    monkeypatch.setattr(_l_router, "_transport", transport)

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "what is two plus two?"}]},
        )

    assert resp.status_code == 200
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT escalated, est_saved_usd FROM requests WHERE status='ok'").fetchone())
    con.close()
    assert row["escalated"] == 1
    assert row["est_saved_usd"] < 0

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_router, "_transport", None)


def test_non_escalated_cheap_positive_savings(tmp_path, monkeypatch):
    """Non-escalated cheap routing → est_saved_usd > 0 in DB."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.router_difficulty_threshold = 1.1  # force cheap path
    _sv._settings = s
    init_db(s.db_path)

    transport = _RouterTransport(confidence="5")  # high confidence → no escalation
    monkeypatch.setattr(_sv, "_transport", transport)
    monkeypatch.setattr(_l_router, "_transport", transport)

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "what is the capital of France?"}]},
        )

    assert resp.status_code == 200
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT escalated, est_saved_usd FROM requests WHERE status='ok'").fetchone())
    con.close()
    assert row["escalated"] == 0
    assert row["est_saved_usd"] > 0

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_router, "_transport", None)
```

- [ ] **Step 2: Run integration tests**

```
pytest tests/test_proxy.py -k "phase_4 or escalat or savings or pipeline_order" -v
```

Expected: all PASS

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_proxy.py
git commit -m "test: Phase 4 integration tests — escalation DB flag, savings sign, pipeline order"
```

---

### Task 8: `retrain_router.py` Script

**Files:**
- Create: `scripts/retrain_router.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""
Offline router retraining script.

Loads cheap-routed request history from the DB and fits a LogisticRegression
classifier to predict escalation probability. Saves coefficients to
data_dir/router_model.pkl.

SELECTION BIAS WARNING: This model only learns the difficulty boundary within
the cheap-routed region. Requests routed directly to strong (above threshold,
tool override, or client override) have no escalation label and are excluded.
The model cannot predict outcomes for requests in the strong-direct region.

Usage:
    python scripts/retrain_router.py [--db PATH] [--out PATH]
"""
from __future__ import annotations
import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path


_FEATURE_KEYS = ["length", "tools", "code", "math", "multi_step", "depth"]


def load_from_db(db_path: Path) -> tuple[list[list[float]], list[int]]:
    """
    Load feature vectors and escalation labels for cheap-routed requests only.

    Returns (features, labels) where:
    - features: list of [length, tools, code, math, multi_step, depth] vectors
    - labels: list of int (1=escalated, 0=not escalated)
    """
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT layers_applied FROM requests WHERE status='ok'"
    ).fetchall()
    con.close()

    features: list[list[float]] = []
    labels: list[int] = []

    for (layers_json,) in rows:
        try:
            layers = json.loads(layers_json)
        except Exception:
            continue

        router_decision = next(
            (l for l in layers if l.get("layer") == "router" and l.get("action") == "applied"),
            None,
        )
        if router_decision is None:
            continue

        detail = router_decision.get("detail", {})
        # Only include cheap-routed requests (selection bias: strong-direct has no label)
        if detail.get("tier") != "cheap":
            continue

        feat_dict = detail.get("features", {})
        vec = [float(feat_dict.get(k, 0.0)) for k in _FEATURE_KEYS]
        escalated = int(bool(detail.get("escalated", False)))

        features.append(vec)
        labels.append(escalated)

    return features, labels


def retrain(db_path: Path, out_path: Path) -> None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    features, labels = load_from_db(db_path)

    if len(features) < 10:
        print(f"ERROR: Only {len(features)} cheap-routed samples found. Need ≥10 to retrain.", file=sys.stderr)
        sys.exit(1)

    print(f"Training on {len(features)} samples ({sum(labels)} escalated, {len(labels)-sum(labels)} not).")
    print("Selection bias: model only covers cheap-routed region. See module docstring.")

    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_keys": _FEATURE_KEYS}, f)

    print(f"Model saved to {out_path}")
    print(f"Coefficients: {dict(zip(_FEATURE_KEYS, model.coef_[0].tolist()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain cascade router difficulty model.")
    parser.add_argument("--db", type=Path, default=None, help="Path to tokengate.db")
    parser.add_argument("--out", type=Path, default=None, help="Output path for router_model.pkl")
    args = parser.parse_args()

    data_dir = Path("~/.rait").expanduser()
    db_path = args.db or (data_dir / "tokengate.db")
    out_path = args.out or (data_dir / "router_model.pkl")

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    retrain(db_path, out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write a smoke test for the data loader**

Add to `tests/test_router.py`:

```python
def test_retrain_load_from_db_filters_cheap_only(tmp_path):
    """load_from_db returns only cheap-routed rows and correct feature shape."""
    import json
    import sqlite3
    from scripts.retrain_router import load_from_db

    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = make_settings(tmp_path)

    db = s.db_path
    con = sqlite3.connect(db)

    cheap_layers = json.dumps([{
        "layer": "router", "action": "applied",
        "detail": {
            "tier": "cheap", "escalated": False,
            "features": {"length": 0.1, "tools": 0.0, "code": 0.0, "math": 0.0, "multi_step": 0.0, "depth": 0.05},
        }
    }])
    strong_layers = json.dumps([{
        "layer": "router", "action": "applied",
        "detail": {
            "tier": "strong", "escalated": False,
            "features": {"length": 0.4, "tools": 0.25, "code": 0.0, "math": 0.0, "multi_step": 0.0, "depth": 0.1},
        }
    }])

    import time
    con.execute(
        "INSERT INTO requests (ts, route, status, layers_applied, est_saved_usd) VALUES (?,?,?,?,?)",
        (time.time(), "openai", "ok", cheap_layers, 0.001),
    )
    con.execute(
        "INSERT INTO requests (ts, route, status, layers_applied, est_saved_usd) VALUES (?,?,?,?,?)",
        (time.time(), "openai", "ok", strong_layers, 0.0),
    )
    con.commit()
    con.close()

    features, labels = load_from_db(db)
    assert len(features) == 1  # only cheap row
    assert len(features[0]) == 6
    assert labels[0] == 0  # not escalated
```

- [ ] **Step 3: Run the smoke test**

```
pytest tests/test_router.py::test_retrain_load_from_db_filters_cheap_only -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/retrain_router.py tests/test_router.py
git commit -m "feat: retrain_router.py offline script with selection bias documentation"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Run the complete test suite**

```
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 2: Verify router decision appears in stats endpoint**

Start the server (optional manual check):

```bash
TOKENGATE_DATA_DIR=/tmp/tg-test uvicorn tokengate.proxy.server:app --port 8787
```

Send a request and check `/stats` — confirm `est_saved_usd` and `escalated` rows appear correctly.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Phase 4 complete — cascade router with difficulty scoring, self-check, and cost accounting"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec section | Task covering it |
|---|---|
| §1 Pipeline reorder (budgeter before router) | Task 2 |
| §2 Streaming skip with logged reason | Task 3 |
| §3 Router settings in yaml/Settings | Task 1 |
| §4 Difficulty scoring (6 features, clamped) | Task 4 |
| §5 Tier selection (6-step cascade) | Task 5 |
| §6.1 Same `call_upstream` function, `synth_req` | Tasks 5 & 6 |
| §6.2 Cheap UpstreamError propagates | Task 6 |
| §6.3 Strong fails after escalation → serve cheap | Task 6 |
| §7.1 Micro-prompt with max_tokens=5, temperature=0 | Task 6 |
| §7.2 Digit parsing | Task 6 |
| §7.3 Fail-safe: check failure → always escalate | Task 6 |
| §7.4 No escalation loops (strong never checked) | Task 5 |
| §8.1 Baseline counterfactual cost | Tasks 5 & 6 |
| §8.2 est_saved_usd formula (negative when escalated) | Task 6 |
| §8.3 Server reads router decision for est_cost/est_saved/escalated | Task 2 |
| §9 LayerDecision schema complete | Tasks 5 & 6 |
| §10 retrain_router.py with selection bias docs | Task 8 |
| §11 No new DB columns needed | Task 2 (write_row already has escalated) |
| §12.1 Unit tests | Tasks 3–6 |
| §12.2 Integration tests | Task 7 |

All spec sections covered. No gaps found.
