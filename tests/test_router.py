"""Unit tests for Phase 4 cascade router."""
from __future__ import annotations
import os
import pytest
from tokengate.core.config import Settings
from tokengate.analytics.db import init_db


def make_settings(tmp_path) -> Settings:
    old = os.environ.get("TOKENGATE_DATA_DIR")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    init_db(s.db_path)
    if old is None:
        os.environ.pop("TOKENGATE_DATA_DIR", None)
    else:
        os.environ["TOKENGATE_DATA_DIR"] = old
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


import copy
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
    # 4000 chars → 1000 tokens → min(1000/2000, 0.40) = 0.40 (capped at max)
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


def test_difficulty_no_code(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("what is the capital of France?")
    score, features = _score_difficulty(req, s)
    assert features["code"] == 0.0


def test_difficulty_math_symbol(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("compute the integral ∫ of x squared")
    score, features = _score_difficulty(req, s)
    assert features["math"] == pytest.approx(0.10)


def test_difficulty_math_keyword(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("what is the derivative of sin(x) with respect to x")
    score, features = _score_difficulty(req, s)
    assert features["math"] == pytest.approx(0.10)


def test_difficulty_no_math(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("tell me a story about a dog")
    score, features = _score_difficulty(req, s)
    assert features["math"] == 0.0


def test_difficulty_multi_step_numbered(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("step 1: open the file\nstep 2: read it\nstep 3: close it")
    score, features = _score_difficulty(req, s)
    assert features["multi_step"] == pytest.approx(0.10)


def test_difficulty_no_multi_step(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("what is the capital of France and why is it famous")
    score, features = _score_difficulty(req, s)
    assert features["multi_step"] == 0.0


def test_difficulty_depth_feature(tmp_path):
    s = make_settings(tmp_path)
    # 10 non-system turns → min(10/200, 0.10) = 0.05
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "msg"}
        for i in range(10)
    ]
    req = GatewayRequest(
        messages=messages, model="gpt-4o", stream=False, max_tokens=None,
        temperature=None, tools=[], route="openai", raw_headers={}, extra={},
    )
    score, features = _score_difficulty(req, s)
    assert features["depth"] == pytest.approx(0.05)


def test_difficulty_clamped_to_one(tmp_path):
    s = make_settings(tmp_path)
    # All features fire simultaneously
    content = (
        "x" * 4000
        + "\n```python\npass\n```"
        + "\ncompute ∫ x dx"
        + "\nstep 1: do this\nstep 2: do that"
    )
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "msg"}
        for i in range(20)
    ]
    messages.append({"role": "user", "content": content})
    req = GatewayRequest(
        messages=messages, model="gpt-4o", stream=False, max_tokens=None,
        temperature=None, tools=[{"name": "fn"}], route="openai", raw_headers={}, extra={},
    )
    score, features = _score_difficulty(req, s)
    assert 0.0 <= score <= 1.0


def test_difficulty_returns_all_feature_keys(tmp_path):
    s = make_settings(tmp_path)
    req = make_req_with_content("hello world")
    score, features = _score_difficulty(req, s)
    assert set(features.keys()) == {"length", "tools", "code", "math", "multi_step", "depth"}


class _RecordTransport(httpx.AsyncBaseTransport):
    """Records model used per call; returns a valid 1-token mock response."""

    def __init__(self):
        self.calls: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json as _j
        body = _j.loads(request.content)
        self.calls.append(body)
        model = body.get("model", "mock")
        is_ant = "/v1/messages" in str(request.url)
        if is_ant:
            resp = {
                "id": "msg_mock", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "4"}],
                "model": model, "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        else:
            resp = {
                "id": "chatcmpl-mock", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
                "model": model,
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
        assert len(transport.calls) == 1
        assert transport.calls[0]["model"] == s.router_strong_model["openai"]
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
        assert transport.calls[0]["model"] == s.router_strong_model["openai"]
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_above_threshold_strong(tmp_path):
    """difficulty >= threshold → strong directly, no self-check."""
    s = make_settings(tmp_path)
    s.router_difficulty_threshold = 0.0  # everything scores above
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


@pytest.mark.asyncio
async def test_strong_direct_no_confidence_check(tmp_path):
    """Strong-direct path never runs a self-check (only 1 upstream call total)."""
    s = make_settings(tmp_path)
    s.router_difficulty_threshold = 0.0
    transport = _RecordTransport()
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        await _router.apply(ctx)
        assert len(transport.calls) == 1
    finally:
        _router.set_transport(None)


@pytest.mark.asyncio
async def test_decision_schema_strong_direct(tmp_path):
    """Strong-direct LayerDecision has all required keys with correct types."""
    s = make_settings(tmp_path)
    s.router_difficulty_threshold = 0.0
    transport = _RecordTransport()
    _router.set_transport(transport)
    try:
        req = make_req()
        ctx = make_ctx(req, s)
        result = await _router.apply(ctx)
        d = next(x for x in result.decisions if x.layer == "router" and x.action == "applied")
        assert isinstance(d.detail["difficulty"], float)
        assert isinstance(d.detail["features"], dict)
        assert d.detail["tier"] == "strong"
        assert isinstance(d.detail["model"], str)
        assert isinstance(d.detail["reason"], str)
        assert d.detail["escalated"] is False
        assert d.detail["confidence_score"] is None
        assert d.detail["escalation_reason"] is None
        assert isinstance(d.detail["est_cost_usd"], float)
        assert isinstance(d.detail["baseline_cost_usd"], float)
        assert d.detail["baseline_is_estimate"] is True
        assert isinstance(d.detail["est_saved_usd"], float)
    finally:
        _router.set_transport(None)
