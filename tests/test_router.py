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
