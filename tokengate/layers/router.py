from __future__ import annotations
from typing import Optional
import re

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


_CODE_FENCE_RE = re.compile(r"```", re.MULTILINE)
_MATH_SYMBOL_RE = re.compile(r"[∑∫√±×÷=∂]|\b(sin|cos|integral|derivative)\b", re.IGNORECASE)
_MULTI_STEP_RE = re.compile(
    r"(step\s+\d|^\d+\.\s|\bthen\b\s+\b(do|run|call|use|make|set|get|go|open|close|read|write|create|delete|update|check|ensure|verify)\b)",
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
    depth_score = min(non_system_turns / 200, 0.10)

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
