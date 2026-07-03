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


def _call_model(req: GatewayRequest, model_name: str) -> GatewayRequest:
    """Return a copy of req with only the model name replaced."""
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

    # 4. Tier selection — first match wins
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

    # 5. Strong-direct path (no self-check ever)
    if tier == "strong":
        strong_resp = await call_upstream(
            _call_model(req, strong_model), settings, transport=_transport
        )
        ctx.response = strong_resp

        strong_cost = (
            compute_cost(strong_model, strong_resp.tokens_in, strong_resp.tokens_out, settings)
            or 0.0
        )

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
            "baseline_cost_usd": strong_cost,
            "baseline_is_estimate": True,
            "est_saved_usd": 0.0,
        }))
        return ctx

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
