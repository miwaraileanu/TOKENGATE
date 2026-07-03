from __future__ import annotations
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
