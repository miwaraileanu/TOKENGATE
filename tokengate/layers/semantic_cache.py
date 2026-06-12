from __future__ import annotations
from tokengate.core.context import LayerContext


async def apply(ctx: LayerContext) -> LayerContext:
    """Phase 1 stub — no-op pass-through."""
    return ctx
