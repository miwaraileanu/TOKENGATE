from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokengate.core.normalize import GatewayRequest, GatewayResponse
    from tokengate.core.config import Settings


@dataclass
class LayerDecision:
    layer: str
    action: str  # "hit" | "miss" | "skip" | "applied" | "escalated"
    detail: dict = field(default_factory=dict)


@dataclass
class LayerContext:
    request: "GatewayRequest"
    response: "GatewayResponse | None" = None
    decisions: list[LayerDecision] = field(default_factory=list)
    settings: "Settings | None" = None
    cache_writers: list = field(default_factory=list)  # list[Callable[[GatewayResponse], Awaitable[None]]]
