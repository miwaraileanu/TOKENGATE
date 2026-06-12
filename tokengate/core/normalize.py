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
