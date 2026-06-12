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
