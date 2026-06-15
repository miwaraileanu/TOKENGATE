from __future__ import annotations
import hashlib
import json
import sqlite3
import time

from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayResponse


def _compute_key(req) -> str:
    temp_bucket = round(req.temperature, 1) if req.temperature is not None else None
    payload = {
        "messages": req.messages,
        "model": req.model,
        "temp": temp_bucket,
        "tools": req.tools,
        "max_tokens": req.max_tokens,
        "extra": req.extra,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def _write_eligible(req) -> bool:
    if req.raw_headers.get("x-tokengate-cache-write", "").lower() == "always":
        return True
    return req.temperature is None or req.temperature <= 0.3


async def apply(ctx: LayerContext) -> LayerContext:
    if ctx.settings is None:
        return ctx

    req = ctx.request

    if req.stream:
        ctx.decisions.append(LayerDecision("exact_cache", "skip", {"reason": "streaming"}))
        return ctx

    if not _write_eligible(req):
        ctx.decisions.append(LayerDecision("exact_cache", "skip", {"reason": "temperature_too_high"}))
        return ctx

    db_path = ctx.settings.db_path
    ttl = ctx.settings.cache_exact_ttl
    cache_key = _compute_key(req)
    now = time.time()

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT body_json FROM cache_exact WHERE cache_key=? AND expires_at>?",
        (cache_key, now),
    ).fetchone()
    con.close()

    if row:
        body = json.loads(row[0])
        ctx.response = GatewayResponse(**body)
        ctx.decisions.append(LayerDecision("exact_cache", "hit", {"key": cache_key}))
        return ctx

    ctx.decisions.append(LayerDecision("exact_cache", "miss", {"key": cache_key}))

    async def _write(response: GatewayResponse) -> None:
        body_json = json.dumps({
            "content": response.content,
            "model": response.model,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "stop_reason": response.stop_reason,
            "raw_body": response.raw_body,
        })
        expires_at = time.time() + ttl
        _con = sqlite3.connect(db_path)
        _con.execute(
            "INSERT OR REPLACE INTO cache_exact (cache_key, expires_at, body_json) VALUES (?,?,?)",
            (cache_key, expires_at, body_json),
        )
        _con.commit()
        _con.close()

    ctx.cache_writers.append(_write)
    return ctx
