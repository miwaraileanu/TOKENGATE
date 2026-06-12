from __future__ import annotations
import ipaddress
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
from fastapi import FastAPI, Request
import json as _json
from fastapi.responses import JSONResponse, Response, StreamingResponse
from tokengate.core.normalize import normalize_openai, normalize_anthropic, serialize_for_upstream
from tokengate.core.provider import call_upstream, stream_upstream, UpstreamError
from tokengate.core.tokens import compute_cost

from tokengate.core.config import Settings
from tokengate.core.context import LayerContext
from tokengate.analytics.db import init_db, write_row
from tokengate.analytics.stats import get_stats
import tokengate.layers.exact_cache as _l_exact
import tokengate.layers.semantic_cache as _l_semantic
import tokengate.layers.distiller as _l_distiller
import tokengate.layers.compressor as _l_compressor
import tokengate.layers.router as _l_router
import tokengate.layers.budgeter as _l_budgeter


_PIPELINE = [_l_exact, _l_semantic, _l_distiller, _l_compressor, _l_router, _l_budgeter]

# Module-level transport override — set in tests via monkeypatch
_transport: httpx.AsyncBaseTransport | None = None
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "::1")


def check_startup(settings: Settings) -> None:
    """Raise SystemExit(1) if binding to non-loopback without a key."""
    if settings.bind != "127.0.0.1" and not settings.tokengate_key:
        print(
            "ERROR: TOKENGATE_KEY must be set when binding to a non-loopback address. "
            "Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    check_startup(s)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(s.db_path)
    yield


app = FastAPI(title="TokenGate", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.client and not _is_loopback(request.client.host):
        s = get_settings()
        key = request.headers.get("x-tokengate-key", "")
        if key != s.tokengate_key:
            try:
                write_row(
                    s.db_path,
                    ts=time.time(),
                    route=str(request.url.path),
                    status="auth_error",
                )
            except Exception:
                pass
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


async def _run_pipeline(ctx: LayerContext) -> LayerContext:
    for layer in _PIPELINE:
        ctx = await layer.apply(ctx)
        if ctx.response is not None:
            break
    return ctx


@app.get("/stats")
async def stats_endpoint():
    s = get_settings()
    return get_stats(s.db_path)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    s = get_settings()
    start_ts = time.time()
    body = await request.json()
    req = normalize_openai(body, dict(request.headers))
    ctx = LayerContext(request=req)
    ctx = await _run_pipeline(ctx)
    return await _handle_request(req, ctx, s, start_ts)


@app.post("/v1/messages")
async def messages(request: Request):
    s = get_settings()
    start_ts = time.time()
    body = await request.json()
    req = normalize_anthropic(body, dict(request.headers))
    ctx = LayerContext(request=req)
    ctx = await _run_pipeline(ctx)
    return await _handle_request(req, ctx, s, start_ts)


async def _handle_request(req, ctx: LayerContext, s: Settings, start_ts: float):
    if req.stream:
        return await _streaming_response(req, ctx, s, start_ts)
    return await _non_streaming_response(req, ctx, s, start_ts)


async def _non_streaming_response(req, ctx: LayerContext, s: Settings, start_ts: float):
    status = "ok"
    error_detail = None
    resp_body: dict = {}
    tokens_in = tokens_out = 0
    model = req.model

    if ctx.response is not None:
        # Pipeline short-circuited (cache hit) — use the cached response
        tokens_in = ctx.response.tokens_in
        tokens_out = ctx.response.tokens_out
        model = ctx.response.model
        resp_body = ctx.response.raw_body
    else:
        try:
            upstream_resp = await call_upstream(req, s, transport=_transport)
            tokens_in = upstream_resp.tokens_in
            tokens_out = upstream_resp.tokens_out
            model = upstream_resp.model
            resp_body = upstream_resp.raw_body
        except UpstreamError as e:
            status = "upstream_error"
            error_detail = str(e)
            resp_body = e.body

    cost = compute_cost(model, tokens_in, tokens_out, s)
    latency_ms = int((time.time() - start_ts) * 1000)
    write_row(
        s.db_path,
        ts=start_ts, route=req.route, status=status, error_detail=error_detail,
        layers_applied=[asdict(d) for d in ctx.decisions],
        tokens_in_raw=tokens_in or None, tokens_in_final=tokens_in or None,
        tokens_out=tokens_out or None, model_used=model,
        latency_ms=latency_ms, est_cost_usd=cost,
    )

    tg_headers = {
        "x-tokengate-cache": "none",
        "x-tokengate-model": model,
        "x-tokengate-saved-tokens": "0",
    }
    http_status = 200 if status == "ok" else 502
    return JSONResponse(content=resp_body, status_code=http_status, headers=tg_headers)


async def _streaming_response(req, ctx: LayerContext, s: Settings, start_ts: float):
    tg_headers = {
        "x-tokengate-cache": "none",
        "x-tokengate-model": req.model,
        "x-tokengate-saved-tokens": "0",
        "cache-control": "no-cache",
    }

    async def generate():
        status = "ok"
        error_detail = None
        tokens_in = tokens_out = 0
        model = req.model

        try:
            async for chunk in stream_upstream(req, s, transport=_transport):
                if isinstance(chunk, dict) and chunk.get("_usage"):
                    tokens_in = chunk["tokens_in"]
                    tokens_out = chunk["tokens_out"]
                    model = chunk.get("model", model)
                else:
                    yield chunk
        except UpstreamError as e:
            status = "upstream_error"
            error_detail = str(e)
            yield _json.dumps(e.body).encode()
        except Exception as e:
            status = "upstream_error"
            error_detail = str(e)
        finally:
            cost = compute_cost(model, tokens_in, tokens_out, s)
            latency_ms = int((time.time() - start_ts) * 1000)
            write_row(
                s.db_path,
                ts=start_ts, route=req.route, status=status, error_detail=error_detail,
                layers_applied=[asdict(d) for d in ctx.decisions],
                tokens_in_raw=tokens_in or None, tokens_in_final=tokens_in or None,
                tokens_out=tokens_out or None, model_used=model,
                latency_ms=latency_ms, est_cost_usd=cost,
            )

    return StreamingResponse(generate(), media_type="text/event-stream", headers=tg_headers)
