from __future__ import annotations
import ipaddress
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
