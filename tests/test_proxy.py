import os
import pytest
from tokengate.core.config import Settings
from tokengate.core.normalize import normalize_openai, normalize_anthropic
from tokengate.core.provider import call_upstream, UpstreamError
from tokengate.core.mock_provider import MockTransport
from fastapi.testclient import TestClient
from tokengate.analytics.db import init_db
import tokengate.proxy.server as _sv


@pytest.fixture
def settings(tmp_path):
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    yield s
    del os.environ["TOKENGATE_DATA_DIR"]


@pytest.fixture
def transport():
    return MockTransport()


@pytest.mark.asyncio
async def test_openai_passthrough(settings, transport):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    req = normalize_openai(body, {"authorization": "Bearer sk-test"})
    resp = await call_upstream(req, settings, transport=transport)
    assert resp.tokens_in == 10
    assert resp.tokens_out == 5
    assert resp.content == "Mock response"


@pytest.mark.asyncio
async def test_anthropic_passthrough(settings, transport):
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    req = normalize_anthropic(body, {"x-api-key": "sk-ant-test"})
    resp = await call_upstream(req, settings, transport=transport)
    assert resp.tokens_in == 10
    assert resp.tokens_out == 5
    assert resp.content == "Mock response"


@pytest.mark.asyncio
async def test_extra_fields_reach_upstream(settings):
    recorded = MockTransport()
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 0.9,
        "stop_sequences": ["END"],
    }
    req = normalize_openai(body, {})
    await call_upstream(req, settings, transport=recorded)
    sent_body = recorded.requests[0]
    assert sent_body["top_p"] == 0.9
    assert sent_body["stop_sequences"] == ["END"]


@pytest.mark.asyncio
async def test_upstream_error_raises(settings):
    transport = MockTransport(mode="error", error_status=500)
    req = normalize_openai({"model": "gpt-4o", "messages": []}, {})
    with pytest.raises(UpstreamError) as exc:
        await call_upstream(req, settings, transport=transport)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_upstream_429_raises(settings):
    transport = MockTransport(mode="error", error_status=429)
    req = normalize_openai({"model": "gpt-4o", "messages": []}, {})
    with pytest.raises(UpstreamError) as exc:
        await call_upstream(req, settings, transport=transport)
    assert exc.value.status_code == 429


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Returns a configured TestClient with mock transport and temp data dir."""
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    _sv._settings = s
    init_db(s.db_path)
    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    monkeypatch.setattr(_l_router, "_transport", transport)
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, transport, s
    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_router, "_transport", None)


def test_openai_endpoint_returns_200(test_client):
    client, transport, _ = test_client
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Mock response"


def test_anthropic_endpoint_returns_200(test_client):
    client, transport, _ = test_client
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "Mock response"


def test_response_headers_present(test_client):
    client, _, _ = test_client
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "x-tokengate-cache" in resp.headers
    assert "x-tokengate-model" in resp.headers
    assert "x-tokengate-saved-tokens" in resp.headers
    assert resp.headers["x-tokengate-cache"] == "none"
    assert resp.headers["x-tokengate-saved-tokens"] == "0"


def test_analytics_row_written_on_success(test_client):
    import sqlite3
    client, _, s = test_client
    client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests WHERE status='ok'").fetchone())
    con.close()
    assert row["tokens_in_raw"] == 10
    assert row["tokens_out"] == 5
    assert row["model_used"] == "mock"
    assert row["route"] == "openai"


def test_analytics_row_on_upstream_error(test_client, monkeypatch):
    import sqlite3
    client, _, s = test_client
    error_transport = MockTransport(mode="error", error_status=500)
    monkeypatch.setattr(_sv, "_transport", error_transport)
    monkeypatch.setattr(_l_router, "_transport", error_transport)
    resp = client.post("/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": []},
                       headers={"accept": "application/json"})
    assert resp.status_code == 502
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests WHERE status='upstream_error'").fetchone())
    con.close()
    assert row["status"] == "upstream_error"
    assert row["error_detail"] is not None


# ── Phase 3 integration tests ─────────────────────────────────────────────────

import json as _json
import httpx as _httpx
import tokengate.layers.distiller as _l_distiller
import tokengate.layers.budgeter as _l_budgeter


class _SummaryTransport(_httpx.AsyncBaseTransport):
    """Returns a valid summary JSON for any call (used by distiller integration tests)."""

    def __init__(self, summary="Compressed summary", pinned_facts=None):
        self.summary = summary
        self.pinned_facts = pinned_facts or []
        self.calls: list[dict] = []

    async def handle_async_request(self, request: _httpx.Request) -> _httpx.Response:
        body = _json.loads(request.content)
        self.calls.append(body)
        # Determine if this is a distiller summarization call or the main upstream call
        # Distiller calls use cheap model; main call uses original model
        text = _json.dumps({"summary": self.summary, "pinned_facts": self.pinned_facts})
        is_anthropic = "/v1/messages" in str(request.url)
        if is_anthropic:
            resp_body = {
                "id": "msg_mock", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": body.get("model", "mock"),
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        else:
            resp_body = {
                "id": "chatcmpl-mock", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "model": body.get("model", "mock"),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        return _httpx.Response(200, json=resp_body)


def test_distiller_fires_on_long_history(tmp_path, monkeypatch):
    """Long conversation history triggers distiller → upstream receives fewer tokens."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.distill_threshold_tokens = 1  # force trigger
    s.distill_keep_recent_turns = 1
    _sv._settings = s
    init_db(s.db_path)

    summary_transport = _SummaryTransport()
    monkeypatch.setattr(_sv, "_transport", summary_transport)
    monkeypatch.setattr(_l_distiller, "_transport", summary_transport)
    monkeypatch.setattr(_l_router, "_transport", summary_transport)

    # Build a long message list
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 200} for i in range(10)]
    messages.append({"role": "user", "content": "recent question"})

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": messages, "max_tokens": 1024},
        )

    assert resp.status_code == 200
    # Distiller should have been applied
    con = sqlite3.connect(s.db_path)
    rows = con.execute("SELECT layers_applied FROM requests").fetchall()
    con.close()
    assert rows
    layers = _json.loads(rows[0][0])
    distiller_decisions = [l for l in layers if l.get("layer") == "distiller"]
    assert distiller_decisions
    applied = [l for l in distiller_decisions if l.get("action") == "applied"]
    assert applied, f"distiller not applied: {distiller_decisions}"

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_distiller, "_transport", None)
    monkeypatch.setattr(_l_router, "_transport", None)


def test_distiller_failsafe_preserves_original_history(tmp_path, monkeypatch):
    """When distiller fails, original history still reaches upstream (partial path)."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.distill_threshold_tokens = 1
    s.distill_keep_recent_turns = 1
    _sv._settings = s
    init_db(s.db_path)

    class _FailTransport(_httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: _httpx.Request) -> _httpx.Response:
            # 500 for distiller call, normal response for main call
            # Since we can't tell them apart easily, return 500 always
            err = _json.dumps({"error": {"message": "fail", "type": "server_error"}})
            return _httpx.Response(500, content=err.encode(), headers={"content-type": "application/json"})

    fail_transport = _FailTransport()
    monkeypatch.setattr(_sv, "_transport", fail_transport)
    monkeypatch.setattr(_l_distiller, "_transport", fail_transport)
    monkeypatch.setattr(_l_router, "_transport", fail_transport)

    messages = [
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "y" * 200},
        {"role": "user", "content": "recent"},
    ]
    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": messages, "max_tokens": 1024},
        )

    # Main call failed (transport returns 500), so we get 502 from server
    assert resp.status_code == 502

    con = sqlite3.connect(s.db_path)
    rows = con.execute("SELECT layers_applied FROM requests").fetchall()
    con.close()
    assert rows
    layers = _json.loads(rows[0][0])
    distiller_decisions = [l for l in layers if l.get("layer") == "distiller"]
    assert distiller_decisions
    # Must be "skip" (fail-safe), not "applied"
    assert all(l["action"] == "skip" for l in distiller_decisions)

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_distiller, "_transport", None)
    monkeypatch.setattr(_l_router, "_transport", None)


def test_budgeter_injects_max_tokens_on_uncapped_request(tmp_path, monkeypatch):
    """Budgeter injects max_tokens when client does not set one."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    _sv._settings = s
    init_db(s.db_path)

    normal_transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", normal_transport)
    monkeypatch.setattr(_l_router, "_transport", normal_transport)

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "can you explain how neural networks learn from data over time?"}],
                # No max_tokens
            },
        )

    assert resp.status_code == 200
    # Check that budgeter applied (injected max_tokens into request body sent to upstream)
    assert len(normal_transport.requests) >= 1
    # Chat default = 1024
    assert normal_transport.requests[-1].get("max_tokens") == 1024

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_router, "_transport", None)


def test_distiller_tokens_in_raw_recorded(tmp_path, monkeypatch):
    """tokens_in_raw in DB reflects pre-distillation token count from distiller decision."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.distill_threshold_tokens = 1
    s.distill_keep_recent_turns = 1
    _sv._settings = s
    init_db(s.db_path)

    summary_transport = _SummaryTransport()
    monkeypatch.setattr(_sv, "_transport", summary_transport)
    monkeypatch.setattr(_l_distiller, "_transport", summary_transport)
    monkeypatch.setattr(_l_router, "_transport", summary_transport)

    messages = [
        {"role": "user", "content": "a" * 300},
        {"role": "assistant", "content": "b" * 300},
        {"role": "user", "content": "recent question here"},
    ]
    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": messages, "max_tokens": 1024},
        )

    assert resp.status_code == 200
    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests WHERE status='ok'").fetchone())
    con.close()

    layers = _json.loads(row["layers_applied"])
    distiller_applied = next((l for l in layers if l["layer"] == "distiller" and l["action"] == "applied"), None)

    if distiller_applied:
        # tokens_in_raw should equal distiller's tokens_in detail
        assert row["tokens_in_raw"] == distiller_applied["detail"]["tokens_in"]
        # tokens_in_final should be the actual upstream tokens (from mock: 10)
        assert row["tokens_in_final"] == 10

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_distiller, "_transport", None)
    monkeypatch.setattr(_l_router, "_transport", None)


def test_budgeter_est_saved_usd_zero(tmp_path, monkeypatch):
    """Budgeter contributes 0 to est_saved_usd (output savings unmeasurable).
    Router is disabled so that est_saved_usd in the DB row reflects only the
    budgeter+server-fallback path (no router cost accounting)."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.router_enabled = False  # isolate budgeter; router contributes its own savings
    _sv._settings = s
    init_db(s.db_path)

    normal_transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", normal_transport)

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "write a function to add two numbers and return the sum"}],
            },
        )

    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT est_saved_usd, layers_applied FROM requests WHERE status='ok'").fetchone())
    con.close()

    layers = _json.loads(row["layers_applied"])
    budgeter_decisions = [l for l in layers if l["layer"] == "budgeter" and l["action"] == "applied"]
    assert budgeter_decisions, "budgeter should have applied"
    # est_saved_usd for non-cache, non-router requests is always 0
    assert row["est_saved_usd"] == 0.0

    _sv._settings = None
    _sv._transport = None


# ── Phase 4 integration tests ─────────────────────────────────────────────────

import tokengate.layers.router as _l_router


def test_pipeline_order_budgeter_before_router(tmp_path, monkeypatch):
    """Budgeter's max_tokens injection reaches upstream before any call is made.
    Uses router_enabled=False so the server fallback fires — proving budgeter
    ran before any upstream call regardless of which code path makes it."""
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.router_enabled = False
    _sv._settings = s
    init_db(s.db_path)

    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "explain the key difference between TCP and UDP in networking"}],
                # No max_tokens — budgeter should inject 1024
            },
        )

    assert resp.status_code == 200
    assert transport.requests[-1].get("max_tokens") == 1024

    _sv._settings = None
    _sv._transport = None


def test_budgeter_max_tokens_reaches_router_upstream(tmp_path, monkeypatch):
    """Budgeter's injected max_tokens is present in the upstream call the router
    makes. Both _sv._transport and _l_router._transport are patched to the same
    MockTransport — the call is captured whether made by the router (after Task 6)
    or by the server fallback (current stub). router_escalation_enabled=False
    keeps it to a single upstream call so transport.requests[0] is unambiguous.

    The router-decision assertion ensures this cannot silently pass via the server
    fallback: the router must have an 'applied' decision with a tier in its detail.
    """
    import sqlite3
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    s.router_escalation_enabled = False  # single call; no self-check noise
    _sv._settings = s
    init_db(s.db_path)

    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    monkeypatch.setattr(_l_router, "_transport", transport)

    with TestClient(_sv.app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "explain the difference between TCP and UDP in networking protocols"}],
                # No max_tokens — budgeter must inject 1024 before any upstream call
            },
        )

    assert resp.status_code == 200
    # First upstream call carries budgeter's max_tokens regardless of who made it
    assert transport.requests[0].get("max_tokens") == 1024

    # Proof the ROUTER made the call, not the server fallback
    con = sqlite3.connect(s.db_path)
    row = con.execute("SELECT layers_applied FROM requests WHERE status='ok'").fetchone()
    con.close()
    layers = _json.loads(row[0])
    router_applied = next(
        (l for l in layers if l["layer"] == "router" and l["action"] == "applied"),
        None,
    )
    assert router_applied is not None, (
        "router did not fire an 'applied' decision — max_tokens may have reached "
        "the server fallback, not the router's own upstream call"
    )
    assert "tier" in router_applied["detail"], "router applied decision missing 'tier' key"

    _sv._settings = None
    _sv._transport = None
    monkeypatch.setattr(_l_router, "_transport", None)
