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
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, transport, s
    _sv._settings = None
    _sv._transport = None


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
    monkeypatch.setattr(_sv, "_transport", MockTransport(mode="error", error_status=500))
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
