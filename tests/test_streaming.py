import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from tokengate.analytics.db import init_db
from tokengate.core.config import Settings
from tokengate.core.mock_provider import MockTransport
import tokengate.proxy.server as _sv


@pytest.fixture
def streaming_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _sv._settings = None
    s = Settings()
    _sv._settings = s
    init_db(s.db_path)
    transport = MockTransport()
    monkeypatch.setattr(_sv, "_transport", transport)
    with TestClient(_sv.app, raise_server_exceptions=True) as c:
        yield c, s
    _sv._settings = None
    _sv._transport = None


def test_openai_streaming_yields_sse(streaming_client):
    client, _ = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.read().decode()
    assert "[DONE]" in body


def test_anthropic_streaming_yields_sse(streaming_client):
    client, _ = streaming_client
    with client.stream(
        "POST", "/v1/messages",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 100, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode()
    assert "message_stop" in body


def test_streaming_x_tokengate_headers_before_body(streaming_client):
    client, _ = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        # Headers are received before reading any body
        assert resp.headers.get("x-tokengate-cache") == "none"
        assert resp.headers.get("x-tokengate-saved-tokens") == "0"
        assert "x-tokengate-model" in resp.headers


def test_streaming_analytics_written_after_stream(streaming_client):
    client, s = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        resp.read()  # consume full stream

    con = sqlite3.connect(s.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM requests").fetchone())
    con.close()
    assert row["tokens_in_raw"] == 10
    assert row["tokens_out"] == 5
    assert row["status"] == "ok"


def test_streaming_token_counts_from_final_usage_event(streaming_client):
    """Token counts must come from the SSE usage event, not be estimated."""
    client, s = streaming_client
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        resp.read()

    con = sqlite3.connect(s.db_path)
    row = con.execute("SELECT tokens_in_raw, tokens_out FROM requests").fetchone()
    con.close()
    # MockTransport reports exactly 10 in / 5 out in the usage event
    assert row[0] == 10
    assert row[1] == 5
