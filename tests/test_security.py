import os
import pytest
from fastapi.testclient import TestClient
from tokengate.core.config import Settings
from tokengate.proxy.server import app, check_startup, _is_loopback
import tokengate.proxy.server as _server


@pytest.fixture(autouse=True)
def reset_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    _server._settings = None  # force re-read
    yield
    _server._settings = None


def make_settings(tmp_path, bind="127.0.0.1", key=""):
    s = Settings.__new__(Settings)
    s.bind = bind
    s.tokengate_key = key
    s.data_dir = tmp_path
    s.db_path = tmp_path / "tokengate.db"
    s.pid_path = tmp_path / "tokengate.pid"
    s.log_path = tmp_path / "logs" / "tokengate.log"
    s.prices = {}
    s.openai_base_url = "https://api.openai.com"
    s.anthropic_base_url = "https://api.anthropic.com"
    return s


def test_check_startup_exits_non_loopback_no_key(tmp_path):
    s = make_settings(tmp_path, bind="0.0.0.0", key="")
    with pytest.raises(SystemExit) as exc:
        check_startup(s)
    assert exc.value.code == 1


def test_check_startup_allows_loopback_no_key(tmp_path):
    s = make_settings(tmp_path, bind="127.0.0.1", key="")
    check_startup(s)  # must not raise


def test_check_startup_allows_non_loopback_with_key(tmp_path):
    s = make_settings(tmp_path, bind="0.0.0.0", key="secret")
    check_startup(s)  # must not raise


def test_is_loopback():
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("localhost") is True
    assert _is_loopback("192.168.1.5") is False
    assert _is_loopback("10.0.0.1") is False


def test_non_loopback_no_key_gets_401(tmp_path, monkeypatch):
    s = make_settings(tmp_path, bind="0.0.0.0", key="secret-key")
    monkeypatch.setattr(_server, "get_settings", lambda: s)
    monkeypatch.setattr(_server, "_is_loopback", lambda host: False)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert resp.status_code == 401


def test_non_loopback_correct_key_passes_auth(tmp_path, monkeypatch):
    from tokengate.analytics.db import init_db
    from tokengate.core.mock_provider import MockTransport
    init_db(tmp_path / "tokengate.db")
    s = make_settings(tmp_path, bind="0.0.0.0", key="secret-key")
    monkeypatch.setattr(_server, "get_settings", lambda: s)
    monkeypatch.setattr(_server, "_is_loopback", lambda host: False)
    monkeypatch.setattr(_server, "_transport", MockTransport())

    client = TestClient(_server.app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-TokenGate-Key": "secret-key"},
    )
    # Passes auth — may succeed or fail at upstream but NOT 401
    assert resp.status_code != 401
