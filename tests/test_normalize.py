import pytest
from tokengate.core.normalize import (
    GatewayRequest, GatewayResponse,
    normalize_openai, normalize_anthropic, serialize_for_upstream,
)


# ── OpenAI normalization ──────────────────────────────────────────────────────

def test_normalize_openai_basic():
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    req = normalize_openai(body, {})
    assert req.route == "openai"
    assert req.model == "gpt-4o"
    assert req.stream is False
    assert req.messages == [{"role": "user", "content": "hi"}]
    assert req.extra == {}


def test_normalize_openai_extra_fields():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 0.9,
        "stop_sequences": ["END"],
        "response_format": {"type": "json_object"},
    }
    req = normalize_openai(body, {})
    assert req.extra == {"top_p": 0.9, "stop_sequences": ["END"], "response_format": {"type": "json_object"}}


def test_serialize_openai_roundtrip_with_extra():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["END"],
    }
    req = normalize_openai(body, {})
    out = serialize_for_upstream(req)
    assert out["top_p"] == 0.9
    assert out["stop_sequences"] == ["END"]
    assert out["temperature"] == 0.7


# ── Anthropic normalization ───────────────────────────────────────────────────

def test_normalize_anthropic_injects_system():
    body = {
        "model": "claude-sonnet-4-6",
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
    }
    req = normalize_anthropic(body, {})
    assert req.route == "anthropic"
    assert req.messages[0] == {"role": "system", "content": "You are helpful."}
    assert req.messages[1] == {"role": "user", "content": "hi"}
    assert req.max_tokens == 1024


def test_normalize_anthropic_extra_fields():
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "top_p": 0.95,
        "metadata": {"user_id": "u123"},
    }
    req = normalize_anthropic(body, {})
    assert req.extra == {"top_p": 0.95, "metadata": {"user_id": "u123"}}


def test_serialize_anthropic_roundtrip():
    body = {
        "model": "claude-sonnet-4-6",
        "system": "Be concise.",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "top_p": 0.95,
    }
    req = normalize_anthropic(body, {})
    out = serialize_for_upstream(req)
    assert out["system"] == "Be concise."
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["top_p"] == 0.95


def test_no_system_anthropic():
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    req = normalize_anthropic(body, {})
    out = serialize_for_upstream(req)
    assert "system" not in out


def test_raw_headers_stored():
    body = {"model": "x", "messages": [], "stream": False}
    req = normalize_openai(body, {"authorization": "Bearer sk-123"})
    assert req.raw_headers["authorization"] == "Bearer sk-123"
