import os
import pytest
from tokengate.core.config import Settings
from tokengate.core.normalize import normalize_openai, normalize_anthropic
from tokengate.core.provider import call_upstream, UpstreamError
from tokengate.core.mock_provider import MockTransport


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
