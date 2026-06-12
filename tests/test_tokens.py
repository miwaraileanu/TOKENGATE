import os
import pytest
from pathlib import Path
from tokengate.core.config import Settings, _DEFAULT_PRICES
from tokengate.core.tokens import compute_cost, extract_usage_openai, extract_usage_anthropic, parse_streaming_usage


@pytest.fixture
def tmp_settings(tmp_path):
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    yield s
    del os.environ["TOKENGATE_DATA_DIR"]


def test_default_prices_loaded(tmp_settings):
    assert "claude-sonnet-4-6" in tmp_settings.prices
    assert "gpt-4o" in tmp_settings.prices


def test_yaml_price_override(tmp_path):
    cfg = tmp_path / "tokengate.yaml"
    cfg.write_text("prices:\n  my-model: [1.00, 5.00]\n")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    del os.environ["TOKENGATE_DATA_DIR"]
    assert s.prices["my-model"] == (1.00, 5.00)
    # defaults still present
    assert "gpt-4o" in s.prices


def test_compute_cost_known_model(tmp_settings):
    # claude-sonnet-4-6: input=3.00, output=15.00 per 1M
    cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, tmp_settings)
    assert cost == pytest.approx(18.00)


def test_compute_cost_unknown_model_returns_none(tmp_settings):
    cost = compute_cost("unknown-model-xyz", 100, 50, tmp_settings)
    assert cost is None


def test_extract_usage_openai():
    body = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    assert extract_usage_openai(body) == (10, 5)


def test_extract_usage_anthropic():
    body = {"usage": {"input_tokens": 12, "output_tokens": 8}}
    assert extract_usage_anthropic(body) == (12, 8)


def test_parse_streaming_usage_openai():
    text = (
        'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        'data: [DONE]\n\n'
    )
    assert parse_streaming_usage(text, "openai") == (10, 5)


def test_parse_streaming_usage_anthropic():
    text = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
        'event: message_stop\n'
        'data: {"type":"message_stop"}\n\n'
    )
    assert parse_streaming_usage(text, "anthropic") == (10, 5)
