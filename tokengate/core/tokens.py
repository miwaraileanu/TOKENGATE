from __future__ import annotations
import json
from tokengate.core.config import Settings


def compute_cost(
    model: str, tokens_in: int, tokens_out: int, settings: Settings
) -> float | None:
    """Returns cost in USD, or None if model is not in the price table."""
    if model not in settings.prices:
        return None
    input_price, output_price = settings.prices[model]
    return (tokens_in * input_price + tokens_out * output_price) / 1_000_000


def extract_usage_openai(body: dict) -> tuple[int, int]:
    u = body.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def extract_usage_anthropic(body: dict) -> tuple[int, int]:
    u = body.get("usage", {})
    return u.get("input_tokens", 0), u.get("output_tokens", 0)


def parse_streaming_usage(full_text: str, route: str) -> tuple[int, int]:
    """Parse (tokens_in, tokens_out) from accumulated SSE text."""
    if route == "openai":
        tokens_in = tokens_out = 0
        for line in full_text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        tokens_in = u.get("prompt_tokens", tokens_in)
                        tokens_out = u.get("completion_tokens", tokens_out)
                except (json.JSONDecodeError, KeyError):
                    pass
        return tokens_in, tokens_out
    else:  # anthropic
        tokens_in = tokens_out = 0
        for line in full_text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
                if event.get("type") == "message_start":
                    u = event.get("message", {}).get("usage", {})
                    tokens_in = u.get("input_tokens", tokens_in)
                elif event.get("type") == "message_delta":
                    u = event.get("usage", {})
                    tokens_out = u.get("output_tokens", tokens_out)
            except (json.JSONDecodeError, KeyError):
                pass
        return tokens_in, tokens_out
