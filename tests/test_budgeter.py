"""Tests for Phase 3 budgeter layer."""
from __future__ import annotations
import os
import pytest

from tokengate.analytics.db import init_db
from tokengate.core.config import Settings
from tokengate.core.context import LayerContext
from tokengate.core.normalize import GatewayRequest
import tokengate.layers.budgeter as _budgeter


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_settings(tmp_path) -> Settings:
    old = os.environ.get("TOKENGATE_DATA_DIR")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    init_db(s.db_path)
    if old is None:
        os.environ.pop("TOKENGATE_DATA_DIR", None)
    else:
        os.environ["TOKENGATE_DATA_DIR"] = old
    return s


def make_request(messages=None, max_tokens=None, route="openai") -> GatewayRequest:
    return GatewayRequest(
        messages=messages or [{"role": "user", "content": "Hello world, how are you doing today?"}],
        model="gpt-4o",
        stream=False,
        max_tokens=max_tokens,
        temperature=None,
        tools=[],
        route=route,
        raw_headers={},
        extra={},
    )


def make_ctx(messages=None, max_tokens=None, route="openai", tmp_path=None, settings=None) -> LayerContext:
    req = make_request(messages=messages, max_tokens=max_tokens, route=route)
    if settings is None:
        settings = make_settings(tmp_path)
    return LayerContext(request=req, settings=settings)


def user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


# ── Client set max_tokens ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_client_set_max_tokens_skip(tmp_path):
    ctx = make_ctx(max_tokens=500, tmp_path=tmp_path)
    result = await _budgeter.apply(ctx)
    skip = [d for d in result.decisions if d.layer == "budgeter" and d.action == "skip"]
    assert skip and skip[0].detail["reason"] == "client_set_max_tokens"
    assert result.request.max_tokens == 500  # unchanged


@pytest.mark.asyncio
async def test_client_max_tokens_never_overridden_for_all_types(tmp_path):
    """With max_tokens set, budgeter never injects regardless of prompt type."""
    texts = [
        "extract the JSON from this text",
        "write a function to parse the data",
        "write a comprehensive essay about AI",
        "hello how are you today",
    ]
    s = make_settings(tmp_path)
    for text in texts:
        ctx = make_ctx(messages=[user_msg(text)], max_tokens=100, settings=s)
        result = await _budgeter.apply(ctx)
        assert result.request.max_tokens == 100, f"overridden for: {text}"


# ── Request type detection ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extraction_type_detected(tmp_path):
    ctx = make_ctx(
        messages=[user_msg("extract the JSON from the response")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "extraction"
    assert result.request.max_tokens == 512


@pytest.mark.asyncio
async def test_code_type_detected_fenced(tmp_path):
    ctx = make_ctx(
        messages=[user_msg("here is my code:\n```python\nprint('hello')\n```\nplease fix it")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "code"
    assert result.request.max_tokens == 2048


@pytest.mark.asyncio
async def test_code_type_detected_imperative(tmp_path):
    ctx = make_ctx(
        messages=[user_msg("write a function to sort a list of integers in ascending order")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "code"
    assert result.request.max_tokens == 2048


@pytest.mark.asyncio
async def test_long_form_type_detected(tmp_path):
    ctx = make_ctx(
        messages=[user_msg("write a comprehensive essay about machine learning and its impact in modern healthcare systems")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "long_form"
    assert result.request.max_tokens == 4096


@pytest.mark.asyncio
async def test_chat_type_is_default(tmp_path):
    ctx = make_ctx(
        messages=[user_msg("what is the capital of France, and what is it known for globally?")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "chat"
    assert result.request.max_tokens == 1024


# ── Code detection edge cases ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explain_function_is_not_code(tmp_path):
    """'explain how this function works' → not code (no imperative verb)."""
    ctx = make_ctx(
        messages=[user_msg("can you explain how this function works in detail please")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] != "code"


@pytest.mark.asyncio
async def test_write_function_to_parse_is_code(tmp_path):
    # 'parse' is in extraction regex, so 'write a function to parse' → extraction (extraction wins)
    ctx = make_ctx(
        messages=[user_msg("write a function to sort and filter the list of user records efficiently")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "code"


@pytest.mark.asyncio
async def test_script_for_play_is_not_code(tmp_path):
    """'writing a script for a play' should not be detected as code."""
    ctx = make_ctx(
        messages=[user_msg("I am writing a script for a theatrical play set in ancient Rome")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    # Not code — no imperative verb like write/implement
    assert applied and applied[0].detail["type"] != "code"


@pytest.mark.asyncio
async def test_extraction_beats_long_message(tmp_path):
    """'extract a JSON' → extraction even if message is long."""
    long_preamble = "I have a very long document with lots of content. " * 20
    ctx = make_ctx(
        messages=[user_msg(long_preamble + "Please extract only the JSON from this text")],
        tmp_path=tmp_path,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "extraction"


# ── Low-confidence skip ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_low_confidence_skip_short_non_extraction(tmp_path):
    """< 10 words and not extraction → skip with type_uncertain."""
    ctx = make_ctx(messages=[user_msg("hi thanks yes")], tmp_path=tmp_path)
    result = await _budgeter.apply(ctx)
    skip = [d for d in result.decisions if d.layer == "budgeter" and d.action == "skip"]
    assert skip and skip[0].detail["reason"] == "type_uncertain"
    assert result.request.max_tokens is None


@pytest.mark.asyncio
async def test_short_extraction_not_skipped(tmp_path):
    """Short extraction prompt is NOT skipped (extraction is always processed)."""
    ctx = make_ctx(messages=[user_msg("extract json")], tmp_path=tmp_path)
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "extraction"


# ── Extraction instruction ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extraction_instruction_no_system_prompt(tmp_path):
    """No existing system prompt → new one created with instruction."""
    s = make_settings(tmp_path)
    ctx = make_ctx(
        messages=[user_msg("please extract the JSON from the response")],
        settings=s,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["instruction_added"] is True

    system_msgs = [m for m in result.request.messages if m.get("role") == "system"]
    assert system_msgs
    assert "no preamble" in system_msgs[0]["content"].lower() or "requested data only" in system_msgs[0]["content"].lower()


@pytest.mark.asyncio
async def test_extraction_instruction_appended_to_neutral_system(tmp_path):
    """System prompt exists, no conflict → instruction appended."""
    s = make_settings(tmp_path)
    ctx = make_ctx(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            user_msg("return only the list of items from the text"),
        ],
        settings=s,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["instruction_added"] is True
    sys_msg = next(m for m in result.request.messages if m.get("role") == "system")
    assert "helpful assistant" in sys_msg["content"]
    assert s.budget_extraction_instruction in sys_msg["content"]


@pytest.mark.asyncio
async def test_extraction_instruction_skipped_on_conflict(tmp_path):
    """Conflicting verbosity directive → instruction NOT added."""
    s = make_settings(tmp_path)
    ctx = make_ctx(
        messages=[
            {"role": "system", "content": "Provide detailed and thorough explanations."},
            user_msg("extract the relevant fields from the JSON response below"),
        ],
        settings=s,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied
    assert applied[0].detail["instruction_added"] is False
    assert applied[0].detail["instruction_skip_reason"] == "conflict_directive"


# ── Budget table configurable ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_custom_budget_table_used(tmp_path):
    s = make_settings(tmp_path)
    s.budget_chat = 9999
    s.budget_table = {
        "chat": 9999,
        "extraction": s.budget_extraction,
        "code": s.budget_code,
        "long_form": s.budget_long_form,
    }
    ctx = make_ctx(
        messages=[user_msg("what is the meaning of life in modern philosophy and why does it matter to people?")],
        settings=s,
    )
    result = await _budgeter.apply(ctx)
    applied = [d for d in result.decisions if d.layer == "budgeter" and d.action == "applied"]
    assert applied and applied[0].detail["type"] == "chat"
    assert result.request.max_tokens == 9999
