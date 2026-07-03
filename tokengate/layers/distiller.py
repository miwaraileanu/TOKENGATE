from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import time
from typing import Optional

import httpx

from tokengate.core.context import LayerContext, LayerDecision
from tokengate.core.normalize import GatewayRequest
from tokengate.core.provider import call_upstream, UpstreamError

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False

# Injectable transport for tests (mirrors server.py pattern)
_transport: Optional[httpx.AsyncBaseTransport] = None


def set_transport(t: Optional[httpx.AsyncBaseTransport]) -> None:
    global _transport
    _transport = t


_DISTILL_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}

_PINNED_PATTERNS = re.compile(
    r"\b(my name is|I prefer|I need you to|always use|never use|call me)\b",
    re.IGNORECASE,
)

_FULL_PROMPT = (
    "Summarize this conversation. Preserve all facts, decisions, and user preferences.\n"
    "Reproduce numbers, names, IDs, and unresolved questions verbatim.\n"
    "Be concise.\n"
    'Return JSON only: {"summary": "...", "pinned_facts": ["..."]}\n'
    "where pinned_facts is a list of explicit user preferences, constraints, or\n"
    'instructions (e.g. "my name is X", "always use Python 3", "never truncate output").\n\n'
)

_INCREMENTAL_TEMPLATE = (
    "Update this conversation summary to include the new turn.\n"
    "Preserve all facts, numbers, names, IDs, and unresolved questions verbatim.\n"
    'Return JSON only: {{"summary": "...", "pinned_facts": ["..."]}}\n\n'
    "Existing summary: {prev_summary}\n"
    "Existing pinned facts: {prev_pinned_facts}\n"
    "New turn: {new_turn}\n"
)


class _ModelUnavailableError(Exception):
    pass


# ── Token counting ────────────────────────────────────────────────────────────

def _count_tokens(messages: list[dict]) -> int:
    """Character ÷ 4 proxy. Under-counts (late-trigger direction) — documented in spec §5.1."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", ""))
    return total // 4


# ── Chain hashing ─────────────────────────────────────────────────────────────

def _chain_hash(turns: list[dict]) -> str:
    """Chained SHA-256 so each prefix has a unique hash."""
    h = ""
    for t in turns:
        h = hashlib.sha256((h + json.dumps(t, sort_keys=True)).encode()).hexdigest()
    return h


# ── Pinned facts ──────────────────────────────────────────────────────────────

def _extract_pinned_from_turns(turns: list[dict]) -> list[str]:
    """Regex fallback: extract explicit-preference sentences from turn content."""
    text_parts = []
    for m in turns:
        content = m.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
    text = " ".join(text_parts)

    sentences = re.split(r"[.!?]+", text)
    facts: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if _PINNED_PATTERNS.search(sentence):
            clipped = sentence[:80]
            key = clipped.lower()
            if key not in seen:
                seen.add(key)
                facts.append(clipped)
            if len(facts) >= 10:
                break
    return facts


def _make_pinned_msg(pinned_facts: list[str]) -> Optional[dict]:
    if not pinned_facts:
        return None
    content = "User preferences:\n" + "\n".join(f"- {f}" for f in pinned_facts)
    return {"role": "system", "content": content}


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_summary_response(text: str) -> tuple[Optional[str], Optional[list[str]]]:
    """Return (summary, pinned_facts) or (None, None) on any parse failure."""
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        result = json.loads(stripped)
        if not isinstance(result, dict):
            return None, None
        summary = result.get("summary")
        pinned_facts = result.get("pinned_facts")
        if not isinstance(summary, str):
            return None, None
        if not isinstance(pinned_facts, list):
            return None, None
        if not all(isinstance(f, str) for f in pinned_facts):
            return None, None
        return summary, pinned_facts
    except Exception:
        return None, None


# ── Cheap model selection ─────────────────────────────────────────────────────

def _distill_model(req: GatewayRequest, settings) -> Optional[str]:
    return settings.distill_model.get(req.route) or _DISTILL_DEFAULTS.get(req.route)


# ── Cheap model calls ─────────────────────────────────────────────────────────

async def _call_cheap_model(req: GatewayRequest, settings, prompt: str) -> str:
    """Call cheap model with a user-message prompt. Raises _ModelUnavailableError on 404."""
    model_name = _distill_model(req, settings)
    synth = GatewayRequest(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        stream=False,
        max_tokens=2048,
        temperature=0.0,
        tools=[],
        route=req.route,
        raw_headers=req.raw_headers,
        extra={},
    )
    try:
        resp = await call_upstream(synth, settings, transport=_transport)
    except UpstreamError as e:
        if e.status_code == 404:
            raise _ModelUnavailableError() from e
        raise
    return resp.content


async def _full_summarize(req: GatewayRequest, settings, older_turns: list[dict]) -> tuple[str, list[str]]:
    prompt = _FULL_PROMPT + json.dumps(older_turns, indent=2)
    text = await _call_cheap_model(req, settings, prompt)
    summary, pinned_facts = _parse_summary_response(text)
    if summary is None:
        raise ValueError("summary parse failed")
    return summary, pinned_facts  # type: ignore[return-value]


async def _incremental_summarize(
    req: GatewayRequest,
    settings,
    prev_summary: str,
    prev_pinned: list[str],
    new_turn: dict,
) -> tuple[str, list[str]]:
    prompt = _INCREMENTAL_TEMPLATE.format(
        prev_summary=prev_summary,
        prev_pinned_facts=json.dumps(prev_pinned),
        new_turn=json.dumps(new_turn),
    )
    text = await _call_cheap_model(req, settings, prompt)
    summary, pinned_facts = _parse_summary_response(text)
    if summary is None:
        raise ValueError("summary parse failed")
    return summary, pinned_facts  # type: ignore[return-value]


# ── Summary cache ─────────────────────────────────────────────────────────────

def _lookup_summary(db_path, turns_hash: str) -> Optional[tuple[str, list[str]]]:
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT summary, pinned_facts FROM cache_summary WHERE turns_hash = ? AND expires_at > ?",
            (turns_hash, time.time()),
        ).fetchone()
    finally:
        con.close()
    if row:
        return row[0], json.loads(row[1])
    return None


def _store_summary(
    db_path,
    turns_hash: str,
    parent_hash: str,
    summary: str,
    pinned_facts: list[str],
    ttl: int,
) -> None:
    now = time.time()
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """INSERT OR REPLACE INTO cache_summary
               (turns_hash, parent_hash, summary, pinned_facts, expires_at, ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (turns_hash, parent_hash, summary, json.dumps(pinned_facts), now + ttl, now),
        )
        con.commit()
    finally:
        con.close()


# ── Top-K retrieval ───────────────────────────────────────────────────────────

def _get_or_store_turn_embedding(db_path, turn: dict, ttl: int):
    """Return a float32 numpy array for this turn, computing and storing if needed."""
    from tokengate.layers.semantic_cache import embed
    turn_text = json.dumps(turn, sort_keys=True)
    turn_hash = hashlib.sha256(turn_text.encode()).hexdigest()

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT embedding FROM distiller_turns WHERE turn_hash = ? AND expires_at > ?",
            (turn_hash, time.time()),
        ).fetchone()
    finally:
        con.close()

    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float32).copy()

    emb = np.array(embed(turn_text), dtype=np.float32)
    emb_bytes = emb.tobytes()
    now = time.time()
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """INSERT OR REPLACE INTO distiller_turns
               (turn_hash, turn_text, embedding, expires_at, ts)
               VALUES (?, ?, ?, ?, ?)""",
            (turn_hash, turn_text, emb_bytes, now + ttl, now),
        )
        con.commit()
    finally:
        con.close()

    return emb


def _retrieve_top_k(db_path, req: GatewayRequest, older_turns: list[dict], k: int, ttl: int) -> list[dict]:
    """Return up to k turns from older_turns most similar to the last user message."""
    from tokengate.layers.semantic_cache import embed

    query = ""
    for msg in reversed(req.messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                query = content
            elif isinstance(content, list):
                query = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    if not query:
        return []

    query_emb = np.array(embed(query), dtype=np.float32)

    scored: list[tuple[float, dict]] = []
    for turn in older_turns:
        turn_emb = _get_or_store_turn_embedding(db_path, turn, ttl)
        score = float(np.dot(query_emb, turn_emb))
        scored.append((score, turn))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


# ── Reconstruction validation ─────────────────────────────────────────────────

def _validate_reconstruction(messages: list[dict], had_system: bool) -> None:
    if not messages:
        raise ValueError("empty messages after reconstruction")
    if had_system and messages[0].get("role") != "system":
        raise ValueError("system prompt was lost during reconstruction")


# ── Partial path ──────────────────────────────────────────────────────────────

def _apply_partial(
    ctx: LayerContext,
    system_msgs: list[dict],
    older_turns: list[dict],
    recent_turns: list[dict],
    had_system: bool,
    reason: str,
) -> None:
    """Inject pinned facts + verbatim older_turns. Logs 'skip' with given reason."""
    pinned = _extract_pinned_from_turns(older_turns)
    pinned_msg = _make_pinned_msg(pinned)
    partial_msgs = (
        system_msgs
        + ([pinned_msg] if pinned_msg else [])
        + older_turns
        + recent_turns
    )
    try:
        _validate_reconstruction(partial_msgs, had_system)
    except Exception:
        ctx.decisions.append(LayerDecision("distiller", "skip", {"reason": "reconstruct_malformed"}))
        return
    ctx.request.messages = partial_msgs
    ctx.decisions.append(LayerDecision("distiller", "skip", {"reason": reason}))


# ── Main apply ────────────────────────────────────────────────────────────────

async def apply(ctx: LayerContext) -> LayerContext:
    req = ctx.request
    settings = ctx.settings

    # 1. Count tokens
    total_tokens = _count_tokens(req.messages)

    # 2. Threshold check
    if total_tokens < settings.distill_threshold_tokens:
        ctx.decisions.append(LayerDecision("distiller", "skip", {"reason": "below_threshold"}))
        return ctx

    # 3. Split messages
    system_msgs = [m for m in req.messages if m.get("role") == "system"]
    other_msgs = [m for m in req.messages if m.get("role") != "system"]
    keep = settings.distill_keep_recent_turns
    recent_turns = other_msgs[-keep:] if keep > 0 else []
    older_turns = other_msgs[:-keep] if keep > 0 else other_msgs

    # 4. Nothing to summarize
    if not older_turns:
        ctx.decisions.append(LayerDecision("distiller", "skip", {"reason": "no_older_turns"}))
        return ctx

    had_system = len(system_msgs) > 0
    turns_hash = _chain_hash(older_turns)
    parent_hash = _chain_hash(older_turns[:-1])
    db_path = settings.db_path
    ttl = settings.distill_ttl_seconds

    # 5. Unsupported provider → partial path immediately
    if _distill_model(req, settings) is None:
        _apply_partial(ctx, system_msgs, older_turns, recent_turns, had_system, "unsupported_provider")
        return ctx

    # 6. Try to get summary (cache hit, incremental, or full)
    summary: Optional[str] = None
    pinned_facts: Optional[list[str]] = None
    cache_hit = False
    incremental = False

    try:
        cached = _lookup_summary(db_path, turns_hash)
        if cached is not None:
            summary, pinned_facts = cached
            cache_hit = True
        else:
            parent_cached = _lookup_summary(db_path, parent_hash)
            if parent_cached is not None:
                prev_summary, prev_pinned = parent_cached
                summary, pinned_facts = await _incremental_summarize(
                    req, settings, prev_summary, prev_pinned, older_turns[-1]
                )
                incremental = True
                _store_summary(db_path, turns_hash, parent_hash, summary, pinned_facts, ttl)
            else:
                summary, pinned_facts = await _full_summarize(req, settings, older_turns)
                _store_summary(db_path, turns_hash, parent_hash, summary, pinned_facts, ttl)

    except _ModelUnavailableError:
        _apply_partial(ctx, system_msgs, older_turns, recent_turns, had_system, "distill_model_unavailable")
        return ctx
    except Exception:
        _apply_partial(ctx, system_msgs, older_turns, recent_turns, had_system, "distill_failed")
        return ctx

    # 7. Top-K retrieval (only when embedder available and numpy present)
    top_k_msgs: list[dict] = []
    from tokengate.layers.semantic_cache import can_embed
    if can_embed() and _HAVE_NUMPY:
        try:
            top_k_msgs = _retrieve_top_k(db_path, req, older_turns, settings.distill_top_k, ttl)
        except Exception:
            top_k_msgs = []

    # Assert older_turns and recent_turns are non-overlapping by object identity
    # (sliced at different positions, so no shared dict objects)
    assert not any(t is rt for t in older_turns for rt in recent_turns), (
        "older_turns and recent_turns overlap — check slicing logic"
    )

    # 8. Build final message list
    pinned_msg = _make_pinned_msg(pinned_facts)
    summary_msg = {"role": "system", "content": f"Conversation summary:\n{summary}"}

    new_messages = (
        system_msgs
        + ([pinned_msg] if pinned_msg else [])
        + [summary_msg]
        + top_k_msgs
        + recent_turns
    )

    # 9. Validate before assigning
    try:
        _validate_reconstruction(new_messages, had_system)
    except Exception:
        ctx.decisions.append(LayerDecision("distiller", "skip", {"reason": "reconstruct_malformed"}))
        return ctx

    new_token_count = _count_tokens(new_messages)
    ctx.request.messages = new_messages
    ctx.decisions.append(LayerDecision("distiller", "applied", {
        "tokens_in": total_tokens,
        "tokens_out": new_token_count,
        "cache_hit": cache_hit,
        "incremental": incremental,
        "pinned_facts": len(pinned_facts) if pinned_facts else 0,
        "top_k_retrieved": len(top_k_msgs),
    }))

    return ctx
