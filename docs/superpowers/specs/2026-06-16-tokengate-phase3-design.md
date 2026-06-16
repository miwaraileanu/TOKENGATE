# TokenGate Phase 3 — Distiller + Budgeter Design

## Overview

Phase 3 activates two pipeline layers that have been stubs since Phase 1:

- **Distiller** (`layers/distiller.py`) — compresses long conversation history into a coherent rolling summary, preserving facts and recent turns verbatim.
- **Budgeter** (`layers/budgeter.py`) — detects request type and injects a conservative `max_tokens` ceiling when the client didn't set one.

Both layers *transform the request* rather than short-circuit it (unlike the cache layers). They run on every non-cached request, in pipeline order:

```
exact_cache → semantic_cache → distiller → compressor → router → budgeter
```

Phase 3 does not touch `compressor.py` (Phase 5) or `router.py` (Phase 4).

---

## 1. Motivation

### 1.1 Distiller

Multi-turn conversations grow unboundedly. By turn 20–30, the full history can consume 10–20k tokens on every request — paying for context the model mostly ignores. Naïve truncation drops old turns silently, breaking coreferences ("the second option we discussed") and losing user preferences stated early. The distiller produces a **coherent compressed history**: a rolling abstractive summary of old turns, with verbatim retention of recent turns and explicit extraction of user preferences as "pinned facts."

### 1.2 Budgeter

Many clients (especially exploratory or agent-driven ones) send requests with no `max_tokens`. Models then generate until their internal stopping criterion, sometimes producing very long outputs for simple requests. The budgeter injects a **generous ceiling** per detected request type — not to shorten answers, but to cap runaway generation. The defaults are chosen so normal replies never hit the ceiling; the cap only fires on pathological outputs.

---

## 2. Pipeline Interaction

**Order of effects within a single request:**

1. Distiller mutates `ctx.request.messages` (compresses history).
2. Budgeter then reads `ctx.request.messages` for type detection.

**Budgeter detection scope:** The budgeter scans only `user`-role messages. The distiller's summary blob is injected as a `system`-role message and is structurally invisible to the budgeter. The current ask is always in `recent_turns` (kept verbatim as user messages), so trigger words are present when they matter.

**Token accounting:**

- `tokens_in_raw` = pre-distillation token count, stored in the distiller's `LayerDecision.detail["tokens_in"]`. The server reads this when writing the analytics row.
- `tokens_in_final` = `upstream_resp.tokens_in` (what was actually sent to upstream after distillation). Already tracked.
- Budgeter output savings are real but unmeasurable (the model's counterfactual output length is unknown). Budgeter contributes `est_saved_usd = 0`. Logged as "applied" with the injected cap; no savings claimed.

---

## 3. New Database Tables

Added to `analytics/db.py` `_SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS cache_summary (
    turns_hash   TEXT PRIMARY KEY,
    parent_hash  TEXT,
    summary      TEXT NOT NULL,
    pinned_facts TEXT NOT NULL DEFAULT '[]',   -- JSON array of strings
    expires_at   REAL NOT NULL,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cache_summary_parent ON cache_summary(parent_hash);

CREATE TABLE IF NOT EXISTS distiller_turns (
    turn_hash    TEXT PRIMARY KEY,
    turn_text    TEXT NOT NULL,
    embedding    BLOB,
    expires_at   REAL NOT NULL,
    ts           REAL NOT NULL
);
```

**Privacy & eviction:**

Both tables contain raw or summarized conversation content — they are **sensitive tables** alongside `cache_exact` and `cache_semantic`. TTL eviction (`DELETE WHERE expires_at < now()`) runs at startup. Both tables are cleared by `rait cache clear`. Both are included in the `TOKENGATE_ENCRYPT_KEY` at-rest encryption scope when that is wired (Phase 5 hardening).

Default TTL: `distill.ttl_seconds` (default 86400, same as `cache_exact_ttl`).

---

## 4. New Configuration Fields

Added to `Settings.__init__` from the `distill:` and `budget:` blocks of `tokengate.yaml`:

```yaml
distill:
  threshold_tokens: 6000       # trigger when history token count exceeds this
  keep_recent_turns: 4         # turns kept verbatim (not summarized)
  top_k: 3                     # old turns retrieved by similarity and injected verbatim
  ttl_seconds: 86400           # TTL for cache_summary and distiller_turns rows
  model:
    anthropic: claude-haiku-4-5
    openai: gpt-4o-mini

budget:
  chat: 1024
  extraction: 512
  code: 2048
  long_form: 4096
  extraction_instruction: "Answer with the requested data only, no preamble."
  extraction_instruction_enabled: true
```

`distill.model.anthropic` and `distill.model.openai` are each independently overridable. If only one is set, the other keeps its default.

---

## 5. Distiller Layer

### 5.1 Trigger

Count tokens in `ctx.request.messages` using `tokens.py` utilities. Fallback: `len(content) // 4` (character ÷ 4 proxy). The proxy **under-counts** (4-char-per-token assumption skews low), so distillation triggers slightly late — late is harmless (a few extra tokens sent to the main model). Early distillation would mean unnecessary LLM calls. Direction documented in code.

If `total_tokens < settings.distill_threshold_tokens`:
→ `LayerDecision("distiller", "skip", {"reason": "below_threshold"})`

### 5.2 Message Splitting

```python
system_msgs  = [m for m in req.messages if m["role"] == "system"]
other_msgs   = [m for m in req.messages if m["role"] != "system"]
recent_turns = other_msgs[-settings.distill_keep_recent_turns:]
older_turns  = other_msgs[:-settings.distill_keep_recent_turns]
```

If `len(older_turns) == 0`: nothing to summarize → skip with `{"reason": "no_older_turns"}`.

### 5.3 Chain Hash

```python
def _chain_hash(turns: list[dict]) -> str:
    h = ""
    for t in turns:
        h = sha256((h + json.dumps(t, sort_keys=True)).encode()).hexdigest()
    return h
```

`turns_hash = _chain_hash(older_turns)`
`parent_hash = _chain_hash(older_turns[:-1])` (hash without the last "newly-old" turn)

### 5.4 Summary Cache Lookup

1. `SELECT summary, pinned_facts FROM cache_summary WHERE turns_hash = ? AND expires_at > now()` (using `turns_hash`)
   → **hit**: use cached summary and pinned_facts. No model call.
2. `SELECT summary, pinned_facts FROM cache_summary WHERE turns_hash = ? AND expires_at > now()` (using `parent_hash`)
   → **incremental hit**: call cheap model with `(prev_summary, prev_pinned_facts, new_turn = older_turns[-1])`.
3. Neither → **full summarize**: call cheap model with all `older_turns`.

**One-parent incremental limit:** The parent check looks back exactly one step (`older_turns[:-1]`). If two turns became "old" since the last summary was cached (e.g. distillation didn't trigger last request because under threshold), both the full and parent hashes miss and the layer falls back to a full re-summarize. This is correct and documented — the cost is one extra model call in an uncommon case. Walking N parents is deferred.

### 5.5 Cheap Model Selection

```python
_DISTILL_DEFAULTS = {
    "anthropic": "claude-haiku-4-5",
    "openai":    "gpt-4o-mini",
}

def _distill_model(req, settings) -> str | None:
    """Return the distill model name for this request's provider, or None if unsupported."""
    overrides = settings.distill_model  # dict with optional "anthropic"/"openai" keys
    return overrides.get(req.route) or _DISTILL_DEFAULTS.get(req.route)
```

If `_distill_model` returns `None` (unknown provider): skip with `LayerDecision("distiller", "skip", {"reason": "unsupported_provider"})`. Pinned facts regex still runs and is injected.

The summary call is made via existing `provider.py` (`call_upstream`), targeting the same provider as the main request, reusing the gateway's API key. No new HTTP client.

### 5.6 Summary Prompt

**Full summarize:**
```
Summarize this conversation. Preserve all facts, decisions, and user preferences.
Reproduce numbers, names, IDs, and unresolved questions verbatim.
Be concise.
Return JSON only: {"summary": "...", "pinned_facts": ["..."]}
where pinned_facts is a list of explicit user preferences, constraints, or
instructions (e.g. "my name is X", "always use Python 3", "never truncate output").
```

**Incremental update:**
```
Update this conversation summary to include the new turn.
Preserve all facts, numbers, names, IDs, and unresolved questions verbatim.
Return JSON only: {"summary": "...", "pinned_facts": ["..."]}

Existing summary: {prev_summary}
Existing pinned facts: {prev_pinned_facts}
New turn: {new_turn}
```

### 5.7 JSON Response Parsing

`_parse_summary_response(text: str) -> tuple[str | None, list[str] | None]`

1. Strip markdown fences: `re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()`
2. `json.loads(stripped)` in `try/except`
3. Validate: result must be a `dict` with `summary: str` and `pinned_facts: list[str]`
4. **On any failure** (exception, wrong type, missing key): return `(None, None)`

`(None, None)` triggers the fail-safe (§5.9). Tests must cover:
- (a) clean JSON → parsed correctly
- (b) fence-wrapped JSON → stripped and parsed correctly
- (c) truncated/invalid JSON → `(None, None)`
- (d) JSON missing `pinned_facts` key → `(None, None)`

### 5.8 Pinned Facts Regex Fallback

Used only when the model call fails or `_parse_summary_response` returns `(None, None)`.

Patterns (explicit preference phrasing only):
```python
_PINNED_PATTERNS = re.compile(
    r"\b(my name is|I prefer|I need you to|always use|never use|call me)\b",
    re.IGNORECASE,
)
```

Extract the containing sentence (split on `.!?` boundaries). Caps: max 10 items, max 80 chars each, deduped (case-insensitive).

### 5.9 Fail-Safe

On **any** failure in the model call or JSON parsing (exception, timeout, or `(None, None)` from the parser):

1. Extract pinned facts via regex from `older_turns` (regex fallback always runs, regardless of model failure).
2. Reconstruct messages **without compression**: `system_msgs + [pinned_facts_msg?] + older_turns + recent_turns`. The `older_turns` are passed verbatim — no summary, no top-K injection.
3. Log `LayerDecision("distiller", "skip", {"reason": "distill_failed"})` for general errors, or `{"reason": "distill_model_unavailable"}` for HTTP 404/model-not-found.

This is *partial distillation*: pinned facts are always extracted and injected (they must never be skipped), but the history is not compressed. The main model receives the full conversation with a pinned-facts header.

For `unsupported_provider` (§5.5): same partial path — pinned facts injected, `older_turns` verbatim.

The reconstruct validation from §5.11 still applies to the partial path. If validation fails on the partial reconstruction: return `ctx` truly unchanged (no modification at all).

Never let a distiller failure block or corrupt the main request. Saving tokens is never worth degrading context.

### 5.10 Top-K Retrieval

If `can_embed()` is True:

1. Embed `query = last user message content` in `ctx.request.messages`
2. Look up or compute embedding for each `older_turn` (stored in `distiller_turns`; embed on first encounter, store BLOB)
3. Score by cosine similarity; take top `settings.distill_top_k`
4. `older_turns` and `recent_turns` are non-overlapping by construction (sliced at `[-keep_recent_turns:]`). Retrieved turns are drawn from `older_turns` only. No deduplication needed structurally; assert and test.

If `can_embed()` is False: skip retrieval silently. Summary + pinned facts still injected.

### 5.11 Message Reconstruction

```python
new_messages = (
    system_msgs
    + ([pinned_facts_msg] if pinned_facts else [])
    + [summary_msg]          # {"role": "system", "content": "Conversation summary:\n{summary}"}
    + top_k_msgs             # verbatim older turns, if any retrieved
    + recent_turns
)
```

**Assignment pattern:** Compute `new_messages` first, validate, then assign. Never mutate `ctx.request.messages` before validation succeeds:

```python
new_messages = [...]      # compute
assert len(new_messages) > 0  # validate
if had_system and new_messages[0]["role"] != "system":
    raise ValueError("system prompt lost")
ctx.request.messages = new_messages   # assign only after passing validation
```

If validation fails → leave `ctx.request.messages` untouched (the original list was never modified) + `LayerDecision("distiller", "skip", {"reason": "reconstruct_malformed"})`. Fail loud in tests (assert raises); fail safe in production (the except branch fires).

### 5.12 LayerDecision

```python
LayerDecision("distiller", "applied", {
    "tokens_in":       original_token_count,   # used by server as tokens_in_raw
    "tokens_out":      new_token_count,         # tokens after distillation
    "cache_hit":       bool,                    # True if summary came from cache
    "incremental":     bool,                    # True if incremental update path
    "pinned_facts":    len(pinned_facts),
    "top_k_retrieved": len(top_k_msgs),
})
```

---

## 6. Budgeter Layer

### 6.1 Skip Conditions

- `req.max_tokens` is already set by client → `LayerDecision("budgeter", "skip", {"reason": "client_set_max_tokens"})`. Absolute. Never override.

### 6.2 Request Type Detection

Evaluate against all `user`-role message content concatenated. First match wins.

| Type | Trigger |
|---|---|
| `extraction` | `\b(json\|extract\|list only\|output only\|parse\|return only)\b` (case-insensitive) |
| `code` | Fenced block (` ``` `) in prompt **OR** imperative `\b(write\|implement\|fix\|refactor\|debug)\b` within 10 words of code noun `\b(function\|class\|script\|program\|method\|api\|endpoint)\b` |
| `long_form` | `\b(essay\|blog\|article\|explain in detail\|write a detailed\|comprehensive)\b` |
| `chat` | Default fallback |

**Low-confidence skip:** If total word count < 10 AND detected type is not `extraction` → `LayerDecision("budgeter", "skip", {"reason": "type_uncertain"})`. Short prompts ("hi", "thanks", "yes") add negligible generation risk; skipping avoids wrong classification in logs.

The `chat` default (1024) is a ceiling chosen so normal conversational replies never hit it. Its purpose is to cap runaway generation, not to shorten answers. Do not lower this without measuring real p99 chat reply lengths.

### 6.3 max_tokens Injection

Inject `req.max_tokens = settings.budget_table[detected_type]`.

Defaults: `chat→1024, extraction→512, code→2048, long_form→4096`. All configurable in `tokengate.yaml`.

### 6.4 Extraction Instruction

For `extraction` type only, when `settings.budget_extraction_instruction_enabled` is True:

Check the existing system prompt (if any) for conflicting verbosity directives:
```python
_CONFLICT_RE = re.compile(r"\b(detailed|thorough|explain fully|verbose)\b", re.IGNORECASE)
```

- **No system prompt:** Create one: `{"role": "system", "content": extraction_instruction}`.
- **System prompt exists, no conflict:** Append `"\n" + extraction_instruction` to its content.
- **System prompt exists, conflict detected:** Skip. Log `{"instruction_added": false, "reason": "conflict_directive"}`.

The instruction is appended, never replacing the client's content.

### 6.5 LayerDecision

```python
LayerDecision("budgeter", "applied", {
    "type":             detected_type,      # "chat"|"extraction"|"code"|"long_form"
    "trigger":          matched_pattern,    # the text pattern that fired, or "default"
    "max_tokens":       injected_value,
    "instruction_added": bool,
    "instruction_skip_reason": str | None,  # "conflict_directive" or None
})
```

`est_saved_usd` contribution: `0`. Output savings are real but unmeasurable; no value is claimed.

---

## 7. Tests

### 7.1 Distiller Tests (`tests/test_distiller.py`)

- **Trigger:** below threshold → skip decision logged
- **Trigger:** at/above threshold → applied
- **Chain hash:** same turns → same hash; different turns → different hash
- **Summary cache hit:** same older_turns twice → model called once
- **Incremental path:** one new turn added → incremental prompt used
- **Incremental miss (2+ new turns):** falls back to full re-summarize
- **JSON parser (a)–(d):** clean / fenced / invalid / missing key
- **Pinned facts model path:** model returns valid JSON with pinned_facts list
- **Pinned facts regex fallback:** fires on model failure; caps at 10/80chars/deduped
- **Model unavailable (404):** fail-safe, original messages unchanged, skip decision logged
- **Model timeout/exception:** fail-safe
- **Top-K retrieval:** injected verbatim from older_turns only
- **Deduplication assert:** older_turns and recent_turns non-overlapping
- **Reconstruct validation:** empty result → fallback; missing system → fallback
- **Pinned facts injected even when summary fails**
- **Provider routing:** OpenAI request → gpt-4o-mini; Anthropic request → claude-haiku-4-5
- **Both providers in same session:** each uses its own default model
- **Token accounting:** `tokens_in` in LayerDecision.detail equals pre-distillation count

### 7.2 Budgeter Tests (`tests/test_budgeter.py`)

- **Client set max_tokens:** skip decision, no injection
- **Each request type:** extraction/code/long_form/chat detected and correct ceiling injected
- **Code detection:** "explain how this function works" → not code; "write a function to parse" → code; fenced block → code; "I'm writing a script for a play" → not code
- **Long-form vs extraction:** "extract a JSON" → extraction even if long message
- **Low-confidence skip:** < 10 words, non-extraction → type_uncertain
- **Extraction instruction:** no system prompt → created; neutral system prompt → appended; conflicting system prompt → skipped with reason logged
- **Client set max_tokens → never overridden (test with all types)**
- **Budget table configurable:** custom value in settings → used over default

### 7.3 Integration Tests (`tests/test_proxy.py` additions)

- Request with long history → distiller fires → upstream receives fewer tokens
- Distiller fail-safe → original history reaches upstream
- Budgeter injects max_tokens on uncapped request
- Distiller `tokens_in_raw` recorded correctly in analytics row (separate from `tokens_in_final`)
- Budgeter `est_saved_usd` = 0 in analytics row

---

## 8. Limitations

- **One-parent incremental only.** If multiple turns become "old" between requests, the distiller falls back to a full re-summarize. Walking N parents deferred to a future patch.
- **Embedding retrieval optional.** If `sentence-transformers` is not installed, top-K retrieval is skipped silently. The summary is still injected; only the verbatim-old-turns retrieval step is absent.
- **Distiller adds latency.** A cache miss requires a cheap model call before the main model call. Cache hit rate grows quickly in steady-state usage (same conversation prefix cached). Cold start on first turn of a conversation always pays the latency cost.
- **Budgeter output savings are not measured.** The layer is logged as "applied" but contributes 0 to `est_saved_usd`. Accurate output savings would require counterfactual measurement; deferred.
- **Single process per data dir.** Inherited from Phase 2. `cache_summary` and `distiller_turns` are SQLite-backed and safe for single-process access. Concurrent writers not supported in Phase 3.
